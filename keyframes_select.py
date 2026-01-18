#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import math
import argparse
import subprocess
from typing import Tuple, Dict, Any, List

import numpy as np
from scipy.special import softmax

import torch


# =============== 全局加速开关（按需改 / 也可改成 argparse 参数） ===============
USE_FFPROBE = True          # 用 ffprobe 取 fps/帧数（更快，且这里只用元信息）
USE_FP16_FPS = True         # FPS 距离计算用 FP16（d=1024 且 m=2~8，一般很稳更快）
DISABLE_PRINT = True        # 大规模跑时建议关掉 print
# ============================================================================


def _log(*args, **kwargs):
    if not DISABLE_PRINT:
        print(*args, **kwargs)


# ----------------------------- video probe -----------------------------
def probe_video_ffprobe(video_path: str) -> Tuple[int, float]:
    """
    用 ffprobe 获取 total_frames 和 fps。比 VideoReader 轻很多（这里只用元信息）。
    若 nb_frames 取不到，则用 duration*fps 近似。
    """
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=avg_frame_rate,nb_frames,duration",
        "-of", "json",
        video_path
    ]
    out = subprocess.check_output(cmd)
    info = json.loads(out)["streams"][0]

    # fps
    afr = info.get("avg_frame_rate", "0/0")
    num, den = afr.split("/")
    num = float(num)
    den = float(den) if float(den) != 0 else 1.0
    fps = num / den if den != 0 else 0.0

    # total_frames
    nb = info.get("nb_frames", None)
    if nb is not None:
        try:
            total_frames = int(nb)
        except Exception:
            total_frames = 0
    else:
        duration = float(info.get("duration", 0.0))
        total_frames = int(max(0, math.floor(duration * fps)))

    return total_frames, fps


def probe_video(video_path: str) -> Tuple[int, float]:
    """
    优先 ffprobe；失败就 fallback 到 decord VideoReader。
    """
    if USE_FFPROBE:
        try:
            return probe_video_ffprobe(video_path)
        except Exception as e:
            _log(f"[WARN] ffprobe failed for {video_path}: {e}. Fallback to decord.")

    # fallback: decord
    from decord import VideoReader, cpu
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=8)
    total_frames = len(vr)
    fps = float(vr.get_avg_fps())
    return total_frames, fps


# ----------------------------- FPS sampling on GPU -----------------------------
@torch.no_grad()
def farthest_point_sampling_gpu(
    segment_feats_t: torch.Tensor,
    candidate_idx_np: np.ndarray,
    anchor_local: int,
    m: int
) -> List[int]:
    """
    segment_feats_t: (L, d) on CUDA (建议 float16/float32)
    candidate_idx_np: (C,) numpy int array (local indices) -- 传进来前最好已 unique/sort
    anchor_local: int
    m: 选择帧数（含 anchor）
    return: list[int] 选择的 local indices（含 anchor）
    """
    if m <= 0:
        return []

    device = segment_feats_t.device

    cand = torch.as_tensor(candidate_idx_np, device=device, dtype=torch.long)
    if cand.numel() == 0:
        return []

    # 确保 anchor 在候选里
    if not torch.any(cand == anchor_local):
        cand = torch.cat([cand, torch.tensor([anchor_local], device=device, dtype=torch.long)], dim=0)
        cand = torch.unique(cand)

    C = cand.numel()
    if C == 0:
        return []

    if C <= m:
        out = cand[:m].tolist()
        if anchor_local in out:
            out.remove(anchor_local)
            out = [anchor_local] + out
        return out[:m]

    X = segment_feats_t.index_select(0, cand)  # (C, d)

    anchor_pos = (cand == anchor_local).nonzero(as_tuple=False).item()

    selected = torch.zeros(C, device=device, dtype=torch.bool)
    selected[anchor_pos] = True

    x2 = (X * X).sum(dim=1)  # (C,)
    a = X[anchor_pos]        # (d,)
    a2 = (a * a).sum()
    min_dist = x2 + a2 - 2.0 * (X @ a)  # (C,)
    min_dist[selected] = -1.0

    for _ in range(m - 1):
        best = torch.argmax(min_dist).item()
        if min_dist[best].item() < 0:
            break
        selected[best] = True

        b = X[best]
        b2 = (b * b).sum()
        dist_to_b = x2 + b2 - 2.0 * (X @ b)
        min_dist = torch.minimum(min_dist, dist_to_b)
        min_dist[selected] = -1.0

    selected_local = cand[selected].tolist()
    if anchor_local in selected_local:
        selected_local.remove(anchor_local)
        selected_local = [anchor_local] + selected_local
    return selected_local[:m]


# ----------------------------- segment variance + softmax -----------------------------
def compute_segment_variance_with_softmax(
    feat_array: np.ndarray,
    num_segments: int = 16,
    T: float = 0.5
):
    """
    返回：
        segment_var:           (K, d) 每段每维方差
        segment_var_score:     (K,)   融合分数：softmax(mean_diff/T) + softmax(trace_cov/T)
        segment_var_softmax:   (K,)   对融合分数再 softmax(T) 后的权重
    """
    n, d = feat_array.shape
    K = num_segments

    if K <= 0:
        raise ValueError("num_segments 必须是正整数")

    base = n // K
    extra = n % K
    if base == 0:
        raise ValueError(f"帧数 {n} 太少，无法划分成 {K} 段。")

    lengths = [base + 1 if i < extra else base for i in range(K)]
    global_mean = feat_array.mean(axis=0)   # (d,)

    segment_var_list = []
    mean_diff_sq_list = []
    trace_cov_list = []

    start = 0
    for seg_len in lengths:
        end = start + seg_len
        segment = feat_array[start:end]

        seg_mean = segment.mean(axis=0) if seg_len > 0 else np.zeros(d, dtype=np.float32)
        mean_diff_sq = float(np.sum((global_mean - seg_mean) ** 2))

        if seg_len <= 1:
            var_per_dim = np.zeros(d, dtype=np.float32)
        else:
            var_per_dim = np.var(segment, axis=0, ddof=1)

        trace_cov = float(var_per_dim.sum())

        segment_var_list.append(var_per_dim)
        mean_diff_sq_list.append(mean_diff_sq)
        trace_cov_list.append(trace_cov)

        start = end

    segment_var = np.stack(segment_var_list, axis=0)
    mean_diff_sq_arr = np.asarray(mean_diff_sq_list, dtype=np.float32)
    trace_cov_arr = np.asarray(trace_cov_list, dtype=np.float32)

    mean_part = softmax(mean_diff_sq_arr / T)
    trace_part = softmax(trace_cov_arr / T)
    fused_score = mean_part + trace_part
    fused_weight = softmax(fused_score / T)

    return segment_var, fused_score.astype(np.float32), fused_weight.astype(np.float32)


def split_second_elements(data, n: int):
    """
    把 data 里每个元素的第二项取出来，然后均分成 n 段（front-heavy）。
    假设 data 的顺序和视频时间顺序对齐（1fps）。
    """
    second_elements = [item[1] for item in data]
    k, m = divmod(len(second_elements), n)
    return [
        second_elements[i * k + min(i, m):(i + 1) * k + min(i + 1, m)]
        for i in range(n)
    ]


def allocate_and_select_frames_from_softmax(
    feat_array: np.ndarray,
    segment_scores: np.ndarray,
    total_keep: int = 64
):
    """
    只负责根据片段权重/分数分配 m_i，不负责真正选帧。

    softmax 后预分配（floor + fractional 补齐）仍可能不足 total_keep。
    若不足，则把剩余预算按“片段分数从大到小”依次分配，循环直到分配满 total_keep
    （且不超过每段上限 frames_per_seg）。
    """
    n, d = feat_array.shape
    num_segments = len(segment_scores)

    frames_per_seg = n // num_segments
    if frames_per_seg == 0:
        raise ValueError(f"帧数 {n} 太少，无法划分成 {num_segments} 段。")

    max_total = frames_per_seg * num_segments
    if total_keep > max_total:
        total_keep = max_total

    usable_n = frames_per_seg * num_segments
    feat_array = feat_array[:usable_n]

    weights = segment_scores.astype(np.float64)
    w_sum = weights.sum()
    if w_sum <= 0:
        raise ValueError(f"segment_scores 的和为 {w_sum}，需要 > 0 才能分配。")
    weights /= w_sum

    raw_counts = weights * total_keep
    m_per_segment = np.floor(raw_counts).astype(int)

    # 1) fractional 补齐
    remaining = total_keep - m_per_segment.sum()
    if remaining > 0:
        fractional = raw_counts - m_per_segment
        order = np.argsort(-fractional)
        for idx in order:
            if remaining <= 0:
                break
            if m_per_segment[idx] < frames_per_seg:
                m_per_segment[idx] += 1
                remaining -= 1

    # 2) 按 segment_scores 从大到小轮流补齐
    remaining = total_keep - m_per_segment.sum()
    if remaining > 0:
        order_score = np.argsort(-segment_scores)
        while remaining > 0:
            changed = False
            for idx in order_score:
                if remaining <= 0:
                    break
                if m_per_segment[idx] < frames_per_seg:
                    m_per_segment[idx] += 1
                    remaining -= 1
                    changed = True
            if not changed:
                break

    # 返回候选索引（主流程可不用，但保留）
    selected_indices = []
    for seg_idx in range(num_segments):
        m_i = m_per_segment[seg_idx]
        if m_i <= 0:
            continue

        start = seg_idx * frames_per_seg
        end = start + frames_per_seg
        seg_len = end - start
        m_i = min(m_i, seg_len)

        if m_i >= seg_len:
            idx = np.arange(start, end)
        else:
            idx = np.linspace(start, end - 1, m_i, dtype=int)
        selected_indices.append(idx)

    if len(selected_indices) == 0:
        return m_per_segment, np.array([], dtype=int), np.empty((0, d), dtype=feat_array.dtype)

    selected_indices = np.concatenate(selected_indices, axis=0)

    if len(selected_indices) > total_keep:
        selected_indices = selected_indices[:total_keep]
    elif len(selected_indices) < total_keep and len(selected_indices) > 0:
        pad_num = total_keep - len(selected_indices)
        pad = np.random.choice(selected_indices, size=pad_num, replace=True)
        selected_indices = np.concatenate([selected_indices, pad], axis=0)

    selected_feats = feat_array[selected_indices]
    return m_per_segment, selected_indices, selected_feats


# ----------------------------- fast json loader -----------------------------
try:
    import orjson  # type: ignore

    def load_json_fast(p: str):
        with open(p, "rb") as f:
            return orjson.loads(f.read())
except Exception:
    def load_json_fast(p: str):
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)


# ----------------------------- argparse + dataset plumbing -----------------------------
def build_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name', type=str, default='longvideobench',
                        help='support longvideobench / videomme / mlvu / lvbench')
    parser.add_argument('--dataset_path', type=str, default='longvideobench',
                        help='your path of the dataset (root dir)')

    parser.add_argument('--num_segments', type=int, default=32)
    parser.add_argument('--total_keep', type=int, default=64)
    parser.add_argument('--tau', type=float, default=0.6)
    parser.add_argument('--alpha', type=float, default=0.5)

    parser.add_argument('--output_dir', type=str, default='',
                        help='if empty, auto build under dataset_path')

    # 可覆盖路径（不填则用默认规则/默认 hardcode）
    parser.add_argument('--anno_path', type=str, default='',
                        help='annotation file path (override)')
    parser.add_argument('--video_root', type=str, default='',
                        help='video root dir (override)')

    parser.add_argument('--feat_root_long', type=str, default='',
                        help='features root for long videos (override)')
    parser.add_argument('--feat_root_short', type=str, default='',
                        help='features root for short videos (override)')

    parser.add_argument('--sim_root_long', type=str, default='',
                        help='sim json root for long videos (override)')
    parser.add_argument('--sim_root_short', type=str, default='',
                        help='sim json root for short videos (override)')

    parser.add_argument('--disable_print', action='store_true',
                        help='disable print logs')

    return parser


def get_dataset_defaults(args) -> Dict[str, str]:
    """
    默认路径规则：
    - 标注 & 视频：尽量用 dataset_path 下的结构（常见）
    - 特征 & sim：默认沿用你原代码里的 ...（方便你“不改参数就能跑”）
      如果你想让所有东西都在 dataset_path 下，直接用 --feat_root_* / --sim_root_* 覆盖即可。
    """
    ds = args.dataset_name.lower()
    root = args.dataset_path

    if ds == 'longvideobench':
        return {
            "anno_path": args.anno_path or os.path.join(root, "lvb_val.json"),
            "video_root": args.video_root or os.path.join(root, "videos"),

            # 原代码默认
            "feat_root_long":  args.feat_root_long  or "frame_features/longvideobench/longvideobench/dinov3",
            "feat_root_short": args.feat_root_short or "frame_features/longvideobench_short/longvideobench/dinov3",
            "sim_root_long":   args.sim_root_long   or "longvideobench/blip",
            "sim_root_short":  args.sim_root_short  or "longvideobench_short/longvideobench/blip",
        }

    if ds == 'videomme':
        return {
            # 你原来是 /data/jsb/datasets/Video-MME/videomme/test_anns.csv
            "anno_path": args.anno_path or os.path.join(root, "videomme", "test_anns.csv"),
            "video_root": args.video_root or os.path.join(root, "data"),

            "feat_root_long":  args.feat_root_long  or "frame_features/videomme/videomme/dinov3",
            "feat_root_short": args.feat_root_short or "frame_features/videomme_short/videomme/dinov3",
            "sim_root_long":   args.sim_root_long   or "videomme/videomme/blip",
            "sim_root_short":  args.sim_root_short  or "videomme_short/videomme/blip",
        }

    if ds == 'mlvu':
        return {
            "anno_path": args.anno_path or os.path.join(root, "annotations", "multiple_choice.json"),
            "video_root": args.video_root or os.path.join(root, "video"),

            "feat_root_long":  args.feat_root_long  or "frame_features/mlvu/mlvu/dinov3",
            "feat_root_short": args.feat_root_short or "frame_features/mlvu/mlvu/dinov3",
            "sim_root_long":   args.sim_root_long   or "mlvu/mlvu/blip",
            "sim_root_short":  args.sim_root_short  or "mlvu/mlvu/blip",
        }

    if ds in ('lvbench', 'lvb'):
        return {
            "anno_path": args.anno_path or os.path.join(root, "qa_file.json"),
            "video_root": args.video_root or os.path.join(root, "videos"),

            "feat_root_long":  args.feat_root_long  or "frame_features/LVBench/LVB/dinov3",
            "feat_root_short": args.feat_root_short or "frame_features/LVBench/LVB/dinov3",
            "sim_root_long":   args.sim_root_long   or "lvbench",
            "sim_root_short":  args.sim_root_short  or "lvbench",
        }

    raise ValueError(f"Unknown dataset_name: {args.dataset_name}")


def load_annotations(dataset_name: str, anno_path: str) -> List[Dict[str, Any]]:
    ds = dataset_name.lower()

    if ds == 'longvideobench':
        with open(anno_path, "r", encoding="utf-8") as f:
            return json.load(f)

    if ds == 'mlvu':
        with open(anno_path, "r", encoding="utf-8") as f:
            return json.load(f)

    if ds == 'videomme':
        import pandas as pd
        df = pd.read_csv(anno_path)
        data = []
        for _, row in df.iterrows():
            option_str = str(row['options'])[1:-1]
            option_dict = re.findall(r'([A-D])\.\s([^\.?]+[\.?])', option_str)
            option = [f'{idx}. {answer}' for idx, answer in option_dict]
            answer_id = row['answer']
            answer = option_dict[ord(answer_id) - ord('A')][1] if option_dict else ""
            index2ans = {idx: ans for idx, ans in option_dict}
            data.append({
                'id': row["question_id"],
                'question': row["question"],
                'video_id': row["videoID"],
                'duration_group': row.get('duration', ''),
                'option': option,
                'answer_id': answer_id,
                'answer': answer,
                'index2ans': index2ans,
            })
        return data

    if ds in ('lvbench', 'lvb'):
        data = []
        with open(anno_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data.append(json.loads(line))
        return data

    raise ValueError(ds)


def get_ids_and_video_path(ds: str, sample: Dict[str, Any], video_root: str) -> Tuple[str, str, str]:
    """
    返回 (video_id, vid_id, video_path)
    - video_id：用于特征文件名
    - vid_id：用于 sim json / 输出文件名
    - video_path：视频路径
    """
    ds = ds.lower()

    if ds == 'longvideobench':
        video_id = sample['video_id']
        vid_id = sample['id']
        video_path = os.path.join(video_root, sample["video_path"])
        return video_id, vid_id, video_path

    if ds == 'videomme':
        video_id = sample["video_id"]
        vid_id = sample["id"]
        video_path = os.path.join(video_root, video_id + ".mp4")
        return video_id, vid_id, video_path

    if ds == 'mlvu':
        video_id = sample["video"]
        vid_id = sample["qid"]
        video_path = os.path.join(video_root, f"{sample['category']}/{sample['video']}")
        return video_id, vid_id, video_path

    if ds in ('lvbench', 'lvb'):
        video_id = sample["key"]
        vid_id = f"{sample['key']}_{sample['uid']}"
        video_path = os.path.join(video_root, f"{sample['video_path']}")
        return video_id, vid_id, video_path

    raise ValueError(ds)


def resolve_paths_for_sample(args, defaults: Dict[str, str], sample: Dict[str, Any], is_long: bool) -> Dict[str, str]:
    video_id, vid_id, video_path = get_ids_and_video_path(args.dataset_name, sample, defaults["video_root"])

    feat_root = defaults["feat_root_long"] if is_long else defaults["feat_root_short"]
    sim_root = defaults["sim_root_long"] if is_long else defaults["sim_root_short"]

    feat_path = os.path.join(feat_root, f"{video_id}.npy")
    sim_path = os.path.join(sim_root, f"{vid_id}.json")

    if args.output_dir:
        out_dir = args.output_dir
    else:
        out_dir = os.path.join(
            args.dataset_path,
            f"{args.dataset_name}_frame_{args.total_keep}_clip_{args.num_segments}_tau_{args.tau}_alpha_{args.alpha}trace_softmax"
        )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{vid_id}.txt")

    return {
        "video_id": video_id,
        "vid_id": vid_id,
        "video_path": video_path,
        "feat_path": feat_path,
        "sim_path": sim_path,
        "out_path": out_path,
    }


# ----------------------------- main -----------------------------
def main():
    global DISABLE_PRINT
    parser = build_parser()
    args = parser.parse_args()
    if args.disable_print:
        DISABLE_PRINT = True

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    num_segments = args.num_segments
    total_keep = args.total_keep
    tau = args.tau
    alpha = args.alpha

    defaults = get_dataset_defaults(args)

    if not os.path.exists(defaults["anno_path"]):
        raise FileNotFoundError(f"Annotation not found: {defaults['anno_path']}")

    data = load_annotations(args.dataset_name, defaults["anno_path"])

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA not available, this pipeline expects CUDA for FPS stage.")

    # 统计
    ok = 0
    skipped = 0
    error = 0

    from tqdm.auto import tqdm
    for _, item in enumerate(tqdm(data, total=len(data), desc=f"processing {args.dataset_name}")):
        try:
            # 先取视频路径 probe -> 判断 long/short
            video_id, vid_id, video_path = get_ids_and_video_path(args.dataset_name, item, defaults["video_root"])
            if not os.path.exists(video_path):
                skipped += 1
                continue

            total_frames, fps = probe_video(video_path)
            if fps <= 0:
                skipped += 1
                continue
            fps_int = int(round(fps)) if fps > 0 else 1
            duration_sec = total_frames / max(fps_int, 1)
            is_long = duration_sec > 128

            paths = resolve_paths_for_sample(args, defaults, item, is_long=is_long)
            output_path = paths["out_path"]

            if os.path.exists(output_path):
                skipped += 1
                continue

            if not os.path.exists(paths["feat_path"]):
                skipped += 1
                continue

            if not os.path.exists(paths["sim_path"]):
                skipped += 1
                continue

            # ===== 读特征 =====
            video_feats = np.load(paths["feat_path"])  # (n, 1024)
            if video_feats.ndim != 2 or video_feats.shape[0] == 0:
                skipped += 1
                continue

            video_feats = video_feats.astype(np.float32, copy=False)
            norms = np.linalg.norm(video_feats, axis=1, keepdims=True) + 1e-8
            video_feats_norm = video_feats / norms

            n_feats = len(video_feats_norm)
            if n_feats == 0:
                skipped += 1
                continue

            # ===== 第一阶段：视觉方差 softmax =====
            try:
                _, _, segment_var_softmax = compute_segment_variance_with_softmax(
                    video_feats_norm, num_segments=num_segments
                )
            except ValueError:
                skipped += 1
                continue

            # ===== 读取语义相似度 =====
            sim_data = load_json_fast(paths["sim_path"])

            # 按时间均分成 num_segments 段
            sim_split_data = split_second_elements(sim_data, num_segments)

            clip_sim = [float(np.mean(seg)) if len(seg) > 0 else 0.0 for seg in sim_split_data]
            T_sim = 0.5
            soft_sim = softmax(np.array(clip_sim, dtype=np.float32) / T_sim)

            # 融合分数（视觉方差 + 语义相似）
            clip_score = alpha * segment_var_softmax + (1.0 - alpha) * soft_sim

            # 分配每段保留帧数
            try:
                m_per_segment, _, _ = allocate_and_select_frames_from_softmax(
                    video_feats_norm, clip_score, total_keep=total_keep
                )
            except ValueError:
                skipped += 1
                continue

            # ===== 第二阶段：FPS 选帧（GPU） =====
            feats_cpu_t = torch.from_numpy(video_feats_norm).pin_memory()
            if USE_FP16_FPS:
                video_feats_t = feats_cpu_t.to(device, dtype=torch.float16, non_blocking=True)
            else:
                video_feats_t = feats_cpu_t.to(device, dtype=torch.float32, non_blocking=True)

            # short 情况：特征帧 -> 原视频帧的 linspace 映射
            sampled_frame_indices = None
            if duration_sec <= 128:
                sampled_frame_indices = np.linspace(0, total_frames - 1, num=n_feats, dtype=int)

            # 每个 sim 段在 1fps 序列中的起始偏移（秒）
            sim_lengths = [len(seg) for seg in sim_split_data]
            sim_offsets = np.cumsum([0] + sim_lengths[:-1])

            result_indices_total = []

            for j in range(num_segments):
                sim_segment = np.asarray(sim_split_data[j], dtype=np.float32)
                L_j = len(sim_segment)
                m_j = int(m_per_segment[j])

                if m_j <= 0 or L_j == 0:
                    continue

                seg_offset = int(sim_offsets[j])

                # anchor
                anchor_local = int(sim_segment.argmax())

                # ===== 阈值自适应筛候选 =====
                max_sim = float(sim_segment.max())
                threshold = max_sim if max_sim <= 0 else (tau * max_sim)

                MIN_THRESHOLD = 1e-6
                while True:
                    candidate_mask = sim_segment >= threshold
                    candidate_indices = np.where(candidate_mask)[0].astype(np.int32, copy=False)

                    if anchor_local not in candidate_indices:
                        candidate_indices = np.concatenate(
                            [np.array([anchor_local], dtype=np.int32), candidate_indices]
                        )

                    candidate_indices = np.unique(candidate_indices)

                    if (len(candidate_indices) >= m_j) or (threshold <= MIN_THRESHOLD) or (len(candidate_indices) == L_j):
                        break

                    threshold *= 0.5

                effective_m_j = min(m_j, len(candidate_indices))
                if effective_m_j <= 0:
                    continue

                # segment_feats_t：直接 slice（不再 H2D copy）
                seg_start = seg_offset
                seg_end = seg_offset + L_j
                if seg_start < 0 or seg_end > n_feats:
                    continue

                segment_feats_t = video_feats_t[seg_start:seg_end]

                selected_local_indices = farthest_point_sampling_gpu(
                    segment_feats_t=segment_feats_t,
                    candidate_idx_np=candidate_indices,
                    anchor_local=anchor_local,
                    m=effective_m_j
                )

                # 映射回原视频帧 index
                if duration_sec > 128:
                    for local_idx in selected_local_indices:
                        time_idx = seg_offset + local_idx
                        frame_idx_original = int(time_idx * fps_int)
                        if 0 <= frame_idx_original < total_frames:
                            result_indices_total.append(frame_idx_original)
                else:
                    assert sampled_frame_indices is not None
                    for local_idx in selected_local_indices:
                        feat_idx = seg_offset + local_idx
                        if 0 <= feat_idx < n_feats:
                            frame_idx_original = int(sampled_frame_indices[feat_idx])
                            result_indices_total.append(frame_idx_original)

            # 写出
            result_indices_total = sorted(set(result_indices_total))
            with open(output_path, "w", encoding="utf-8") as f:
                for frame_idx in result_indices_total:
                    f.write(str(frame_idx) + "\n")

            ok += 1

        except Exception as e:
            error += 1
            _log(f"[ERR] item failed: {e}")
            continue

    print(f"[DONE] ok={ok}, skipped={skipped}, error={error}")
    print(f"Output dir: {args.output_dir or os.path.join(args.dataset_path, f'{args.dataset_name}_frame_{args.total_keep}_clip_{args.num_segments}_tau_{args.tau}_alpha_{args.alpha}trace_softmax')}")


if __name__ == "__main__":
    main()
