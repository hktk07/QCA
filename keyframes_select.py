import os
import json
import math
import subprocess
from typing import Tuple

import numpy as np
from scipy.special import softmax

import torch
from decord import VideoReader, cpu
USE_FFPROBE = True        
USE_FP16_FPS = True         
# DISABLE_PRINT = True        
DISABLE_PRINT = False 

# def print(*args, **kwargs):
#     if not DISABLE_PRINT:
#         print(*args, **kwargs)

def probe_video_ffprobe(video_path: str) -> Tuple[int, float]:
    """
    用 ffprobe 获取 total_frames 和 fps。比 VideoReader 轻很多（你这里只用元信息）。
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
    if USE_FFPROBE:
        try:
            return probe_video_ffprobe(video_path)
        except Exception as e:
            print(f"[WARN] ffprobe failed for {video_path}: {e}. Fallback to decord.")
    # fallback: decord
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=8)
    total_frames = len(vr)
    fps = float(vr.get_avg_fps())
    return total_frames, fps

@torch.no_grad()
def farthest_point_sampling_gpu(
    segment_feats_t: torch.Tensor,
    candidate_idx_np: np.ndarray,
    anchor_local: int,
    m: int
):
    if m <= 0:
        return []
    device = segment_feats_t.device
    cand = torch.as_tensor(candidate_idx_np, device=device, dtype=torch.long)
    if cand.numel() == 0:
        return []

    if not torch.any(cand == anchor_local):
        cand = torch.cat([cand, torch.tensor([anchor_local], device=device, dtype=torch.long)], dim=0)
        cand = torch.unique(cand)
    C = cand.numel()
    if C == 0:
        return []

    if C <= m:
        return cand[:m].tolist()

    # 取候选特征 (C, d)
    X = segment_feats_t.index_select(0, cand)  # (C, d)

    # anchor 位置
    anchor_pos = (cand == anchor_local).nonzero(as_tuple=False).item()

    # bool mask 标记已选（比 python list 索引快）
    selected = torch.zeros(C, device=device, dtype=torch.bool)
    selected[anchor_pos] = True

    # min_dist: 到已选集合最小距离（L2^2）
    # dist^2 = ||x||^2 + ||a||^2 - 2 x·a
    x2 = (X * X).sum(dim=1)  # (C,)
    a = X[anchor_pos]        # (d,)
    a2 = (a * a).sum()
    min_dist = x2 + a2 - 2.0 * (X @ a)  # (C,)
    min_dist[selected] = -1.0  # 已选标记

    # 迭代选剩下的
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

    # 输出 selected 对应的 local idx
    selected_local = cand[selected].tolist()
    # 保证 anchor 在最前（可选）
    if anchor_local in selected_local:
        selected_local.remove(anchor_local)
        selected_local = [anchor_local] + selected_local
    return selected_local[:m]
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


def split_second_elements(data, n):
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


def redistribute_segment_quota(
    m_per_segment: np.ndarray,
    segment_capacities: np.ndarray,
    segment_scores: np.ndarray,
    target_total: int,
) -> np.ndarray:
    """
    将超过片段实际容量的配额收回，并重新分配给仍有剩余容量的片段。

    参数：
        m_per_segment:     初始分配结果，形状 (K,)
        segment_capacities:每段最多可选帧数，例如每段相似度数据长度 L_j
        segment_scores:    每段优先级分数，分数越高越优先获得回收配额
        target_total:      期望总帧数；若所有片段总容量不足，则最多分配总容量
    """
    m = np.asarray(m_per_segment, dtype=np.int64).copy()
    capacities = np.asarray(segment_capacities, dtype=np.int64)
    scores = np.asarray(segment_scores, dtype=np.float64)

    if not (len(m) == len(capacities) == len(scores)):
        raise ValueError("m_per_segment、segment_capacities、segment_scores 长度必须一致")

    capacities = np.maximum(capacities, 0)
    target_eff = min(int(target_total), int(capacities.sum()))

    # 先截断超出真实容量的分配，超出的部分自动形成待补额度。
    m = np.minimum(np.maximum(m, 0), capacities)
    remaining = target_eff - int(m.sum())

    if remaining <= 0:
        return m

    # 优先给分数高的片段补；同分时保持原片段顺序。
    order = np.argsort(-scores, kind="stable")

    # 批量补齐，避免逐帧 while。
    for idx in order:
        if remaining <= 0:
            break
        spare = int(capacities[idx] - m[idx])
        if spare <= 0:
            continue
        add = min(spare, remaining)
        m[idx] += add
        remaining -= add

    return m

def allocate_and_select_frames_from_softmax(
    feat_array: np.ndarray,
    segment_var_softmax: np.ndarray,
    total_keep: int = 64
):
    """
    只负责根据片段权重分配 m_i，不负责真正选帧。
    修改点：
      - 若按 fractional 分配后仍不足 total_keep，则按片段分数从高到低依次补齐（每段最多补到 frames_per_seg）。
      - 若视频可用帧数本身不足 total_keep，则最多只能分配 usable_n 帧（避免死循环）。
    """
    n, d = feat_array.shape
    num_segments = len(segment_var_softmax)

    frames_per_seg = n // num_segments
    if frames_per_seg == 0:
        raise ValueError(f"帧数 {n} 太少，无法划分成 {num_segments} 段。")

    usable_n = frames_per_seg * num_segments
    # print(feat_array)
    feat_array = feat_array[:usable_n]
    # print(feat_array)
    # exit(0)
    # print(feat)
    # 如果可用帧数都不够 total_keep，那最多只能分配 usable_n
    total_keep_eff = min(int(total_keep), int(usable_n))

    scores = segment_var_softmax.astype(np.float64)
    s_sum = scores.sum()
    if s_sum <= 0:
        raise ValueError(f"segment_var_softmax 的和为 {s_sum}，需要 > 0 才能分配。")
    weights = scores / s_sum

    raw_counts = weights * total_keep_eff
    m_per_segment = np.floor(raw_counts).astype(int)
    print('m_per_segment', sum(m_per_segment))
    # 先按 fractional 补
    remaining = total_keep_eff - int(m_per_segment.sum())
    print('remaining',remaining)
    if remaining > 0:
        fractional = raw_counts - m_per_segment
        order_frac = np.argsort(-fractional)
        for idx in order_frac:
            if remaining <= 0:
                break
            if m_per_segment[idx] < frames_per_seg:
                m_per_segment[idx] += 1
                remaining -= 1
    print('m_per_segment', m_per_segment, sum(m_per_segment))
    # 若仍不足：按“片段分数从高到低”补齐（你要求的逻辑）
    # 例如 sum=60, remaining=4 -> 给分数最高的4个片段各+1
    print('remaining',remaining)
    # exit(0)
    if remaining > 0:
        order_score = np.argsort(-scores)  # 分数从高到低
        # 可能一轮不够（比如 remaining > 段数，或部分段已到上限），就多轮循环
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
                # 所有段都到上限了，还剩余则无法再分配（理论上 total_keep_eff<=usable_n 时不会发生）
                break

    # 返回候选索引（这里主流程不使用，但保留）
    selected_indices = []
    for seg_idx in range(num_segments):
        m_i = int(m_per_segment[seg_idx])
        if m_i <= 0:
            # print(2222)
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

    if len(selected_indices) > total_keep_eff:
        selected_indices = selected_indices[:total_keep_eff]
    elif len(selected_indices) < total_keep_eff and len(selected_indices) > 0:
        pad_num = total_keep_eff - len(selected_indices)
        pad = np.random.choice(selected_indices, size=pad_num, replace=True)
        selected_indices = np.concatenate([selected_indices, pad], axis=0)

    selected_feats = feat_array[selected_indices]
    return m_per_segment, selected_indices, selected_feats

if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    error = 0
    correct = 0
    num_segments = 16   # segment number
    total_keep = 64    # keyframes number
    gemma = 0.5        # gemma in stage2 controls the size of the candidate pool by filtering frames whose relevance is below a fraction of the anchor score
    beta = 0.5        # beta in stage1 balance M_s and D_s
    softmax_tau = 1
    dataset = 'videomme'  # longvideobench/videomme/mlvu/lvbench
    output_dir = f'{dataset}/{dataset}/blip_test/frame_{total_keep}_clip_{num_segments}_gemma_{gemma}_beta_{beta}_softmax_tau_{softmax_tau}'
    os.makedirs(output_dir, exist_ok=True)
    if dataset == 'longvideobench':
        path = '/datasets/LongVideoBench/lvb_val.json'
        with open(path, 'r') as f:
            data = json.load(f)
    elif dataset == 'mlvu':
        path = '/datasets/MLVU/annotations/multiple_choice.json'
        with open(path, 'r') as f:
            data = json.load(f)
    elif dataset == 'videomme':
        import pandas as pd
        import re
        path = '/datasets/Video-MME/videomme/test_anns.csv'
        df = pd.read_csv(path)
        data = []
        for _, row in df.iterrows():
            option_str = str(row['options'])[1:-1]
            option_dict = re.findall(r'([A-D])\.\s([^\.?]+[\.?])', option_str)
            option = [f'{idx}. {answer}' for idx, answer in option_dict]
            answer_id = row['answer']
            answer = option_dict[ord(answer_id) - ord('A')][1]
            index2ans = {idx: ans for idx, ans in option_dict}
            data.append({
                'id': row["question_id"],
                'question': row["question"],
                'video_id': row["videoID"],
                'duration_group': row['duration'],
                'option': option,
                'answer_id': answer_id,
                'answer': answer,
                'index2ans': index2ans,
            })
    elif dataset == 'lvbench':
        data = []
        path = '/datasets/LVBench/qa_file.json'
        with open(path, 'r') as f:
            for line in f:
                data.append(json.loads(line))
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    # JSON 解析加速：优先 orjson（没有就退回 json）
    def load_json(p):
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
    from tqdm.auto import tqdm
    for idx, item in enumerate(tqdm(data, total=len(data), desc=f"processing {dataset}")): 
        # 路径解析
        if dataset == 'longvideobench':
            video_root = '/datasets/LongVideoBench/videos'
            # short_video_path = item["video_path"]
            video_path = os.path.join(video_root, item["video_path"])
            video_id = item['video_id']
            if '@' in video_id:
                # video_id = video_id.split('-')[1]
                video_id = video_id.replace('@', '')
                print(video_id)
                # exit(0)
            qid = item['id'].replace('@', '')
            if qid[0]=='_':
                qid = qid[1:]
        elif dataset == 'videomme':
            video_root = '/datasets/Video-MME/data'
            video_path = os.path.join(video_root, item["video_id"] + '.mp4')
            video_id = item['video_id']
            qid = item['id']
        elif dataset == 'mlvu':
            video_path = os.path.join('/datasets/MLVU/video', f"{item['category']}/{item['video']}")
            video_id = item['video']
            qid = item['qid']
        elif dataset == 'lvbench':
            video_path = os.path.join('/datasets/LVBench/videos', f"{item['video_path']}")
            video_id = item['key']
            qid = f"{item['key']}_{item['uid']}"
        else:
            raise ValueError("Unknown dataset")
        output_path = f'{output_dir}/{qid}.txt'
        if dataset == 'longvideobench':
            output_path = f'{output_dir}/{item["id"]}.txt'
        if os.path.exists(output_path):
            print(f'Exists, skip: {output_path}')
            continue
        try:
            total_frames, fps = probe_video(video_path)
            if fps <= 0:
                print(f"[WARN] fps<=0 for {video_path}, skip")
                error += 1
                continue
            fps_int = int(fps)
            # print('fps', fps,total_frames)
        except Exception as e:
            print(f"[ERR] probe video failed: {video_path}, {e}")
            error += 1
            continue
        if dataset == 'mlvu':
            video_feats_path = f'/blip_feature_mlvu_test/MLVU/blip/features/{qid}.npy'
        elif dataset == 'lvbench':
            video_feats_path = f'/blip_feature_lvbench_test/LVBench/LVB/dinov3/{qid}.npy'
        elif dataset == 'longvideobench':
            video_feats_path = f'/blip_feature_videomme_test/LongVideoBench/blip/features/{qid}.npy'
        elif dataset == 'videomme':
            video_feats_path = f'/blip_feature_videomme_test/Video-MME/blip/features/{qid}.npy'
        if not os.path.exists(video_feats_path):
            print(f'Not exists, skip: {video_feats_path}')
            error += 1
            continue

        video_feats = np.load(video_feats_path)  # (n, 1024)
        if video_feats.ndim != 2 or video_feats.shape[0] == 0:
            print("[WARN] empty feats, skip")
            error += 1
            continue
        video_feats = video_feats.astype(np.float32, copy=False)
        norms = np.linalg.norm(video_feats, axis=1, keepdims=True) + 1e-8
        video_feats_norm = video_feats / norms  # (n, d)
        n_feats = len(video_feats_norm)
        if n_feats == 0:
            print('error, no frames')
            error += 1
            continue
        cur_num_segments = num_segments
        cur_total_keep = total_keep
        correct += 1
        # ===== D_s =====
        try:
            segment_var, segment_var_mean, segment_var_softmax = compute_segment_variance_with_softmax(
                video_feats_norm, num_segments=cur_num_segments
            )
        except ValueError:
            print('error in segment variance')
            error += 1
            continue
        # ===== load ITM score =====
        if dataset == 'mlvu':
            itm_path = f'/data/blip_feature_mlvu_test/MLVU/blip/itm/{qid}.json'
        elif dataset == 'lvbench':
            itm_path = f'/data/blip_feature_lvbench_test/LVBench/LVB/blip/itm/{qid}.json'
        elif dataset == 'longvideobench':
            itm_path = f'/data/blip_feature_videomme_test/LongVideoBench/blip/itm/{qid}.json'
        elif dataset == 'videomme':
            itm_path = f'/data/blip_feature_videomme_test/Video-MME/blip/itm/{qid}.json'
        if not os.path.exists(itm_path):
            print('no itm_path',itm_path)
            continue
        itm_data = load_json(itm_path)

        # 按时间均分成 cur_num_segments 

        sim_split_data = split_second_elements(itm_data['frame_scores'], cur_num_segments)
        # 每个片段平均相似度 -> softmax
        clip_sim = [float(np.mean(seg)) if len(seg) > 0 else 0.0 for seg in sim_split_data]
        soft_sim = softmax(np.array(clip_sim, dtype=np.float32) / softmax_tau)
        print('soft_sim',soft_sim)
        print('segment_var_softmax',segment_var_softmax)

        # c_s =  beta*M_s + (1-beta)*D_s   融合分数（视觉方差 + 语义相似）
        clip_score = beta * segment_var_softmax + (1 - beta) * soft_sim

        # 分配每段保留帧数
        try:
            m_per_segment, _, _ = allocate_and_select_frames_from_softmax(
                video_feats_norm, clip_score, total_keep=cur_total_keep
            )
            # 按每段真实的相似度数据长度校正容量，并重新分配溢出配额。
            # 例如某段初始分到 5 帧但 L_j=3，则收回 2 帧，
            # 再优先补给 clip_score 高且尚有容量的其他片段。
            segment_capacities = np.asarray(
                [len(seg) for seg in sim_split_data], dtype=np.int64
            )
            before_redistribute = m_per_segment.copy()
            m_per_segment = redistribute_segment_quota(
                m_per_segment=m_per_segment,
                segment_capacities=segment_capacities,
                segment_scores=clip_score,
                target_total=cur_total_keep,
            )

            print("m_per_segment before redistribute:",
                  before_redistribute, int(before_redistribute.sum()))
            print("segment capacities:",
                  segment_capacities, int(segment_capacities.sum()))
            print("m_per_segment after redistribute:",
                  m_per_segment, int(m_per_segment.sum()))
        except ValueError as e:
            error += 1
            print('error allocate:', e)
            continue

        # ===== 第二阶段：FPS 选帧（GPU） =====
        # 加速点 2：整段特征一次性搬 GPU（不要每个 segment 搬一次）
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type != "cuda":
            print("[WARN] CUDA not available, this version expects CUDA for FPS. Skip.")
            error += 1
            continue

        # pin + non_blocking（减少 H2D 开销）
        feats_cpu_t = torch.from_numpy(video_feats_norm).pin_memory()
        if USE_FP16_FPS:
            video_feats_t = feats_cpu_t.to(device, dtype=torch.float16, non_blocking=True)
        else:
            video_feats_t = feats_cpu_t.to(device, dtype=torch.float32, non_blocking=True)

        # 你原逻辑：当 total_frames/fps > 128 用 time_idx*fps，否则用 linspace 映射
        if total_frames / fps_int <= 128:
            sampled_frame_indices = np.linspace(0, total_frames - 1, num=n_feats, dtype=int)

        # 每个 sim 段在 1fps 序列中的起始偏移（秒）
        sim_lengths = [len(seg) for seg in sim_split_data]
        sim_offsets = np.cumsum([0] + sim_lengths[:-1])
        # print("n_feats:", n_feats, "sum(sim_lengths):", sum(sim_lengths))

        result_indices_total = []
        # print('cur_num_segments', cur_num_segments)
        got=0
        for j in range(cur_num_segments):
            sim_segment = np.asarray(sim_split_data[j], dtype=np.float32)
            L_j = len(sim_segment)
            m_j = int(m_per_segment[j])
            # print('L_j, m_j',L_j, m_j)
            if m_j <= 0 or L_j == 0:
                continue
            seg_offset = int(sim_offsets[j])
            # anchor
            anchor_local = int(sim_segment.argmax())

            # ===== 阈值自适应筛候选 =====
            max_sim = float(sim_segment.max())
            threshold = max_sim if max_sim <= 0 else (gemma * max_sim)

            MIN_THRESHOLD = 1e-6
            while True:
                candidate_mask = sim_segment >= threshold
                candidate_indices = np.where(candidate_mask)[0].astype(np.int32, copy=False)

                if anchor_local not in candidate_indices:
                    candidate_indices = np.concatenate([np.array([anchor_local], dtype=np.int32), candidate_indices])

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
            # print('effective_m_j',effective_m_j)
            # print(' seg_end', seg_end)
            # print('n_feats',n_feats)
            if seg_start < 0 or seg_end > n_feats:
                # 安全保护
                continue
            segment_feats_t = video_feats_t[seg_start:seg_end]

            selected_local_indices = farthest_point_sampling_gpu(
                segment_feats_t=segment_feats_t,
                candidate_idx_np=candidate_indices,
                anchor_local=anchor_local,
                m=effective_m_j
            )
            got+=len(selected_local_indices)
            # print("seg", j, "need", effective_m_j, "got", got)
            # 映射回原视频帧 index
            # print('selected_local_indices',len(selected_local_indices))
            print('selected_local_indices',selected_local_indices)

            print(total_frames)
            for local_idx in selected_local_indices:
                score_idx = seg_offset + local_idx

                if 0 <= score_idx < len(itm_data['frame_scores']):
                    frame_idx_original = int(itm_data['frame_scores'][score_idx][0])

                    if 0 <= frame_idx_original < total_frames:
                        result_indices_total.append(frame_idx_original)
            print('result_indices_total', result_indices_total)
            # exit(0)
        # 写出
        # print('result_indices_total',len(result_indices_total))
        # exit(0)
        result_indices_total = sorted(set(result_indices_total))
        with open(output_path, "w", encoding="utf-8") as f:
            for frame_idx in result_indices_total:
                f.write(str(frame_idx) + "\n")

    print("error videos:", error)
    print("correct videos:", correct)
