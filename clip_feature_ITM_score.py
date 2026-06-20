import argparse
import json
import os
import random
import re
from typing import List, Tuple
import numpy as np
import torch
from PIL import Image
from decord import VideoReader, cpu
import decord
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
# pip install git+https://github.com/openai/CLIP.git
import clip  # noqa: E402
VIDEOMME_SYSTEM_PROMPT = (
    "Select the best answer to the following multiple-choice question based on "
    "the video and the subtitles. Respond with only the letter (A, B, C, or D) "
    "of the correct option.\n"
)
def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Extract OpenAI CLIP frame features and frame-text similarity "
            "scores for VideoQA datasets"
        )
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="MLVU",
        help="LongVideoBench / Video-MME / MLVU / LVBench",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/data/jsb/datasets/MLVU",
        help="Dataset root directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="clip_feature_mlvu_test",
        help="Root directory for output files",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="cuda or cpu; LOCAL_RANK is used for distributed CUDA runs",
    )
    parser.add_argument(
        "--clip_model",
        type=str,
        default="ViT-L/14",
        help="OpenAI CLIP model name, e.g. ViT-B/32, ViT-B/16, ViT-L/14",
    )
    parser.add_argument(
        "--min_frames",
        type=int,
        default=128,
        help=(
            "If 1-FPS sampling produces fewer frames, uniformly sample at "
            "least this many frames"
        ),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="CLIP image encoding batch size",
    )
    parser.add_argument(
        "--io_chunk_size",
        type=int,
        default=128,
        help="Number of sampled frames read from decord per chunk",
    )
    parser.add_argument(
        "--feat_dtype",
        type=str,
        default="float16",
        choices=["float16", "float32"],
        help="dtype used when saving raw CLIP image features",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=-1,
        help="Process only the first N samples; <=0 means all samples",
    )
    parser.add_argument(
        "--num_chunks",
        type=int,
        default=1,
        help="Number of processes/GPUs used to split the dataset",
    )
    parser.add_argument(
        "--chunk_id",
        type=int,
        default=0,
        help="Current chunk index in [0, num_chunks - 1]",
    )
    return parser.parse_args()
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
def safe_filename(name: str) -> str:
    name = str(name).replace("\\", "/")
    name = name.replace("/", "__")
    name = re.sub(r"[^0-9a-zA-Z._-]+", "_", name)
    return name.strip("._") or "sample"
def load_json_dict(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            value = json.load(f)
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}
def parse_videomme_question(text_or_prompt: str) -> str:
    s = text_or_prompt or ""
    if s.startswith(VIDEOMME_SYSTEM_PROMPT):
        rest = s[len(VIDEOMME_SYSTEM_PROMPT):]
        first_line = rest.split("\n")[0].strip()
        if first_line:
            return first_line
    for line in s.splitlines():
        line = line.strip()
        if line:
            return line
    return s.strip()
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def sample_frame_indices(
    vr: VideoReader,
    target_frames: int = 128,
) -> Tuple[List[int], float, float]:
    """
    Preserve the original CLIP sampling rule:
    1. Sample approximately 1 FPS.
    2. If fewer than target_frames are obtained, uniformly sample exactly
       target_frames across the full video (repeating frames when necessary).
    3. Otherwise retain all 1-FPS frames without an upper limit.
    """
    total = len(vr)
    avg_fps = float(vr.get_avg_fps()) if vr.get_avg_fps() else 30.0
    video_time = total / avg_fps if total > 0 else 0.0

    if total <= 0:
        return [], avg_fps, video_time

    num_secs = int(np.floor(video_time)) + 1
    one_fps = []
    for second in range(num_secs):
        frame_id = int(round(second * avg_fps))
        if frame_id >= total:
            frame_id = total - 1
        one_fps.append(frame_id)
    one_fps = sorted(set(one_fps))

    if len(one_fps) < target_frames:
        frame_ids = np.linspace(
            0,
            total - 1,
            target_frames,
            dtype=int,
        ).tolist()
        frame_ids = sorted(set(frame_ids))
        if len(frame_ids) < target_frames:
            index = 0
            while len(frame_ids) < target_frames:
                frame_ids.append(frame_ids[index % len(frame_ids)])
                index += 1
        return frame_ids, avg_fps, video_time

    return one_fps, avg_fps, video_time


class CLIPFrameScorer:
    """
    Original CLIP computation logic:
    - cosine similarity uses normalized image/text features;
    - saved image features are raw, unnormalized CLIP image features;
    - text longer than 77 tokens keeps the final available tokens;
    - frames are read and encoded in chunks.
    """

    def __init__(
        self,
        model_name: str,
        device: str,
        batch_size: int,
        io_chunk_size: int,
    ):
        self.device = device
        self.batch_size = int(batch_size)
        self.io_chunk_size = int(io_chunk_size)

        model, preprocess = clip.load(model_name, device=device, jit=False)
        model.eval()
        self.model = model
        self.preprocess = preprocess

    def _build_text_tokens_last77(self, text: str) -> torch.Tensor:
        context_len = 77
        tokenizer = clip.simple_tokenizer.SimpleTokenizer()

        ids = tokenizer.encode(text)
        sot = tokenizer.encoder["<|startoftext|>"]
        eot = tokenizer.encoder["<|endoftext|>"]

        budget = context_len - 2
        if len(ids) > budget:
            ids = ids[-budget:]

        text_tokens = torch.zeros(1, context_len, dtype=torch.long)
        sequence = [sot] + ids + [eot]
        text_tokens[0, : len(sequence)] = torch.tensor(
            sequence,
            dtype=torch.long,
        )
        return text_tokens.to(self.device)

    @torch.no_grad()
    def score_and_extract_streaming(
        self,
        vr: VideoReader,
        frame_ids: List[int],
        text: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        use_amp = self.device.startswith("cuda")
        text_tokens = self._build_text_tokens_last77(text)

        scores = []
        raw_feature_batches = []

        with torch.cuda.amp.autocast(enabled=use_amp):
            text_feature_raw = self.model.encode_text(text_tokens)
            text_feature = text_feature_raw / (
                text_feature_raw.norm(dim=-1, keepdim=True) + 1e-6
            )

            num_frames = len(frame_ids)
            for io_start in range(0, num_frames, self.io_chunk_size):
                io_end = min(io_start + self.io_chunk_size, num_frames)
                id_chunk = frame_ids[io_start:io_end]
                frame_chunk = vr.get_batch(id_chunk).asnumpy()

                chunk_size = int(frame_chunk.shape[0])
                for start in range(0, chunk_size, self.batch_size):
                    end = min(start + self.batch_size, chunk_size)

                    image_list = [
                        self.preprocess(Image.fromarray(frame_chunk[i]))
                        for i in range(start, end)
                    ]
                    image_tensors = torch.stack(image_list, dim=0).to(
                        self.device,
                        non_blocking=True,
                    )

                    image_feature_raw = self.model.encode_image(image_tensors)
                    raw_feature_batches.append(
                        image_feature_raw.detach().cpu()
                    )

                    image_feature = image_feature_raw / (
                        image_feature_raw.norm(dim=-1, keepdim=True) + 1e-6
                    )
                    similarity = (
                        image_feature @ text_feature.T
                    ).squeeze(-1)
                    scores.extend(
                        similarity.detach().float().cpu().tolist()
                    )

                    del image_tensors
                    del image_feature_raw
                    del image_feature
                    del similarity

        if not raw_feature_batches:
            return (
                np.zeros((0,), dtype=np.float32),
                np.zeros((0, 0), dtype=np.float32),
            )

        raw_features = torch.cat(raw_feature_batches, dim=0).numpy()
        return np.asarray(scores, dtype=np.float32), raw_features


def load_labels(dataset_name: str, dataset_path: str):
    print(
        f"[INFO] Loading labels for dataset={dataset_name} "
        f"from {dataset_path}"
    )
    normalized_name = dataset_name.lower()

    if normalized_name == "longvideobench":
        label_path = os.path.join(dataset_path, "your_lvb_val.json")
        video_root = os.path.join(dataset_path, "videos")
        with open(label_path, "r", encoding="utf-8") as f:
            datas = json.load(f)

    elif normalized_name in {"videomme", "video-mme"}:
        import pandas as pd

        label_path = os.path.join(dataset_path, "your_test_anns.csv")
        video_root = os.path.join(dataset_path, "data")
        dataframe = pd.read_csv(label_path)
        datas = []
        for _, row in dataframe.iterrows():
            option_str = str(row["options"])[1:-1]
            option_pairs = re.findall(
                r"([A-D])\.\s([^\.?]+[\.?])",
                option_str,
            )
            options = [
                f"{option_id}. {answer}"
                for option_id, answer in option_pairs
            ]
            index_to_answer = {
                option_id: answer
                for option_id, answer in option_pairs
            }
            answer_id = str(row["answer"])
            answer = index_to_answer.get(answer_id, "")
            datas.append(
                {
                    "qid": row["question_id"],
                    "question": row["question"],
                    "video_id": row["videoID"],
                    "duration_group": row.get("duration", ""),
                    "option": options,
                    "answer_id": answer_id,
                    "answer": answer,
                    "index2ans": index_to_answer,
                }
            )

    elif normalized_name == "mlvu":
        label_path = os.path.join(
            dataset_path,
            "annotations",
            "multiple_choice.json",
        )
        video_root = os.path.join(dataset_path, "video")
        with open(label_path, "r", encoding="utf-8") as f:
            datas = json.load(f)

    elif normalized_name in {"lvbench", "lvb"}:
        label_path = os.path.join(dataset_path, "your_qa_file.json")
        video_root = os.path.join(dataset_path, "all_videos")
        datas = []
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    datas.append(json.loads(line))

    else:
        raise ValueError(
            "Unsupported dataset_name. Expected one of: "
            "LongVideoBench, Video-MME, MLVU, LVBench."
        )

    return datas, video_root


def get_video_relative_path(dataset_name: str, data: dict) -> str:
    normalized_name = dataset_name.lower()

    if normalized_name == "longvideobench":
        return data["video_path"]

    if normalized_name in {"videomme", "video-mme"}:
        video_id = str(data["video_id"])
        return (
            video_id
            if video_id.lower().endswith(".mp4")
            else f"{video_id}.mp4"
        )

    if normalized_name == "mlvu":
        return f"{data['category']}/{data['video']}"

    if normalized_name in {"lvbench", "lvb"}:
        return data["video_path"]

    raise ValueError(f"Unsupported dataset_name: {dataset_name}")


def get_sample_id(data: dict, index: int) -> str:
    for key in ("qid", "question_id", "id", "uid", "qa_id"):
        value = data.get(key)
        if value is not None and str(value) != "":
            return str(value)
    return str(index)


def configure_runtime(args):
    world_size = int(os.environ.get("WORLD_SIZE", args.num_chunks))
    local_rank = int(os.environ.get("LOCAL_RANK", args.chunk_id))

    args.num_chunks = world_size
    args.chunk_id = local_rank

    if args.num_chunks <= 0:
        raise ValueError("--num_chunks must be positive.")

    if not 0 <= args.chunk_id < args.num_chunks:
        raise ValueError(
            f"chunk_id={args.chunk_id} is outside "
            f"[0, {args.num_chunks - 1}]."
        )

    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but torch.cuda.is_available() is False."
            )
        torch.cuda.set_device(local_rank)
        return f"cuda:{local_rank}"

    return args.device


def main(args):
    device = configure_runtime(args)
    set_seed(args.seed + args.chunk_id)
    datas, video_root = load_labels(
        args.dataset_name,
        args.dataset_path,
    )
    if args.limit > 0:
        datas = datas[: args.limit]

    print(f"[INFO] device={device}")
    print(f"[INFO] total samples={len(datas)}")
    print(
        f"[INFO] chunk={args.chunk_id}/{args.num_chunks}, "
        f"video_root={video_root}"
    )

    scorer = CLIPFrameScorer(
        model_name=args.clip_model,
        device=device,
        batch_size=args.batch_size,
        io_chunk_size=args.io_chunk_size,
    )

    out_dir = os.path.join(
        args.output_dir,
        args.dataset_name,
        "clip",
    )
    feature_dir = os.path.join(out_dir, "features")
    itm_dir = os.path.join(out_dir, "itm")
    ensure_dir(feature_dir)
    ensure_dir(itm_dir)

    frames_json_path = os.path.join(
        out_dir,
        f"frames.chunk{args.chunk_id}.json",
    )
    manifest_path = os.path.join(
        out_dir,
        f"manifest.chunk{args.chunk_id}.json",
    )

    sample_to_frames = load_json_dict(frames_json_path)
    manifest = load_json_dict(manifest_path)

    print(f"[INFO] output directory: {out_dir}")
    print(f"[INFO] feature directory: {feature_dir}")
    print(f"[INFO] ITM directory: {itm_dir}")

    from tqdm import tqdm

    for index, data in enumerate(
        tqdm(
            datas,
            desc="Processing samples",
            total=len(datas),
        )
    ):
        if index % args.num_chunks != args.chunk_id:
            continue

        video_relative_path = get_video_relative_path(
            args.dataset_name,
            data,
        )
        video_path = os.path.join(video_root, video_relative_path)
        video_key = video_relative_path

        prompt = str(
            data.get(
                "prompt",
                data.get("question", ""),
            )
        )
        question = parse_videomme_question(prompt)
        sample_id = get_sample_id(data, index)
        sample_key = str(sample_id)
        safe_sample_id = safe_filename(sample_id)

        feature_file = os.path.join(
            feature_dir,
            f"{safe_sample_id}.npy",
        )
        itm_file = os.path.join(
            itm_dir,
            f"{safe_sample_id}.json",
        )

        if os.path.exists(feature_file) and os.path.exists(itm_file):
            manifest[sample_key] = {
                "sample_index": index,
                "sample_id": sample_id,
                "video_key": video_key,
                "question": question,
                "feature": os.path.relpath(feature_file, out_dir),
                "itm": os.path.relpath(itm_file, out_dir),
            }
            continue

        if not os.path.exists(video_path):
            print(
                f"[GPU{args.chunk_id}] [WARN] video not found: "
                f"{video_path}"
            )
            continue

        try:
            video_reader = decord.VideoReader(
                video_path,
                ctx=cpu(0),
                num_threads=1,
            )
        except Exception as error:
            print(
                f"[GPU{args.chunk_id}] [WARN] failed to read video: "
                f"{video_path} | error={error}"
            )
            continue

        frame_indices, avg_fps, video_time = sample_frame_indices(
            video_reader,
            target_frames=args.min_frames,
        )
        if not frame_indices:
            print(
                f"[GPU{args.chunk_id}] [WARN] no frames sampled: "
                f"{video_path}"
            )
            continue

        try:
            itm_match_scores, features = (
                scorer.score_and_extract_streaming(
                    vr=video_reader,
                    frame_ids=frame_indices,
                    text=question,
                )
            )
        except Exception as error:
            print(
                f"[GPU{args.chunk_id}] [WARN] extraction failed: "
                f"sample={sample_id} | video={video_path} | "
                f"error={error}"
            )
            continue

        if features.shape[0] != len(frame_indices):
            print(
                f"[GPU{args.chunk_id}] [WARN] feature/frame mismatch: "
                f"features={features.shape[0]}, "
                f"frames={len(frame_indices)}"
            )
            continue

        if itm_match_scores.shape[0] != len(frame_indices):
            print(
                f"[GPU{args.chunk_id}] [WARN] ITM/frame mismatch: "
                f"itm={itm_match_scores.shape[0]}, "
                f"frames={len(frame_indices)}"
            )
            continue

        if args.feat_dtype == "float16":
            np.save(feature_file, features.astype(np.float16))
        else:
            np.save(feature_file, features.astype(np.float32))
        itm_payload = {
            "sample_id": sample_id,
            "frame_scores": [
                [int(frame_idx), float(score)]
                for frame_idx, score in zip(
                    frame_indices,
                    itm_match_scores.tolist(),
                )
            ],
        }
        with open(itm_file, "w", encoding="utf-8") as f:
            json.dump(itm_payload, f, ensure_ascii=False)

        sample_to_frames[sample_key] = {
            "sample_index": index,
            "sample_id": sample_id,
            "video_key": video_key,
            "indices": frame_indices,
            "avg_fps": float(avg_fps),
            "video_time_sec": float(video_time),
        }
        manifest[sample_key] = {
            "sample_index": index,
            "sample_id": sample_id,
            "video_key": video_key,
            "question": question,
            "feature": os.path.relpath(feature_file, out_dir),
            "itm": os.path.relpath(itm_file, out_dir),
        }
        print(
            f"[GPU{args.chunk_id}] [OK] "
            f"sample={sample_id} | "
            f"video={video_key} | "
            f"frames={len(frame_indices)} | "
            f"feature_shape={features.shape} | "
            f"itm_shape={itm_match_scores.shape}"
        )
if __name__ == "__main__":
    main(parse_arguments())
