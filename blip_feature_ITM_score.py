import os
import re
import json
import argparse
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from PIL import Image
from decord import VideoReader, cpu
from lavis.models import load_model_and_preprocess
def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Extract frame features and ITM scores with BLIP"
            "for VideoQA datasets"
        )
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="Video-MME",
        help="LongVideoBench / Video-MME / MLVU / LVBench",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/data/jsb/datasets/Video-MME",
        help="Dataset root directory",
    )
    parser.add_argument(
        "--extract_feature_model",
        type=str,
        default="blip",
        choices=["blip"],
        help="Feature extractor",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="blip_feature_mlvu_test",
        help="Root directory for output files",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="cuda or cpu; LOCAL_RANK is used for distributed CUDA runs",
    )
    parser.add_argument(
        "--min_frames",
        type=int,
        default=128,
        help=(
            "If 1-FPS sampling produces fewer frames, uniformly sample "
            "at least this many frames"
        ),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="BLIP image batch size; reduce it if CUDA OOM occurs",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of CPU threads used for image preprocessing",
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
    """
    Convert a sample key to a filesystem-safe name.

    Keep directory information by replacing '/' with '__', rather than
    discarding parent directories. This avoids collisions between videos
    with the same basename in different categories.
    """
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


def sample_frame_indices(vr: VideoReader, target_frames: int = 128):
    total_frames = len(vr)
    if total_frames <= 0:
        return []

    fps = float(vr.get_avg_fps())
    step = max(int(fps), 1)

    # Approximately one frame per second.
    frame_nums = total_frames // step
    one_fps_indices = [j * step for j in range(frame_nums)]

    # Preserve all 1-FPS frames when their number is already sufficient.
    if len(one_fps_indices) >= target_frames:
        return one_fps_indices

    # Otherwise uniformly sample target_frames frames from the full video.
    if total_frames >= target_frames:
        return (
            np.linspace(0, total_frames - 1, target_frames)
            .astype(np.int64)
            .tolist()
        )

    # Very short video: use all frames, then repeat the final frame.
    base = list(range(total_frames))
    pad = [total_frames - 1] * (target_frames - len(base))
    return base + pad


@torch.no_grad()
def extract_blip_frame_features(
    vr,
    indices,
    model,
    vis_processors,
    text_processors,
    device: str,
    text: str,
    batch_size: int = 32,
    num_workers: int | None = None,
):
    """
    Compute one BLIP embedding and one ITM matching probability per frame.

    Returns:
        features: np.ndarray with shape [T, D]
        itm_scores: np.ndarray with shape [T]

    The caller stores each question's ITM scores as a JSON object:
        {
            "sample_id": "...",
            "frame_scores": [[absolute_frame_idx, itm_score], ...]
        }
    """
    if not text or not str(text).strip():
        raise ValueError("BLIP ITM requires a non-empty question.")

    processed_text = text_processors["eval"](str(text))
    indices = list(indices)

    if len(indices) == 0:
        return None, None

    if num_workers is None:
        cpu_count = os.cpu_count() or 8
        num_workers = min(16, max(4, cpu_count // 2))

    processor = vis_processors["eval"]
    feature_batches = []
    itm_score_batches = []

    def transform_one(frame_hwc_uint8: np.ndarray):
        return processor(Image.fromarray(frame_hwc_uint8))

    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]

        if hasattr(vr, "get_batch"):
            batch = vr.get_batch(batch_indices)

            if hasattr(batch, "asnumpy"):
                raw_batch = batch.asnumpy()
            elif isinstance(batch, torch.Tensor):
                raw_batch = batch.detach().cpu().numpy()
            elif hasattr(batch, "numpy"):
                raw_batch = batch.numpy()
            else:
                raw_batch = np.asarray(batch)
        else:
            raw_batch = np.stack(
                [np.asarray(vr[index]) for index in batch_indices],
                axis=0,
            )

        if num_workers <= 1:
            image_list = [transform_one(frame) for frame in raw_batch]
        else:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                image_list = list(executor.map(transform_one, raw_batch))

        image_batch = torch.stack(image_list, dim=0).to(
            device,
            non_blocking=True,
        )
        text_batch = [processed_text] * image_batch.size(0)

        model_result = model(
            {
                "image": image_batch,
                "text_input": text_batch,
            },
            match_head="itm",
        )

        # This script expects the locally installed/customized LAVIS model
        # to return (ITM logits, embedding).
        if not isinstance(model_result, (tuple, list)) or len(model_result) != 2:
            raise RuntimeError(
                "Expected BLIP forward() to return "
                "(blip_output, blip_embedding), but got "
                f"{type(model_result).__name__}. "
                "Check the forward() implementation of "
                "blip_image_text_matching in your LAVIS installation."
            )

        blip_output, blip_embedding = model_result

        if blip_output is None:
            raise RuntimeError("blip_output is None.")

        if blip_output.ndim != 2 or blip_output.shape[-1] < 2:
            raise RuntimeError(
                "Expected ITM logits with shape [B, 2], but got "
                f"{tuple(blip_output.shape)}."
            )

        # Probability of the matching class.
        batch_itm_scores = torch.softmax(
            blip_output.float(),
            dim=-1,
        )[:, 1]
        itm_score_batches.append(
            batch_itm_scores.detach().cpu().numpy().astype(np.float32)
        )
        if blip_embedding is None:
            raise RuntimeError(
                "blip_embedding is None. Check the customized BLIP "
                "forward() return value."
            )
        embedding = blip_embedding
        # Convert token-level embeddings [B, L, D] to frame-level [B, D].
        if embedding.ndim == 3:
            embedding = embedding.mean(dim=1)
        if embedding.ndim != 2:
            raise RuntimeError(
                "Expected BLIP embedding with shape [B, D] or [B, L, D], "
                f"but got {tuple(embedding.shape)}."
            )
        feature_batches.append(
            embedding.detach().cpu().numpy().astype(np.float32)
        )
    features = np.concatenate(feature_batches, axis=0)
    itm_scores = np.concatenate(itm_score_batches, axis=0)
    expected_length = len(indices)
    if features.shape[0] != expected_length:
        raise RuntimeError(
            f"Feature count mismatch: expected {expected_length}, "
            f"got {features.shape[0]}."
        )
    if itm_scores.shape[0] != expected_length:
        raise RuntimeError(
            f"ITM score count mismatch: expected {expected_length}, "
            f"got {itm_scores.shape[0]}."
        )
    return features, itm_scores


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
            # Append exactly once per QA sample.
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
        return video_id if video_id.lower().endswith(".mp4") else f"{video_id}.mp4"

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
    datas, video_root = load_labels(
        args.dataset_name,
        args.dataset_path,
    )
    print(f"[INFO] device={device}")
    print(f"[INFO] total samples={len(datas)}")
    print(
        f"[INFO] chunk={args.chunk_id}/{args.num_chunks}, "
        f"video_root={video_root}"
    )
    if args.extract_feature_model == "blip":
        model, vis_processors, text_processors = (
            load_model_and_preprocess(
                "blip_image_text_matching",
                "large",
                device=device,
                is_eval=True,
            )
        )
        processor = None
    else:
        raise ValueError(
            f"Unsupported model: {args.extract_feature_model}"
        )
    out_dir = os.path.join(
        args.output_dir,
        args.dataset_name,
        args.extract_feature_model,
    )
    feature_dir = os.path.join(out_dir, "features")
    itm_dir = os.path.join(out_dir, "itm")
    ensure_dir(feature_dir)
    # Only BLIP produces ITM files.
    if args.extract_feature_model == "blip":
        ensure_dir(itm_dir)
    frames_json_path = os.path.join(
        out_dir,
        f"frames.chunk{args.chunk_id}.json",
    )
    manifest_path = os.path.join(
        out_dir,
        f"manifest.chunk{args.chunk_id}.json",
    )
    # Preserve records from an interrupted/resumed run.
    sample_to_frames = load_json_dict(frames_json_path)
    manifest = load_json_dict(manifest_path)
    print(f"[INFO] output directory: {out_dir}")
    print(f"[INFO] feature directory: {feature_dir}")
    if args.extract_feature_model == "blip":
        print(f"[INFO] ITM directory: {itm_dir}")
    from tqdm import tqdm
    for index, data in enumerate(
        tqdm(
            datas,
            desc="Processing samples",
            total=len(datas),
        )
    ):
        # Split QA samples across processes/GPUs.
        if index % args.num_chunks != args.chunk_id:
            continue
        video_relative_path = get_video_relative_path(
            args.dataset_name,
            data,
        )
        video_path = os.path.join(
            video_root,
            video_relative_path,
        )
        video_key = video_relative_path
        question = str(data.get("question", "")).strip()
        sample_id = get_sample_id(data, index)
        # Save both feature and ITM files using sample_id as the filename.
        sample_key = str(sample_id)
        safe_sample_id = safe_filename(sample_id)
        feature_file = os.path.join(feature_dir, f"{safe_sample_id}.npy")
        itm_file = os.path.join(itm_dir, f"{safe_sample_id}.json")
        manifest_key = sample_key
        feature_exists = os.path.exists(feature_file)
        itm_exists = (
            os.path.exists(itm_file)
            if args.extract_feature_model == "blip"
            else True
        )
        if feature_exists and itm_exists:
            manifest_item = {
                "sample_index": index,
                "sample_id": sample_id,
                "video_key": video_key,
                "question": question,
                "feature": os.path.relpath(
                    feature_file,
                    out_dir,
                ),
            }
            if args.extract_feature_model == "blip":
                manifest_item["itm"] = os.path.relpath(
                    itm_file,
                    out_dir,
                )
            manifest[manifest_key] = manifest_item
            continue
        if not os.path.exists(video_path):
            print(
                f"[GPU{args.chunk_id}] [WARN] video not found: "
                f"{video_path}"
            )
            continue
        try:
            video_reader = VideoReader(
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
        frame_indices = sample_frame_indices(
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
            features, itm_match_scores = (
                    extract_blip_frame_features(
                        vr=video_reader,
                        indices=frame_indices,
                        model=model,
                        vis_processors=vis_processors,
                        text_processors=text_processors,
                        device=device,
                        text=question,
                        batch_size=args.batch_size,
                        num_workers=args.num_workers,
                    )
                )
        except Exception as error:
            print(
                f"[GPU{args.chunk_id}] [WARN] extraction failed: "
                f"sample={sample_id} | video={video_path} | "
                f"error={error}"
            )
            continue
        if features is None:
            print(
                f"[GPU{args.chunk_id}] [WARN] empty features: "
                f"sample={sample_id} | video={video_path}"
            )
            continue
        if features.shape[0] != len(frame_indices):
            print(
                f"[GPU{args.chunk_id}] [WARN] feature/frame mismatch: "
                f"features={features.shape[0]}, "
                f"frames={len(frame_indices)}"
            )
            continue
        np.save(feature_file, features)
        if itm_match_scores is not None:
            if itm_match_scores.shape[0] != len(frame_indices):
                print(
                    f"[GPU{args.chunk_id}] [WARN] ITM/frame mismatch: "
                    f"itm={itm_match_scores.shape[0]}, "
                    f"frames={len(frame_indices)}"
                )
                # Avoid retaining a feature file without its matching ITM file.
                try:
                    os.remove(feature_file)
                except OSError:
                    pass
                continue
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
                json.dump(itm_payload,f,ensure_ascii=False)
        sample_to_frames[manifest_key] = {
            "sample_index": index,
            "sample_id": sample_id,
            "video_key": video_key,
            "indices": frame_indices,
        }
        manifest_item = {
            "sample_index": index,
            "sample_id": sample_id,
            "video_key": video_key,
            "question": question,
            "feature": os.path.relpath(
                feature_file,
                out_dir,
            ),
        }
        if itm_match_scores is not None:
            manifest_item["itm"] = os.path.relpath(
                itm_file,
                out_dir,
            )
        manifest[manifest_key] = manifest_item
        itm_message = (
            f" | itm_shape={itm_match_scores.shape}"
            if itm_match_scores is not None
            else ""
        )
        print(
            f"[GPU{args.chunk_id}] [OK] "
            f"sample={sample_id} | "
            f"video={video_key} | "
            f"frames={len(frame_indices)} | "
            f"feature_shape={features.shape}"
            f"{itm_message}"
        )
if __name__ == "__main__":
    main(parse_arguments())
