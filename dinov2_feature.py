import os
import json
import argparse
import multiprocessing as mp
from multiprocessing import Process

import torch
from PIL import Image
from tqdm import tqdm
from decord import VideoReader, cpu
import numpy as np
import torchvision.transforms as T
from numpy.lib.format import open_memmap


def parse_arguments():
    parser = argparse.ArgumentParser(description='Extract Video Features (DINOv2)')

    parser.add_argument('--dataset_name', type=str, default='longvideobench',
                        help='support longvideobench / videomme / mlvu / LVB')
    parser.add_argument('--dataset_path', type=str,
                        default='longvideobench',
                        help='your path of the dataset')
    parser.add_argument('--output_file', type=str,
                        default='frame_features/LVBench',
                        help='path of output features')
    parser.add_argument('--device', type=str, default='cuda',
                        help='base device, e.g., cuda or cpu')
    parser.add_argument('--world_size', type=int, default=4,
                        help='number of gpus / processes, e.g., 8')

    # 控制内存占用
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch size for DINOv2 forward')
    parser.add_argument('--chunk_frames', type=int, default=256,
                        help='how many frames to process in one chunk')
    parser.add_argument('--max_frames', type=int, default=None,
                        help='upper limit of frames per video after 1fps sampling; '
                             'None means no limit')

    # DINOv2 模型名（可选：dinov2_vits14 / dinov2_vitb14 / dinov2_vitl14 / dinov2_vitg14）
    parser.add_argument('--dinov2_model', type=str, default='dinov2_vitl14',
                        help='which dinov2 model to load from torch hub')

    return parser.parse_args()


def prepare_paths(args):
    """准备标签路径、视频路径和输出路径。"""
    if args.dataset_name == "longvideobench":
        label_path = os.path.join(args.dataset_path, 'lvb_val.json')
        video_path = os.path.join(args.dataset_path, 'videos')
    elif args.dataset_name == "videomme":
        label_path = os.path.join(args.dataset_path, 'videomme/test_anns.csv')
        video_path = os.path.join(args.dataset_path, 'data')
    elif args.dataset_name == 'mlvu':
        label_path = os.path.join(args.dataset_path, 'annotations/multiple_choice.json')
        video_path = os.path.join(args.dataset_path, 'video')
    elif args.dataset_name == 'LVB':
        # LVB: jsonl（每行一个 json）
        label_path = os.path.join(args.dataset_path, 'qa_file.json')
        video_path = os.path.join(args.dataset_path, 'videos')
    else:
        raise ValueError("dataset_name: longvideobench / videomme / mlvu / LVB")

    if not os.path.exists(label_path):
        raise OSError(f"label file does not exist: {label_path}")

    out_dir = os.path.join(args.output_file, args.dataset_name, 'dinov2')
    os.makedirs(out_dir, exist_ok=True)
    return label_path, video_path, out_dir


def build_dinov2(device: str, model_name: str):
    """加载 DINOv2 模型 + 官方推荐的 torchvision 预处理。"""
    model = torch.hub.load('facebookresearch/dinov2', model_name)
    model.to(device)
    model.eval()

    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    return model, transform


def get_video_abspath(args, video_path, data):
    """根据 dataset_name 拼视频路径。"""
    if args.dataset_name == 'longvideobench':
        return os.path.join(video_path, data["video_path"])
    elif args.dataset_name == 'videomme':
        return os.path.join(video_path, data["video_id"] + '.mp4')
    elif args.dataset_name == 'mlvu':
        return os.path.join(video_path, f"{data['category']}/{data['video']}")
    elif args.dataset_name == 'LVB':
        return os.path.join(video_path, data["video_path"])
    else:
        return None


def get_video_id(data, fallback: str):
    """优先用 key（你 LVB 里用 key），否则兜底。"""
    vid = data.get('key', None)
    if vid is None:
        vid = data.get('video_id', data.get('uid', fallback))
    return str(vid)


def worker(rank, world_size, args, label_path, video_path, out_dir):
    """
    每个进程在自己的 GPU 上处理 idx % world_size == rank 的样本。
    1fps 抽帧 -> DINOv2 特征 -> 直接 memmap 写 .npy
    """
    # 设备
    if args.device.startswith("cuda"):
        device = f"cuda:{rank}"
    else:
        device = args.device
    print(f"[Rank {rank}] using device: {device}")

    # 模型
    model, transform = build_dinov2(device, args.dinov2_model)

    batch_size = args.batch_size
    chunk_frames = args.chunk_frames
    max_frames = args.max_frames

    # 逐行读取 label（避免整文件进内存）
    with open(label_path, 'r') as f:
        for idx, line in enumerate(tqdm(f, desc=f"GPU {rank}", position=rank)):
            if idx % world_size != rank:
                continue

            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[Rank {rank}] JSON decode error in line {idx}: {e}")
                continue

            video = get_video_abspath(args, video_path, data)
            if not video or (not os.path.exists(video)):
                print(f"[Rank {rank}] video not found: {video}, skip.")
                continue

            video_id = get_video_id(data, fallback=str(idx))
            feat_path = os.path.join(out_dir, f"{video_id}.npy")
            if os.path.exists(feat_path):
                continue

            # 打开视频
            try:
                vr = VideoReader(video, ctx=cpu(0), num_threads=1)
            except Exception as e:
                print(f"[Rank {rank}] failed to open video {video}: {e}")
                continue

            fps = float(vr.get_avg_fps())
            if fps <= 0:
                print(f"[Rank {rank}] invalid fps ({fps}) for video {video}, skip.")
                del vr
                continue

            step = max(int(fps), 1)  # 约等于 1fps
            frame_indices = list(range(0, len(vr), step))

            if max_frames is not None:
                frame_indices = frame_indices[:max_frames]

            if len(frame_indices) == 0:
                print(f"[Rank {rank}] no frames sampled for video_id={video_id}, skip.")
                del vr
                continue

            # 先跑一帧确定维度 D
            try:
                first_idx = frame_indices[0]
                raw0 = vr[first_idx].asnumpy()
            except Exception as e:
                print(f"[Rank {rank}] failed to read first frame {first_idx} for {video}: {e}")
                del vr
                continue

            img0 = transform(Image.fromarray(raw0)).unsqueeze(0).to(device)
            with torch.no_grad():
                out0 = model(img0)
                if not isinstance(out0, torch.Tensor):
                    print(f"[Rank {rank}] unexpected DINOv2 output type: {type(out0)}, skip.")
                    del vr
                    continue
                feat0 = out0.view(out0.size(0), -1).detach().cpu().numpy()

            D = feat0.shape[1]
            T_frames = len(frame_indices)

            # 创建 memmap
            try:
                video_feats = open_memmap(
                    feat_path,
                    mode='w+',
                    dtype=np.float32,
                    shape=(T_frames, D)
                )
            except Exception as e:
                print(f"[Rank {rank}] failed to create memmap {feat_path}: {e}")
                del vr
                continue

            video_feats[0] = feat0[0]

            remaining_indices = frame_indices[1:]
            write_pos = 1

            if args.device.startswith("cuda"):
                model.eval()
                torch.backends.cudnn.benchmark = True

            try:
                with torch.no_grad():
                    for chunk_start in range(0, len(remaining_indices), chunk_frames):
                        chunk_ids = remaining_indices[chunk_start:chunk_start + chunk_frames]
                        if not chunk_ids:
                            break

                        try:
                            raw_chunk = vr.get_batch(chunk_ids).asnumpy()  # [Tc, H, W, 3]
                        except Exception as e:
                            print(f"[Rank {rank}] failed to read chunk for {video}: {e}")
                            break

                        imgs = [transform(Image.fromarray(img)) for img in raw_chunk]
                        imgs = torch.stack(imgs, dim=0)  # [Tc, C, H, W]

                        if args.device.startswith("cuda"):
                            imgs = imgs.pin_memory()
                            non_blocking = True
                        else:
                            non_blocking = False

                        for start in range(0, imgs.size(0), batch_size):
                            end = start + batch_size
                            batch = imgs[start:end].to(device, non_blocking=non_blocking)

                            out = model(batch)
                            if not isinstance(out, torch.Tensor):
                                raise TypeError(f"Unexpected DINOv2 output type: {type(out)}")

                            feat = out.view(out.size(0), -1).detach().cpu().numpy()
                            bs = feat.shape[0]

                            video_feats[write_pos:write_pos + bs] = feat
                            write_pos += bs

                        del raw_chunk, imgs
                        if args.device.startswith("cuda"):
                            torch.cuda.empty_cache()

                if write_pos < T_frames:
                    print(f"[Rank {rank}] warning: write_pos({write_pos}) < T_frames({T_frames}) "
                          f"for video_id={video_id}, tail frames skipped.")
            finally:
                del vr
                video_feats.flush()
                del video_feats

            print(f"[Rank {rank}] dinov2 feature saved for video_id={video_id} -> {feat_path}")

    print(f"[Rank {rank}] finished.")


def main(args):
    label_path, video_path, out_dir = prepare_paths(args)
    world_size = args.world_size

    processes = []
    for rank in range(world_size):
        p = Process(target=worker, args=(rank, world_size, args, label_path, video_path, out_dir))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("[Main] done. (dinov2 only)")


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    args = parse_arguments()
    main(args)
