import torch
from PIL import Image
from lavis.models import load_model_and_preprocess

import json
from decord import VideoReader
from decord import cpu
import numpy as np
import os

import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description='Extract Video Feature')

    parser.add_argument('--dataset_name', type=str, default='longvideobench',
                        help='support longvideobench, videomme, mlvu, lvbench')
    parser.add_argument('--dataset_path', type=str, default='./datasets/longvideobench',
                        help='your path of the dataset')
    parser.add_argument('--extract_feature_model', type=str, default='blip',
                        help='only support blip')
    parser.add_argument('--output_file', type=str, default='./outscores',
                        help='path of output scores and frames')
    parser.add_argument('--device', type=str, default='cuda')

    return parser.parse_args()


def main(args):
    if args.dataset_name =="longvideobench":
        label_path = os.path.join(args.dataset_path,'lvb_val.json')
        video_path = os.path.join(args.dataset_path,'videos')
    elif args.dataset_name =="videomme":
        label_path = os.path.join(args.dataset_path,'videomme.json')
        video_path = os.path.join(args.dataset_path,'data')
    elif args.dataset_name =='mlvu':
        label_path = os.path.join(args.dataset_path,'annotations/multiple_choice.json')
        video_path = os.path.join(args.dataset_path,'video')
    else:
       raise ValueError("dataset_name: longvideobench or videomme")

    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            datas = json.load(f)
    else:
        raise OSError("the label file does not exist")

    if args.extract_feature_model != 'blip':
        raise ValueError("Only 'blip' is supported now (clip/sevila branches removed).")

    device = args.device
    model, vis_processors, text_processors = load_model_and_preprocess(
        "blip_image_text_matching", "large", device=device, is_eval=True
    )

    # output dirs
    dataset_out_dir = os.path.join(args.output_file, args.dataset_name)
    if not os.path.exists(dataset_out_dir):
        os.makedirs(dataset_out_dir, exist_ok=True)

    out_score_path = os.path.join(dataset_out_dir, args.extract_feature_model)
    if not os.path.exists(out_score_path):
        os.makedirs(out_score_path, exist_ok=True)

    scores = []
    fn = []
    score_path = os.path.join(out_score_path, 'scores.json')
    frame_path = os.path.join(out_score_path, 'frames.json')

    for data in datas:
        text = data['question']

        if args.dataset_name == 'longvideobench':
            video = os.path.join(video_path, data["video_path"])
        elif args.dataset_name == 'videomme':
            video = os.path.join(video_path, data["videoID"]+'.mp4')
        elif args.dataset_name == 'mlvu':
            video = os.path.join(video_path, f"{data['category']}/{data['video']}"+'.mp4')

        vr = VideoReader(video, ctx=cpu(0), num_threads=1)
        fps = vr.get_avg_fps()
        frame_nums = int(len(vr) / int(fps))

        score = []
        frame_num = []

        # BLIP branch only
        txt = text_processors["eval"](text)
        for j in range(frame_nums):
            raw_image = np.array(vr[j * int(fps)])
            raw_image = Image.fromarray(raw_image)
            img = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
            with torch.no_grad():
                blip_output = model({"image": img, "text_input": txt}, match_head="itm")
            blip_scores = torch.nn.functional.softmax(blip_output, dim=1)
            score.append(blip_scores[:, 1].item())
            frame_num.append(j * int(fps))

        fn.append(frame_num)
        scores.append(score)

    with open(frame_path, 'w') as f:
        json.dump(fn, f)
    with open(score_path, 'w') as f:
        json.dump(scores, f)


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
