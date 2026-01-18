import math
import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import re
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=False, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

path = 'qwen3-vl/OpenGVLab/InternVL3_5-8B'
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    load_in_8bit=False,
    low_cpu_mem_usage=True,
    use_flash_attn=False,
    trust_remote_code=True,
    # device_map = 'auto',
    ).to('cuda').eval()
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)


generation_config = dict(max_new_tokens=256, do_sample=False)

# video multi-round conversation (视频多轮对话)
def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices


# def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32,keyframes=None):
#     vr = VideoReader(video_path, ctx=cpu(0))
#     max_frame = len(vr) - 1
#     fps = float(vr.get_avg_fps())

#     pixel_values_list, num_patches_list = [], []
#     transform = build_transform(input_size=input_size)
#     frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
#     for frame_index in frame_indices:
#         img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
#         img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
#         pixel_values = [transform(tile) for tile in img]
#         pixel_values = torch.stack(pixel_values)
#         num_patches_list.append(pixel_values.shape[0])
#         pixel_values_list.append(pixel_values)
#     pixel_values = torch.cat(pixel_values_list)
#     return pixel_values, num_patches_list


def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32, keyframes=None):
    vr = VideoReader(video_path, ctx=cpu(0))
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    print(keyframes)
    # ===== 关键修改部分 =====
    if keyframes is not None:
        # print(1)
        # 假设 keyframes 已经是帧 index 的列表
        # 这里可以顺便做一下合法性过滤（可选）
        frame_indices = []
        for idx in keyframes:
            idx = int(idx)
            if 0 <= idx <= max_frame:
                frame_indices.append(idx)
        # 如果过滤后为空，可以根据需要决定是否退回到等间隔采样
        # print(frame_indices)
        # exit(0)
        if len(frame_indices) == 0:
            frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    else:
        # 原来的等间隔采样逻辑
        frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    # ========================

    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)

    pixel_values = torch.cat(pixel_values_list)
    # print(pixel_values.shape)
    # print(num_patches_list)
    # exit(0)
    return pixel_values, num_patches_list

# def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
#     vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
#     max_frame = len(vr) - 1
#     fps = float(vr.get_avg_fps())

#     pixel_values_list, num_patches_list = [], []
#     transform = build_transform(input_size=input_size)
#     frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
#     print(frame_indices)
#     for frame_index in frame_indices:
#         img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
#         img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
#         pixel_values = [transform(tile) for tile in img]
#         pixel_values = torch.stack(pixel_values)
#         num_patches_list.append(pixel_values.shape[0])
#         pixel_values_list.append(pixel_values)
#     pixel_values = torch.cat(pixel_values_list)
#     print(pixel_values.shape)
#     print(num_patches_list)
#     exit(0)
#     return pixel_values, num_patches_list



import math

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i: i + chunk_size] for i in range(0, len(lst), chunk_size)]
# def get_chunk(lst, n, k):
#     chunks = split_list(lst, n)
#     return chunks[k]
def get_chunk(lst, n, k):
    """
    把样本按轮询方式分到 n 个 chunk 中，
    返回第 k 个 chunk（k 从 0 开始）
    例如：n = 4
      k = 0 -> lst[0], lst[4], lst[8], ...
      k = 1 -> lst[1], lst[5], lst[9], ...
    """
    return lst[k::n]

def parse_multi_choice_response(response, all_choices, index2ans):
    """
    Parse the prediction from the generated response.
    Return the predicted index e.g., A, B, C, D.
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py#L10
    """
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = " " + response + " "  # add space to avoid partial match

    index_ans = True
    ans_with_brack = False
    candidates = []
    for choice in all_choices:  # e.g., (A) (B) (C) (D)
        if f"({choice})" in response:
            candidates.append(choice)
            ans_with_brack = True

    if len(candidates) == 0:
        for choice in all_choices:  # e.g., A B C D
            if f"{choice} " in response:
                candidates.append(choice)

    if len(candidates) == 0:
        for choice in all_choices:  # e.g., A. B. C. D.
            if f"{choice}." in response:
                candidates.append(choice)

    # if all above doesn't get candidates, check if the content is larger than 5 tokens and try to parse the example
    if len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index)
                index_ans = False  # it's content ans.

    if len(candidates) == 0:  # still not get answer, randomly choose one.
        pred_index = random.choice(all_choices)
    elif len(candidates) > 1:
        start_indexes = []
        if index_ans:
            if ans_with_brack:
                for can in candidates:
                    index = response.rfind(f"({can})")
                    start_indexes.append(index)  # -1 will be ignored anyway
                # start_indexes = [generated_response.index(f'({can})') for can in candidates]
            else:
                for can in candidates:
                    index = response.rfind(f" {can} ")
                    start_indexes.append(index)
        else:
            for can in candidates:
                index = response.lower().rfind(index2ans[can].lower())
                start_indexes.append(index)
        # get the last one
        pred_index = candidates[np.argmax(start_indexes)]
    else:  # if only one candidate, use it.
        pred_index = candidates[0]

    return pred_index

def get_option_prompt(candidates, version="default"):
    option_prompt = ""
    options = []
    for idx, candidate in enumerate(candidates):
        choice = chr(ord("A") + idx)
        if version == "v4":
            option_prompt += f"({choice}) {candidate}\n"
        else:
            option_prompt += f"({choice}):{candidate} "
        options.append(choice)
    options = "(" + ",".join(options) + ")"
    return option_prompt, options  

# video_path = 'qwen3-vl/k1SE25mURhc.mp4'
# pixel_values, num_patches_list = load_video(video_path,input_size=336, num_segments=64, max_num=1)
# pixel_values = pixel_values.to(torch.bfloat16).cuda()
# video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
# question = video_prefix + 'What is the red panda doing?'
# # Frame1: <image>\nFrame2: <image>\n...\nFrame8: <image>\n{question}
# response, history = model.chat(tokenizer, pixel_values, question, generation_config,
#                                num_patches_list=num_patches_list, history=None, return_history=True)
# print(f'User: {question}\nAssistant: {response}')

# question = 'Describe this video in detail.'
# response, history = model.chat(tokenizer, pixel_values, question, generation_config,
#                                num_patches_list=num_patches_list, history=history, return_history=True)
# print(f'User: {question}\nAssistant: {response}')



import json
from tqdm import tqdm
import os
# 输入信息
def run_inference(args):
    data = open(args.gt_file, 'r').readlines()
    gt_questions = []
    for row in data:
        row = eval(row)

        video_id = row['key']
   
    
      
        qid = f"{video_id}_{row['uid']}"
        question = row['question'].split('\n')[0]
        option = ' '.join(row['question'].split('\n')[1:])
        option_dict = re.findall(r'\(([A-D])\)\s*([^\(]+?)(?=\s*\([A-D]\)|$)', option)
        options = [f'{idx}. {answer}' for idx, answer in option_dict]
        index2ans = {idx: answer for idx, answer in option_dict}
        answer = row['answer']
        time_reference = row['time_reference']
        try:
            start, end = time_reference.split('-')
            sh, ss = start.split(':')
            eh, es = end.split(':')
        except:
            continue
        start_end = [int(sh)*60+int(ss), int(eh)*60+int(es)]
            
        gt_questions.append({
            'question_id': qid,
            'video_id': video_id,
            'question': question,
            'time_reference': start_end,
            'answer': answer,
            'options': options,
            'index2ans': index2ans,
        })
    questions = get_chunk(gt_questions, args.num_chunks, args.chunk_idx)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    if args.num_chunks > 1:
        output_name = f"{args.num_chunks}_{args.chunk_idx}"
    else:
        output_name = args.output_name
    answers_file = os.path.join(args.output_dir, f"{output_name}.json")
    existing_ids = set()
    if os.path.exists(answers_file):
        with open(answers_file, "r", encoding="utf-8") as f_in:
            for line in f_in:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if "id" in rec:
                        existing_ids.add(rec["id"])
                except json.JSONDecodeError:
                    # 如果存在坏行，忽略以保证不中断
                    continue
    # —— 以追加模式打开输出文件 —— 
    ans_file = open(answers_file, "a", encoding="utf-8")

    # —— 读取已存在的 id —— 
    existing_ids = set()
    if os.path.exists(answers_file):
        with open(answers_file, "r", encoding="utf-8") as f_in:
            for line in f_in:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if "id" in rec:
                        existing_ids.add(rec["id"])
                except json.JSONDecodeError:
                    # 如果存在坏行，忽略以保证不中断
                    continue
    # —— 以追加模式打开输出文件 —— 
    ans_file = open(answers_file, "a", encoding="utf-8")
    miss_qid = []
    # i=0
    for line in tqdm(questions):
        qid = line["question_id"]
        if qid in existing_ids:
            continue
        answer = line["answer"]
        video_id = line["video_id"]
        time_reference = line['time_reference']
        options = line['options']
        index2ans = line['index2ans']

        ori_question = line['question']
        question = [line['question']] + options
        question = '\n'.join(question)
        question = f'{question}\nPlease answer directly with only the letter of the correct option and nothing else.'

        sample_set = {
            "id": qid, 
            "video_id": video_id,
            # "question": question, 
            "answer": answer, 
            'time_reference': time_reference,
        }
        # print(line)
        video_path = os.path.join(args.video_dir, video_id+'.mp4')

        # Check if the video exists
        if not os.path.exists(video_path):
            print(f'Miss video {video_path}')
            continue
        if args.keyframe_path!='None':
            text_path = f"{args.keyframe_path}/{sample_set['id']}.txt"
            # print('keyframe')
            # print(text_path)
            # exit(0)
            if not os.path.exists(text_path):
                continue
            with open(text_path, "r", encoding="utf-8") as f:
                lines = f.readlines()  # 每行是一个字符串
            # 转成整数列表
                keyframes = [int(line.strip()) for line in lines if line.strip()]
                # print(1,keyframes)
        else:
            # keyframes = None
            vr = VideoReader(video_path, ctx=cpu(0))
            total_frames = len(vr)
            n_frames=64
            if n_frames <= 0:
                raise ValueError("n_frames 必须为正整数")
            if n_frames > total_frames:
                # 如果想强制 n_frames 不超过总帧数，可以改成 raise
                n_frames = total_frames
            # 在 [0, total_frames-1] 之间均匀采样 n_frames 个索引
            # 使用 linspace 再取整，保证首尾都包含
            keyframes = np.linspace(0, total_frames - 1, n_frames, dtype=np.int32).tolist()
            # print(2,keyframes)
        # print(3,keyframes)
        # video_path = 'qwen3-vl/k1SE25mURhc.mp4'
        num_frames=64
        num_segments = num_frames
        pixel_values, num_patches_list = load_video(video_path,input_size=336, num_segments=num_segments, max_num=1,keyframes=keyframes)
        # pixel_values, num_patches_list = load_video(video_path,input_size=336, num_segments=num_segments, max_num=1)
        print('pixel_values.shape',pixel_values.shape)
        pixel_values = pixel_values.to(torch.bfloat16).cuda()
        video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
        question = video_prefix + question
        # Frame1: <image>\nFrame2: <image>\n...\nFrame8: <image>\n{question}
        outputs, history = model.chat(tokenizer, pixel_values, question, generation_config,
                                    num_patches_list=num_patches_list, history=None, return_history=True)
        print(outputs)
        parsed_pred = parse_multi_choice_response(outputs, ["A", "B", "C", "D","E"], index2ans)
        sample_set['acc'] = str(parsed_pred == answer)   
        sample_set["pred"] = outputs
        print('parsed_pred', parsed_pred)
        print('outputs', outputs)
        print('sample_set[acc]', sample_set['acc'])
        # ans_id = shortuuid.uuid()
        ans_file.write(json.dumps(sample_set)+ "\n")
        # output_text = output_text[0].replace("In the image", "In the video")
        #         # print(output)
        # sample_set["pred"] = output_text
        # # print(output_text)
        # ans_file.write(json.dumps(sample_set) + "\n")
        # exit(0)
    ans_file.close()

import argparse
def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", help="Directory containing video files.", required=True)
    parser.add_argument("--gt_file", help="Path to the ground truth file containing question and answer.", required=True)
    parser.add_argument("--output_dir", help="Directory to save the model results JSON.", required=True)
    parser.add_argument("--output_name", help="Name of the file for storing results JSON.", required=True)
    # parser.add_argument("--model_path", type=str, required=True)
    # parser.add_argument("--model_base", type=str, default=None)
    # parser.add_argument("--conv_mode", type=str, default="vicuna_v1")
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--keyframe_path", type=str, default=None)
    # parser.add_argument("--num_frames", type=int, default=100)
    # parser.add_argument("--temperature", type=float, default=0.2)
    # parser.add_argument("--top_p", type=float, default=None)
    # parser.add_argument("--num_beams", type=int, default=1)
    # parser.add_argument("--input_structure", type=str, default="image_seq")
    # parser.add_argument("--image_aspect_ratio", type=str, default=None)
    # parser.add_argument("--temporal_aggregation", type=str, default=None)
    # parser.add_argument("--rope_scaling_factor", type=int, default=1)
    # parser.add_argument("--key_frame_path", type=str, default=None)
    # parser.add_argument("--prune_mode", type=str, default=None)
    # parser.add_argument("--rate", help='this_global_rate', type=float,default=None)
    # parser.add_argument("--tokens_num", help='tokens_num', type=int,default=936)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_inference(args)