from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils_new import process_vision_info
import os
import torch
import argparse
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

import json
from tqdm import tqdm

# 输入信息
def run_inference(args):
    # print(args.model_path)
    
    model = AutoModelForImageTextToText.from_pretrained(
    "/models/qwen3-vl-8b",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",  
    # device_map="auto",
)
    model.to("cuda:0")

    # processor = AutoProcessor.from_pretrained(args.model_path,size={"longest_edge": 128*128, "shortest_edge": 256*256})
    # processor = AutoProcessor.from_pretrained(args.model_path)
    processor = AutoProcessor.from_pretrained("/models/qwen3-vl-8b")
    gt_qa_pairs = json.load(open(args.gt_file, "r"))
    gt_questions = []
    keyframe_num = 0
    for index, item in enumerate(gt_qa_pairs):
        question = item['question']
        option = [". ".join([chr(ord("A")+i), candidate]) for i, candidate in enumerate(item["candidates"])]
        qid = item['id']
        video_id = item["video_path"].split('.')[0]
        answer_id = chr(ord("A")+item["correct_choice"])
        answer = item["candidates"][item["correct_choice"]]
        duration_group = item['duration_group']
        index2ans = {}
        for i in range(len(item["candidates"])):
            idx = chr(ord("A")+i)
            ans = item["candidates"][i]
            index2ans[idx] = ans
        gt_questions.append({
            'qid': qid, 
            'question': question, 
            'option': option, 
            'video_id': video_id, 
            'answer_id': answer_id, 
            'answer': answer, 
            'index2ans': index2ans,
            'duration_group': duration_group,
        })
    gt_qa_pairs = get_chunk(gt_questions, args.num_chunks, args.chunk_idx)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    if args.num_chunks > 1:
        output_name = f"{args.num_chunks}_{args.chunk_idx}"
    else:
        output_name = args.output_name
    answers_file = os.path.join(args.output_dir, f"{output_name}.json")  

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
                    continue
    # —— 以追加模式打开输出文件 —— 
    ans_file = open(answers_file, "a", encoding="utf-8")
    for line in tqdm(gt_qa_pairs):

        qid = line["qid"]
        if qid in existing_ids:
            continue
        answer = line["answer"]
        video_id = line["video_id"]
        answer_id = line["answer_id"]
        option = line["option"]
        index2ans = line["index2ans"]
        duration_group = line["duration_group"]
        question = (
        f"Question:\n{line['question']}\n"
        + "Options:\n"
        + '\n'.join(option)
    )
        option_prompt = "Select the best answer to the following multiple-choice question based on the video." \
        " Respond with only the letter (A, B, C or D) of the correct option."
        question = option_prompt + "\n" + f"{question}\nAnswer with the option's letter from the given choices directly. " 
        sample_set = {
            "id": qid, 
            "video_id": video_id,
            "question": question, 
            "answer_id": answer_id, 
            "duration_group": duration_group,
            'answer': answer,
        }
        video_path = os.path.join(args.video_dir, f'{video_id}.mp4')

        print('args.keyframe_path',args.keyframe_path)
        if args.keyframe_path!='None':
            text_path = f"{args.keyframe_path}/{sample_set['id']}.txt"
            print(text_path)
            if not os.path.exists(text_path):
                print('Keyframe file not found for sample id:', sample_set['id'])
                continue
            with open(text_path, "r", encoding="utf-8") as f:
                lines = f.readlines() 
                keyframes = [int(line.strip()) for line in lines if line.strip()]
        else:
            keyframes = []
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,      
                        "keyframes":keyframes
                    },
                    {"type": "text", "text": question},
                ],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos, video_kwargs = process_vision_info(messages, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True)
        if videos is not None:
            videos, video_metadatas = zip(*videos)
            videos, video_metadatas = list(videos), list(video_metadatas)
        else:
            video_metadatas = None

        # since qwen-vl-utils has resize the images/videos, \
        # we should pass do_resize=False to avoid duplicate operation in processor!
        inputs = processor(text=text, images=images, videos=videos, video_metadata=video_metadatas, return_tensors="pt", do_resize=False, **video_kwargs)
        # inputs = inputs.to(model.device)
        inputs = inputs.to(model.device)
        # for key, value in inputs.items():
        #     print(key,value.shape)
        '''
        Shape of input_ids: torch.Size([1, 12090])
        Shape of attention_mask: torch.Size([1, 12090])
        input_ids torch.Size([1, 12090])
        attention_mask torch.Size([1, 12090])
        pixel_values_videos torch.Size([45288, 1536])
        video_grid_thw torch.Size([1, 3])
        '''
        do_sample=False
        generation_kwargs = {
            "do_sample": do_sample,
            "top_p": float(os.getenv("top_p", 0.8)),
            "top_k": int(os.getenv("top_k", 20)),
            "temperature": float(os.getenv("temperature", 0.7)),
            "repetition_penalty": float(os.getenv("repetition_penalty", 1.0)),
            "max_new_tokens": int(os.getenv("out_seq_length", 32)), 
        }
        generated_ids = model.generate(**inputs, **generation_kwargs)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        outputs = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        print(outputs)
        parsed_pred = parse_multi_choice_response(outputs, ["A", "B", "C", "D","E"], index2ans)
        sample_set['acc'] = str(parsed_pred == answer_id)   
        sample_set["pred"] = outputs
        print('parsed_pred', parsed_pred)
        print('outputs', outputs)
        print('sample_set[acc]', sample_set['acc'])
        ans_file.write(json.dumps(sample_set)+ "\n")
    ans_file.close()

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", help="Directory containing video files.", required=True)
    parser.add_argument("--gt_file", help="Path to the ground truth file containing question and answer.", required=True)
    parser.add_argument("--output_dir", help="Directory to save the model results JSON.", required=True)
    parser.add_argument("--output_name", help="Name of the file for storing results JSON.", required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--keyframe_path", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_inference(args)
