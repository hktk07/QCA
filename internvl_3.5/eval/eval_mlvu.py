import torch
import transformers
import json
import pandas as pd
from glob import glob
from decord import VideoReader, cpu

# try:
#     from pywsd.utils import lemmatize_sentence
# except ImportError:
#     eval_logger.debug("pywsd not installed. Please install pywsd to use this module. You can install it by running 'pip install pywsd'")

# from nltk.tokenize import word_tokenize
# from nltk.corpus import wordnet
# try:
#     from nltk.tokenize import word_tokenize
#     from nltk.corpus import wordnet

#     try:
#         import nltk

#         nltk.download("averaged_perceptron_tagger", quiet=True)
#         nltk.download("wordnet", quiet=True)
#         nltk.download("punkt", quiet=True)
#     except Exception as e:
#         eval_logger.debug(f"nltk download failed: {e}")
# except ImportError:
#     eval_logger.debug("nltk not installed. Please install nltk to use this module. You can install it by running 'pip install nltk'")
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="question-answer-generation-using-gpt-3")
    parser.add_argument("--output-dir", default=r'', help="The path to save annotation json files.")
    parser.add_argument("--eval-type", default="multi_choice", help="The path to save annotation final combined json file.")
    parser.add_argument("--num-chunks", type=int, default=1)
    args = parser.parse_args()
    return args


def main():
    """
    Main function to control the flow of the program.
    """
    # Parse arguments.
    args = parse_args()

    # wups, multi_choice
    pred_type=args.eval_type
    ori_dir = args.output_dir
    files = glob(ori_dir + f'/{args.num_chunks}_*')
    if len(files) == 0:
        files = glob(ori_dir + f'/*')
    # files = glob(ori_dir + f'/res.json')
    print(files)
    
    if pred_type == 'multi_choice':
        total_cnt, total_acc = 0, 0
        plotqa_cnt, needle_cnt, ego_cnt, count_cnt, order_cnt, anoma_cnt, topic_cnt = 0, 0, 0, 0, 0, 0, 0 
        plotqa_acc, needle_acc, ego_acc, count_acc, order_acc, anoma_acc, topic_acc = 0, 0, 0, 0, 0, 0, 0
        for file in files:
            lines = open(file, 'r').readlines()
            for line in lines:
                line = eval(line)
                total_cnt += 1
                category = line['video_id'].split('/')[0]
                if category == '1_plotQA':
                    plotqa_cnt += 1
                elif category == '2_needle':
                    needle_cnt += 1
                elif category == '3_ego':
                    ego_cnt += 1
                elif category == '4_count':
                    count_cnt += 1
                elif category == '5_order':
                    order_cnt += 1
                elif category == '6_anomaly_reco':
                    anoma_cnt += 1
                elif category == '7_topic_reasoning':
                    topic_cnt += 1

                if line['acc'] == 'True':
                # if line['answer_id'] == line['pred'][0]:
                # if chr(ord("A")+line["answer_id"]) == line['pred'].rstrip('.'):
                    total_acc += 1
                    if category == '1_plotQA':
                        plotqa_acc += 1
                    elif category == '2_needle':
                        needle_acc += 1
                    elif category == '3_ego':
                        ego_acc += 1
                    elif category == '4_count':
                        count_acc += 1
                    elif category == '5_order':
                        order_acc += 1
                    elif category == '6_anomaly_reco':
                        anoma_acc += 1
                    elif category == '7_topic_reasoning':
                        topic_acc += 1
                        
        assert plotqa_cnt + needle_cnt + ego_cnt + count_cnt + order_cnt + anoma_cnt + topic_cnt == total_cnt
        if plotqa_cnt > 0:
            print(f'plotqa acc: {plotqa_acc / plotqa_cnt}')
            print(f'plotqa_cnt: {plotqa_cnt}')
        if needle_cnt > 0:
            print(f'needle acc: {needle_acc / needle_cnt}')
            print(f'needle_cnt: {needle_cnt}')
        if ego_cnt > 0:
            print(f'ego acc: {ego_acc / ego_cnt}')
            print(f'ego_cnt: {ego_cnt}')
        if count_cnt > 0:
            print(f'count acc: {count_acc / count_cnt}')
            print(f'count_cnt: {count_cnt}')
        if order_cnt > 0:
            print(f'order acc: {order_acc / order_cnt}')
            print(f'order_cnt: {order_cnt}')
        if anoma_cnt > 0:
            print(f'anoma acc: {anoma_acc / anoma_cnt}')
            print(f'anoma_cnt: {anoma_cnt}')
        if topic_cnt > 0:
            print(f'topic acc: {topic_acc / topic_cnt}')
            print(f'topic_cnt: {topic_cnt}')
        print(f'single detail acc: {(plotqa_acc+needle_acc+ego_acc)/(plotqa_cnt+needle_cnt+ego_cnt)}')
        print(f'multi detail acc: {(count_acc+order_acc)/(count_cnt+order_cnt)}')
        print(f'holistic acc: {(topic_acc+anoma_acc)/(topic_cnt+anoma_cnt)}')
        print(f'total acc: {total_acc / total_cnt}, total_cnt: {total_cnt}')
        

if __name__ == "__main__":
    main()