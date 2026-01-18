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
        short16_cnt, short60_cnt, medium_cnt, long_cnt = 0, 0, 0, 0
        short16_acc, short60_acc, medium_acc, long_acc = 0, 0, 0, 0
        # duration_group = 60
        for file in files:
            lines = open(file, 'r').readlines()
            for line in lines:
                line = eval(line)
                total_cnt += 1
                duration_group = line['duration_group']
                if duration_group == 15:
                    short16_cnt += 1
                elif duration_group == 60:
                    short60_cnt += 1
                elif duration_group == 600:
                    medium_cnt += 1
                elif duration_group == 3600:
                    long_cnt += 1

                if line['acc'] == 'True':
                    total_acc += 1
                    if duration_group == 15:
                        short16_acc += 1
                    elif duration_group == 60:
                        short60_acc += 1
                    elif duration_group == 600:
                        medium_acc += 1
                    elif duration_group == 3600:
                        long_acc += 1
                        
        assert short60_cnt + short16_cnt + medium_cnt + long_cnt == total_cnt
        if short16_cnt > 0:
            print(f'short acc: {short16_acc / short16_cnt}')
            print(f'short_cnt: {short16_cnt}')
        if short60_cnt > 0:
            print(f'short acc: {short60_acc / short60_cnt}')
            print(f'short_cnt: {short60_cnt}')
        if medium_cnt > 0:
            print(f'medium acc: {medium_acc / medium_cnt}')
            print(f'medium_cnt: {medium_cnt}')
        if long_cnt > 0:
            print(f'long acc: {long_acc / long_cnt}')
            print(f'long_cnt: {long_cnt}')
        print(f'total acc: {total_acc / total_cnt}, total_cnt: {total_cnt}')
        

if __name__ == "__main__":
    main()