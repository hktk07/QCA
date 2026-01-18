import torch
import transformers
import json
import pandas as pd
from glob import glob
from decord import VideoReader, cpu
import numpy as np
import argparse
from collections import defaultdict
import os

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
    # print('files', files)
    if len(files) == 0:
        files = glob(ori_dir + f'/*')
    # print(files)
    total = 0
    acc = 0
    for file in files:
        # file = os.path.join(file,'pred.json')
        lines = open(file, 'r').readlines()
        t = 0
        a = 0
        for line in lines:
            t += 1
            if eval(line)["acc"] == "True":
                a += 1

        total += t
        acc += a
    print(total, acc)
    print(acc / total)

if __name__ == "__main__":
    main()