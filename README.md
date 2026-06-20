# ECCV 2026 QCA: Query- and Content-Aware Keyframe Selection for Efficient Long Video Understanding
This is the official implementaion of paper **QCA: Query- and Content-Aware Keyframe Selection for Efficient Long Video Understanding**
## Abstract
Video understanding is often plagued by severe temporal redundancy, where processing dense frame sequences is both semantically inefficient and computationally expensive.
This challenge is further amplified when only a small subset of frames is truly relevant to the given query.
In this paper, we propose a \textbf{Q}uery- and \textbf{C}ontent-\textbf{A}ware (\textbf{QCA}) keyframe selection framework that can select a compact yet information-rich set of frames from long videos.
QCA first partitions the video into temporal segments and evaluates the information contribution of each segment by jointly modeling the query-frame semantic matching degree and segment content deviation.
The normalized contribution scores dynamically determine the keyframe budget allocation for each segment. Within each segment, QCA anchors on the most query-relevant frame and iteratively incorporates additional frames to maximize diversity while maintaining high semantic relevance to the query.
Crucially, our method requires no additional training and can be seamlessly integrated into existing Video-LLMs. Extensive experiments across multiple long video understanding benchmarks demonstrate that our proposed approach achieves state-of-the-art performance and has strong generalization ability. For instance, QCA achieves 67.8\% on LongVideoBench using 128 frames, while GPT-4o achieves 66.7\% using 256 frames.
## Overview
<p align="center">
    <img src="./assets/overall_new_cropped (4) (1).png" width="80%"></a> <br>
    The overall framework of our approach.
</p>

## Dataset 

LongVideoBench

VideoMME

MLVU

LVBench

## Extract visual feature and compute ITM Score 
We use the BLIP/CLIP to extract the visual feature of video frames and compute the ITM score between frames and question.
For BLIP we use blip_feature_ITM_score.py e.g.,
```python
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --standalone --nproc_per_node=4 \
blip_feature_ITM_score.py \
--dataset_name MLVU \
--dataset_path your_mlvu_dataset_path \
--extract_feature_model blip \
--output_dir your_output_dir \
--device cuda \
--min_frames 128 \
--batch_size 128 \
--num_workers 4
```

For CLIP we use clip_feature_ITM_score.py e.g.,
```python
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --standalone --nproc_per_node=4 \
clip_feature_ITM_score.py \
--dataset_name MLVU \
--dataset_path /data/jsb/datasets/MLVU \
--extract_feature_model clip \
--output_dir clip_feature_mlvu_test \
--device cuda \
--min_frames 128 \
--batch_size 128 \
--num_workers 4
```

## QCA select keyframes
```python
python keyframe_select.py --dataset_name [dataset_name] --dataset_path [your dataset path] --num_segments [number of segments(e.g. 16)] --total_keep [number of all keyframes(e.g. 64)] --tau [0.6] --alpha[0.5] --output_dir [your output dir]
```

## qwen3-vl test
```python
cd qwen3-vl
bash scripts/eval_qwen3_{dataset}.sh
```

## qwen3-vl test
```python
cd internvl_3.5
bash scripts/eval_internvl3.5_{dataset}.sh
```
