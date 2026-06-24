#!/bin/bash
set -euo pipefail
task=mlvu
model=qwen3-vl-8b
model_path=/models/$model
video_dir=/datasets/MLVU/video
gt_file=/datasets/MLVU/annotations/multiple_choice.json

CHUNKS=8
NUM_GPUS=8
GPU_OFFSET=0   

base_output_name="output_name"
keyframe_path="keyframe_path"
output_dir="/output/${model}/${base_output_name}"
mkdir -p "$output_dir"

# multi GPU run multi chunks
for IDX in $(seq 0 $((CHUNKS-1))); do
  GPU_ID=$((IDX + GPU_OFFSET))
  echo "Launch chunk $IDX on GPU $GPU_ID"
  CUDA_VISIBLE_DEVICES=$GPU_ID \
  python3 /qwen3-vl/qwen3_mlvu.py \
    --model_path "$model_path" \
    --video_dir "$video_dir" \
    --gt_file "$gt_file" \
    --output_dir "$output_dir" \
    --output_name "${base_output_name}_chunk${IDX}" \
    --num_chunks "$CHUNKS" \
    --keyframe_path "$keyframe_path" \
    --chunk_idx "$IDX" &

python3 /qwen3-vl/eval/eval_mlvu.py \
  --num-chunks "$CHUNKS" \
  --output-dir "$output_dir"
