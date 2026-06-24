#!/bin/bash
set -euo pipefail
task=Videomme
model=qwen3-vl-8b
model_path=/models/${model}
video_dir="/datasets/Video-MME/data"
gt_file="/datasets/Video-MME/videomme/test_anns.csv"
CHUNKS=8
NUM_GPUS=8
GPU_START_ID=0

echo "Use $NUM_GPUS GPUs, split data into $CHUNKS chunks."
base_output_name="output_name"
output_dir="/output/${model}/videomme_${base_output_name}"
mkdir -p "$output_dir"
keyframe_path="keyframe_path"
log_dir="${output_dir}/logs"
mkdir -p "$log_dir"

# multi GPU run multi chunks
for IDX in $(seq 0 $((CHUNKS-1))); do
  GPU_ID=$((GPU_START_ID + IDX))

  echo "Launch chunk $IDX on GPU $GPU_ID"

  log_file="${log_dir}/${base_output_name}_chunk${IDX}.log"

  CUDA_VISIBLE_DEVICES=$GPU_ID \
  python3 /qwen3-vl/qwen3_videomme.py \
    --model_path "$model_path" \
    --video_dir "$video_dir" \
    --gt_file "$gt_file" \
    --output_dir "$output_dir" \
    --output_name "${base_output_name}_chunk${IDX}" \
    --num_chunks "$CHUNKS" \
    --keyframe_path "$keyframe_path" \
    --chunk_idx "$IDX" \
    2>&1 | tee "$log_file" &

python3 /qwen3-vl/eval/eval_videomme.py \
  --num-chunks "$CHUNKS" \
  --output-dir "$output_dir"

