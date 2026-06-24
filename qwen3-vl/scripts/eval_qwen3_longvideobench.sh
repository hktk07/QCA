#!/bin/bash
set -euo pipefail
task=longvideobench
model=qwen3-vl-8b
model_path=/models/qwen3-vl-8b
video_dir=/datasets/LongVideoBench/videos
gt_file=/datasets/LongVideoBench/lvb_val.json

GPUS=(0 1 2 3 4 5 6 7)    
CHUNKS=${#GPUS[@]}  
echo "Use GPUs: ${GPUS[*]}"
echo "Total chunks: $CHUNKS"

base_output_name="frame_64_clip_12_gemma_0.5_beta_0.5_softmax_tau_1"
keyframe_path="/longvideobench_keyframes/${base_output_name}"
output_dir="/output/${model}/longvideobench_${base_output_name}"
mkdir -p "$output_dir"

# multi GPU run multi chunks
for IDX in $(seq 0 $((CHUNKS-1))); do
  GPU_ID=${GPUS[$IDX]}
  echo "Launch chunk $IDX on GPU $GPU_ID"
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  CUDA_VISIBLE_DEVICES=$GPU_ID \
  python3 /qwen3-vl/qwen3_longvideobench.py \
    --model_path "$model_path" \
    --video_dir "$video_dir" \
    --gt_file "$gt_file" \
    --output_dir "$output_dir" \
    --output_name "${base_output_name}_chunk${IDX}" \
    --num_chunks "$CHUNKS" \
    --keyframe_path "$keyframe_path" \
    --chunk_idx "$IDX" &

python3 /qwen3-vl/eval/eval_longvideobench.py \
  --num-chunks "$CHUNKS" \
  --output-dir "$output_dir"

