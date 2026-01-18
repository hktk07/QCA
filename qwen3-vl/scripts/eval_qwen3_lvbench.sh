




#!/bin/bash
set -euo pipefail

ROOT_DIR="/mntqwen3-vl"
task=lvbench
model=qwen3-vl-8b

model_path="qwen3-vl/${model}"
video_dir="AKS/datasets/lvbench/videos"
gt_file="AKS/datasets/lvbench/qa_file.json"

# ========= 这里填你要跑的多个值 =========
clips=(16)
taus=(0.5)
as=(0 0.1 0.3 0.7 0.9 1)
# clips=(12 16 20)
# # taus=(0.7 0.8)
# taus=(0.5 0.6)
# as=(0.5)
# clips=(12)
# taus=(0.5 )
# as=(0.5)

# ======================================

# ------------ 关键参数：8 卡数据并行 ------------
CHUNKS=8
NUM_GPUS=8
GPU_START_ID=0

echo "Use $NUM_GPUS GPUs, split data into $CHUNKS chunks."

failed_list=()

# ============ 组合循环：clip x tau ============
for clip in "${clips[@]}"; do
  for tau in "${taus[@]}"; do
    for a in "${as[@]}"; do
      # base_output_name="lvbench_clip_${clip}_tau_${tau}_alpha_${a}trace_softmax"
      base_output_name="lvbench_random_anchor"
      # base_output_name="lvbench_clip_${clip}_tau_${tau}_trace_softmax_no_do_sample"
      # base_output_name="lvbench_uni"
      # base_output_name="aks"
      # base_output_name="lvbench_clip_${clip}_no_stage2"
      output_dir="qwen3-vl/output/${model}/${base_output_name}"
      # mkdir -p "$output_dir"

      # keyframe_path="lvbench_keyframes/lvbench_clip_${clip}_tau_${tau}_alpha_${a}trace_softmax"
      keyframe_path="lvbench_keyframes/lvbench_random_anchor"
      # keyframe_path="lvbench_keyframes/lvbench_clip_${clip}_tau_${tau}_trace_softmax"
      # # # keyframe_path="lvbench_keyframes/lvbench_clip_${clip}_no_stage2"
      # # # keyframe_path=None
      # # # keyframe_path='lvbench_keyframes/lvbench_aks'
      log_dir="${output_dir}/logs"
      mkdir -p "$log_dir"

      echo "=============================="
      echo "Run combo: clip=$clip, tau=$tau"
      echo "Output dir: $output_dir"
      echo "Keyframe path: $keyframe_path"
      echo "=============================="

      # （可选）keyframe_path 不存在就跳过该组合
      # if [[ ! -d "$keyframe_path" ]]; then
      #   echo "[WARN] keyframe_path not found, skip: $keyframe_path"
      #   failed_list+=("clip=$clip tau=$tau (missing keyframes)")
      #   continue
      # fi

      # ---- 关键：这个组合内部允许失败，不退出整个脚本 ----
      set +e
      combo_ok=1
      pids=()

      # 启动 8 个进程，每个进程用一张卡、处理一个 chunk
      for IDX in $(seq 0 $((CHUNKS-1))); do
        GPU_ID=$((GPU_START_ID + IDX))  # 默认 0..7

        echo "Launch chunk $IDX on GPU $GPU_ID"

        log_file="${log_dir}/${base_output_name}_chunk${IDX}.log"

        CUDA_VISIBLE_DEVICES=$GPU_ID \
        python3 qwen3-vl/qwen3_lvbench.py \
          --model_path "$model_path" \
          --video_dir "$video_dir" \
          --gt_file "$gt_file" \
          --output_dir "$output_dir" \
          --output_name "${base_output_name}_chunk${IDX}" \
          --num_chunks "$CHUNKS" \
          --keyframe_path "$keyframe_path" \
          --chunk_idx "$IDX" \
          2>&1 | tee "$log_file" &

        pids+=($!)
      done

      # wait 任意一个 chunk 失败都标记 combo_ok=0（但不退出）
      for pid in "${pids[@]}"; do
        wait "$pid"
        if [[ $? -ne 0 ]]; then
          combo_ok=0
        fi
      done

      if [[ $combo_ok -ne 1 ]]; then
        echo "[WARN] combo failed: clip=$clip tau=$tau (skip merge/eval, continue)"
        failed_list+=("clip=$clip tau=$tau (infer)")
        set -e
        continue
      fi

      echo "All chunks finished for clip=$clip tau=$tau. Merging jsonlines..."

      cat "${output_dir}/${base_output_name}_chunk"*.json > "${output_dir}/${base_output_name}.json"
      if [[ $? -ne 0 ]]; then
        echo "[WARN] merge failed: clip=$clip tau=$tau (continue)"
        failed_list+=("clip=$clip tau=$tau (merge)")
        set -e
        continue
      fi

      python3 qwen3-vl/eval/eval_lvbench.py \
        --num-chunks "$CHUNKS" \
        --output-dir "$output_dir"
      if [[ $? -ne 0 ]]; then
        echo "[WARN] eval failed: clip=$clip tau=$tau (continue)"
        failed_list+=("clip=$clip tau=$tau (eval)")
        set -e
        continue
      fi

      set -e
      echo "Merged file: ${output_dir}/${base_output_name}.json"
      echo "Done combo: clip=$clip tau=$tau"
      echo
    done
  done
done
# =============================================

echo "=============================="
echo "All combos finished."
if ((${#failed_list[@]})); then
  echo "[SUMMARY] Failed combos:"
  printf '  - %s\n' "${failed_list[@]}"
else
  echo "[SUMMARY] All combos succeeded."
fi
