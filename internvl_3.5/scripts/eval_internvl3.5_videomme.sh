
#!/bin/bash
set -euo pipefail

task=videomme
model=intervl-3.5-8b

video_dir=/data/jsb/datasets/Video-MME/data
gt_file=/data/jsb/datasets/Video-MME/videomme/test_anns.csv

# ========= 这里填你要跑的多个值 =========
clips=(12)
taus=(0.5)
as=(0.1)
# ======================================

# --------- GPU 配置部分 ---------
NUM_GPUS=4
GPUS_PER_TASK=1
CHUNKS=$((NUM_GPUS / GPUS_PER_TASK))
GPU_START_ID=4

if (( NUM_GPUS % GPUS_PER_TASK != 0 )); then
  echo "Error: NUM_GPUS($NUM_GPUS) 不能被 GPUS_PER_TASK($GPUS_PER_TASK) 整除"
  exit 1
fi

echo "Use $NUM_GPUS GPUs (IDs ${GPU_START_ID}-$((GPU_START_ID+NUM_GPUS-1))), $GPUS_PER_TASK GPUs per process."
echo "Split data into $CHUNKS chunks (one chunk per process)."

# 你原脚本里固定的 short keyframe
# keyframe_path_short="videomme_keyframes/videomme_short_64frame_6clip"

failed_list=()

# ============ 组合循环：clip x tau ============
for clip in "${clips[@]}"; do
  for tau in "${taus[@]}"; do
    for a in "${as[@]}"; do
    # ✅ 每个组合独立命名，避免覆盖
      # base_output_name="videomme_clip_${clip}_tau_${tau}_alpha_${a}trace_softmax"
      base_output_name="videomme_topk"
      output_dir="qwen3-vl/output/${model}/${base_output_name}"
      mkdir -p "$output_dir"

      # # ✅ 你按组合变化的 keyframe_path（按你实际目录命名改这里）
      # keyframe_path="videomme_keyframes/videomme_clip_${clip}_tau_${tau}_alpha_${a}trace_softmax"
      keyframe_path="videomme_keyframes/topk"
      log_dir="${output_dir}/logs"
      mkdir -p "$log_dir"

      echo "=============================="
      echo "Run combo: clip=$clip, tau=$tau"
      echo "Output dir: $output_dir"
      echo "Keyframe path: $keyframe_path"
      # echo "Keyframe path short: $keyframe_path_short"
      echo "=============================="

      # （可选）keyframe_path 不存在就跳过该组合
      if [[ ! -d "$keyframe_path" ]]; then
        echo "[WARN] keyframe_path not found, skip: $keyframe_path"
        failed_list+=("clip=$clip tau=$tau (missing keyframes)")
        continue
      fi

      # ---- 关键：这个组合内部允许失败，不退出整个脚本 ----
      set +e
      combo_ok=1
      pids=()

      # 启动若干个进程，每个进程占用一张卡、处理一个 chunk
      for IDX in $(seq 0 $((CHUNKS-1))); do
        GPU0=$((GPU_START_ID + IDX * GPUS_PER_TASK))
        echo "Launch chunk $IDX on GPU $GPU0"

        log_file="${log_dir}/${base_output_name}_chunk${IDX}.log"

        CUDA_VISIBLE_DEVICES=${GPU0} \
        python3 qwen3-vl/internvl-3.5_videomme.py \
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

      python3 qwen3-vl/eval/eval_videomme.py \
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
