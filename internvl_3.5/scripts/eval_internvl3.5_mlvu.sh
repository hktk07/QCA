
#!/bin/bash
set -euo pipefail

task=mlvu
model=intervl-3.5-8b

video_dir=/data/jsb/datasets/MLVU/video
gt_file=/data/jsb/datasets/MLVU/annotations/multiple_choice.json

# clips=(12 16 20)
# taus=(0.5 0.6 0.7 0.8)
# as=(0.3) 
clips=(12 )
taus=(0.5)
as=(0.3) 

# --------- GPU 配置部分（改这里）---------
# ✅ 指定使用的 GPU（非连续也可以）
GPU_IDS=(0 1 2 3)     # 例如只用 0,2,3,5 四张卡

GPUS_PER_TASK=1       # 每个进程占几张卡（通常=1）
NUM_GPUS=${#GPU_IDS[@]}
CHUNKS=$((NUM_GPUS / GPUS_PER_TASK))

if (( NUM_GPUS % GPUS_PER_TASK != 0 )); then
  echo "Error: GPU_IDS数量($NUM_GPUS) 不能被 GPUS_PER_TASK($GPUS_PER_TASK) 整除"
  exit 1
fi

echo "Use GPU IDs: ${GPU_IDS[*]}"
echo "GPUS_PER_TASK=$GPUS_PER_TASK, total processes(CHUNKS)=$CHUNKS"
# ----------------------------------------

keyframe_path_short="None"
failed_list=()

for clip in "${clips[@]}"; do
  for tau in "${taus[@]}"; do
    for a in "${as[@]}"; do
      # base_output_name="mlvu_clip_${clip}_tau_${tau}_alpha_${a}trace_softmax"
      # base_output_name="mlvu_clip_${clip}_tau_${tau}_trace_softmax"
      base_output_name="mlvu_topk"
      output_dir="qwen3-vl/output/${model}/${base_output_name}"
      mkdir -p "$output_dir"

      # keyframe_path="mlvu_keyframes/mlvu_clip_${clip}_tau_${tau}_alpha_${a}trace_softmax"
      # keyframe_path="mlvu_keyframes/mlvu_clip_${clip}_tau_${tau}_trace_softmax"
      keyframe_path="mlvu_keyframes/topk"
      log_dir="${output_dir}/logs"
      mkdir -p "$log_dir"

      echo "=============================="
      echo "Run combo: clip=$clip, tau=$tau, alpha=$a"
      echo "Output dir: $output_dir"
      echo "Keyframe path: $keyframe_path"
      echo "=============================="

      if [[ ! -d "$keyframe_path" ]]; then
        echo "[WARN] keyframe_path not found, skip: $keyframe_path"
        failed_list+=("clip=$clip tau=$tau alpha=$a (missing keyframes)")
        continue
      fi

      set +e
      combo_ok=1
      pids=()

      for IDX in $(seq 0 $((CHUNKS-1))); do
        # ✅ 关键改动：从 GPU_IDS 里取，不再用 GPU_START_ID 连续分配
        offset=$((IDX * GPUS_PER_TASK))
        gpu_slice=("${GPU_IDS[@]:offset:GPUS_PER_TASK}")

        if (( ${#gpu_slice[@]} != GPUS_PER_TASK )); then
          echo "[ERROR] GPU slice size mismatch at chunk $IDX"
          combo_ok=0
          break
        fi

        CUDA_DEVICES=$(IFS=,; echo "${gpu_slice[*]}")
        echo "Launch chunk $IDX on GPU(s): $CUDA_DEVICES"

        log_file="${log_dir}/${base_output_name}_chunk${IDX}.log"

        CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
        python3 qwen3-vl/internvl-3.5_mlvu.py \
          --video_dir "$video_dir" \
          --gt_file "$gt_file" \
          --output_dir "$output_dir" \
          --output_name "${base_output_name}_chunk${IDX}" \
          --num_chunks "$CHUNKS" \
          --keyframe_path "$keyframe_path" \
          --keyframe_path_short "$keyframe_path_short" \
          --chunk_idx "$IDX" \
          2>&1 | tee "$log_file" &

        pids+=($!)
      done

      for pid in "${pids[@]}"; do
        wait "$pid"
        if [[ $? -ne 0 ]]; then
          combo_ok=0
        fi
      done

      if [[ $combo_ok -ne 1 ]]; then
        echo "[WARN] combo failed: clip=$clip tau=$tau alpha=$a (skip merge/eval, continue)"
        failed_list+=("clip=$clip tau=$tau alpha=$a (infer)")
        set -e
        continue
      fi

      echo "All chunks finished. Merging jsonlines..."
      cat "${output_dir}/${base_output_name}_chunk"*.json > "${output_dir}/${base_output_name}.json"
      if [[ $? -ne 0 ]]; then
        echo "[WARN] merge failed: clip=$clip tau=$tau alpha=$a (continue)"
        failed_list+=("clip=$clip tau=$tau alpha=$a (merge)")
        set -e
        continue
      fi

      python3 qwen3-vl/eval/eval_mlvu.py \
        --num-chunks "$CHUNKS" \
        --output-dir "$output_dir"
      if [[ $? -ne 0 ]]; then
        echo "[WARN] eval failed: clip=$clip tau=$tau alpha=$a (continue)"
        failed_list+=("clip=$clip tau=$tau alpha=$a (eval)")
        set -e
        continue
      fi

      set -e
      echo "Merged file: ${output_dir}/${base_output_name}.json"
      echo "Done combo: clip=$clip tau=$tau alpha=$a"
      echo
    done
  done
done

echo "=============================="
echo "All combos finished."
if ((${#failed_list[@]})); then
  echo "[SUMMARY] Failed combos:"
  printf '  - %s\n' "${failed_list[@]}"
else
  echo "[SUMMARY] All combos succeeded."
fi
