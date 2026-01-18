
#!/bin/bash
set -euo pipefail

task=lvbench
model=intervl-3.5-8b

video_dir=AKS/datasets/lvbench/videos
gt_file=AKS/datasets/lvbench/qa_file.json

# ========= 这里填你要跑的多个值 =========
# clips=(12 16 20)
# taus=(0.5 0.6 0.7 0.8)
# as=(0.1)
clips=(12)
taus=(0.5)
as=(0.1)
# ======================================

# --------- GPU 配置部分（显式指定 GPU 列表） ---------
GPUS_PER_TASK=1
GPU_IDS=(0 1 2)          # ✅ 只使用这几张 GPU（物理 GPU ID）
NUM_GPUS=${#GPU_IDS[@]}
CHUNKS=$((NUM_GPUS / GPUS_PER_TASK))

if (( NUM_GPUS % GPUS_PER_TASK != 0 )); then
  echo "Error: NUM_GPUS($NUM_GPUS) 不能被 GPUS_PER_TASK($GPUS_PER_TASK) 整除"
  exit 1
fi

echo "Use GPUs: ${GPU_IDS[*]}, $GPUS_PER_TASK GPUs per process."
echo "Split data into $CHUNKS chunks (one chunk per process)."

failed_list=()

# ============ 组合循环：clip x tau x alpha ============
for clip in "${clips[@]}"; do
  for tau in "${taus[@]}"; do
    for a in "${as[@]}"; do
      # ✅ 每个组合独立命名，避免覆盖
      # base_output_name="lvbench_clip_${clip}_tau_${tau}_alpha_${a}trace_softmax"
      # base_output_name="lvbench_clip_${clip}_tau_${tau}_trace_softmax"
      # base_output_name="lvbench_qframe"
      base_output_name="lvbench_qframe"
      output_dir="qwen3-vl/output/${model}/${base_output_name}"
      mkdir -p "$output_dir"

      # ✅ 你按组合变化的 keyframe_path（按你实际目录命名改这里）
      # keyframe_path="lvbench_keyframes/lvbench_clip_${clip}_tau_${tau}_alpha_${a}trace_softmax"
      # keyframe_path="lvbench_keyframes/lvbench_clip_${clip}_tau_${tau}_trace_softmax"
      keyframe_path="lvbench_keyframes/qframe"

      # log_dir="${output_dir}/logs"
      # mkdir -p "$log_dir"

      # echo "=============================="
      # echo "Run combo: clip=$clip, tau=$tau, alpha=$a"
      # echo "Output dir: $output_dir"
      # echo "Keyframe path: $keyframe_path"
      # echo "=============================="

      # # （可选）keyframe_path 不存在就跳过该组合
      # # if [[ ! -d "$keyframe_path" ]]; then
      # #   echo "[WARN] keyframe_path not found, skip: $keyframe_path"
      # #   failed_list+=("clip=$clip tau=$tau alpha=$a (missing keyframes)")
      # #   continue
      # # fi

      # # ---- 关键：这个组合内部允许失败，不退出整个脚本 ----
      # set +e
      # combo_ok=1
      # pids=()

      # # 打乱 GPU 分配顺序（长度为 CHUNKS）
      # mapfile -t RIDX_LIST < <(seq 0 $((CHUNKS-1)) | shuf)

      # # 启动若干个进程，每个进程占用一张卡、处理一个 chunk
      # for IDX in $(seq 0 $((CHUNKS-1))); do
      #   RIDX=${RIDX_LIST[$IDX]}
      #   GPU0=${GPU_IDS[$RIDX]}   # ✅ 从显式 GPU 列表里取卡

      #   echo "Launch chunk $IDX on GPU $GPU0"

      #   log_file="${log_dir}/${base_output_name}_chunk${IDX}.log"

      #   CUDA_VISIBLE_DEVICES=${GPU0} \
      #   python3 qwen3-vl/internvl-3.5_lvbench.py \
      #     --video_dir "$video_dir" \
      #     --gt_file "$gt_file" \
      #     --output_dir "$output_dir" \
      #     --output_name "${base_output_name}_chunk${IDX}" \
      #     --num_chunks "$CHUNKS" \
      #     --keyframe_path "$keyframe_path" \
      #     --chunk_idx "$IDX" \
      #     2>&1 | tee "$log_file" &

      #   pids+=($!)
      # done

      # # wait：任意一个 chunk 失败都标记 combo_ok=0（但不退出）
      # for pid in "${pids[@]}"; do
      #   wait "$pid"
      #   if [[ $? -ne 0 ]]; then
      #     combo_ok=0
      #   fi
      # done

      # if [[ $combo_ok -ne 1 ]]; then
      #   echo "[WARN] combo failed: clip=$clip tau=$tau alpha=$a (skip merge/eval, continue)"
      #   failed_list+=("clip=$clip tau=$tau alpha=$a (infer)")
      #   set -e
      #   continue
      # fi

      # echo "All chunks finished for clip=$clip tau=$tau alpha=$a. Merging jsonlines..."
      # cat "${output_dir}/${base_output_name}_chunk"*.json > "${output_dir}/${base_output_name}.json"
      # if [[ $? -ne 0 ]]; then
      #   echo "[WARN] merge failed: clip=$clip tau=$tau alpha=$a (continue)"
      #   failed_list+=("clip=$clip tau=$tau alpha=$a (merge)")
      #   set -e
      #   continue
      # fi

      python3 qwen3-vl/eval/eval_lvbench.py \
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
# =============================================

echo "=============================="
echo "All combos finished."
if ((${#failed_list[@]})); then
  echo "[SUMMARY] Failed combos:"
  printf '  - %s\n' "${failed_list[@]}"
else
  echo "[SUMMARY] All combos succeeded."
fi
