
#!/bin/bash
ROOT_DIR="/mntqwen3-vl"

task=longvideobench
model=qwen3-vl-8b

model_path=qwen3-vl/$model
video_dir=/data/jsb/datasets/LongVideoBench/videos
gt_file=/data/jsb/datasets/LongVideoBench/lvb_val.json

# base_output_name=longvideobench_frame_stage1_clip_trace_sim_set_not_sim_less_64_12clip
# base_output_name=aks_64
# base_output_name=longvideobench_64frames_uniform
# 输出目录 & 基础名字（注意这里只是名字，不是全路径）
# output_dir=qwen3-vl/output/$model/$base_output_name
output_dir=qwen3-vl/output/$model/$base_output_name
mkdir -p "$output_dir"
# clips=(12 16 20)
# taus=(0.5 0.6 0.7 0.8)
# as=(0.5)
clips=(8)
taus=(0.6)
as=(0.5)
# ------------ 关键参数：8 卡数据并行 ------------
CHUNKS=4   # 把数据切成 8 份
NUM_GPUS=4 # 使用 8 张 GPU
GPU_START_ID=4
# method='ours_64'
keyframe_path='qwen3-vl/longvideobench_keyframe/longvideobench_frame_stage1_clip_trace_sim_set_not_sim_less_64_12clip'
# keyframe_path='qwen3-vl/videomme_keyframe/aks_64'
# keyframe_path=None
# -------------------------------------------


echo "Use $NUM_GPUS GPUs, split data into $CHUNKS chunks."

# 启动 8 个进程，每个进程用一张卡、处理一个 chunk
for clip in "${clips[@]}"; do
  for tau in "${taus[@]}"; do
    # for a in "${as[@]}"; do
    # base_output_name="longvideobench_clip_${clip}_no_stage2"
    # base_output_name=longvideobench_clip_${clip}_tau_${tau}_trace_softmax_no_do_sample
    base_output_name=longvideobench_test_time
    # base_output_name=longvideobench_one-clip
    # base_output_name=
    output_dir="qwen3-vl/output/${model}/${base_output_name}"
    mkdir -p "$output_dir"

    # keyframe_path="longvideobench_keyframes/longvideobench_clip_${clip}_tau_${tau}_trace_softmax"
    # keyframe_path="longvideobench_keyframes/longvideobench_clip_${clip}_tau_${tau}_trace_softmax"
    # keyframe_path="longvideobench_keyframes/one-clip"
    keyframe_path="longvideobench_keyframes/longvideobench_random_anchor"
    log_dir="${output_dir}/logs"
    mkdir -p "$log_dir"

    echo "=============================="
    echo "Run combo: clip=$clip, tau=$tau"
    echo "Output dir: $output_dir"
    echo "Keyframe path: $keyframe_path"
    echo "=============================="

    # （可选）keyframe_path 不存在就跳过该组合


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
      python3 qwen3-vl/longvideobench_test_8b.py \
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

    python3 qwen3-vl/eval/eval_longvideobench.py \
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
    # done
  done
done

# 等所有子进程结束
wait
echo "All chunks finished. Merging jsonlines..."

# 合并 jsonl 文件（每行一个结果）
cat ${output_dir}/${base_output_name}_chunk*.json > ${output_dir}/${base_output_name}.json

python3 qwen3-vl/eval/eval_longvideobench.py\
  --num-chunks $CHUNKS\
  --output-dir $output_dir \


echo "Merged file: ${output_dir}/${base_output_name}.json"
echo "Done."
