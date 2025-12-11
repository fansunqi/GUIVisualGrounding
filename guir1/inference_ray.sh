#!/bin/bash
# ================================
# vLLM + Ray 启动脚本 (独立缓存)
# ================================

# 设置总的 worker 数量
NUM_WORKERS=8

# 主目录（可以改成 /data/tmp 之类的更大磁盘路径）
CACHE_BASE="/tmp/torchinductor_ray"

# 清理旧缓存（可选）
rm -rf ${CACHE_BASE}
mkdir -p ${CACHE_BASE}

# 循环启动多个 worker
for ((RANK=0; RANK<NUM_WORKERS; RANK++)); do
  export TORCHINDUCTOR_CACHE_DIR=${CACHE_BASE}/worker_${RANK}

  echo ">>> 启动 worker ${RANK}, 缓存目录: ${TORCHINDUCTOR_CACHE_DIR}"

  # 这里替换成你自己的启动命令，比如:
  # python inference_vllm_mind2web_accelerate.py --rank ${RANK} --num-workers ${NUM_WORKERS} &

  python inference/inference_vllm_mind2web.py \
    --model_path /data/fsq/GUI-R1_exp/mind2web_ws_grpo_qwen2_5_vl_3b/global_step_50/actor/huggingface \
    --data_path /data/fsq/gui_agent_data/Mind2Web/metadata/hf_test_domain.json \
    --output_name mind2web_ws_grpo_qwen2_5_vl_3b_global_step_50 \
    --rank ${RANK} \
    --num-workers ${NUM_WORKERS}

  ray start --head --port=$((6379+RANK)) \
    --object-store-memory=10GB \
    --dashboard-host=0.0.0.0 &

  # 确保每个 worker 有自己的缓存目录
  mkdir -p ${TORCHINDUCTOR_CACHE_DIR}
done

wait
