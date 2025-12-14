set -x

MODEL_PATH=/mnt/Shared_05_disk/fsq/hf_home/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3  # replace it with your local file path
SAVE_PATH=/mnt/Shared_05_disk/fsq/gui_agent_exp/gui-r1/mind2web_ws_grpo_qwen2_5_vl_3b_105
CONFIG_PATH=examples/config_mind2web_105.yaml

# Create SAVE_PATH directory if it doesn't exist
mkdir -p "${SAVE_PATH}"

# Copy this script to SAVE_PATH
cp "$0" "${SAVE_PATH}/$(basename "$0")"

# Copy CONFIG_PATH file to SAVE_PATH
cp "${CONFIG_PATH}" "${SAVE_PATH}/$(basename "${CONFIG_PATH}")"

SYSTEM_PROMPT=""""""

python3 -m verl.trainer.main \
    config=${CONFIG_PATH} \
    data.train_files=/mnt/Shared_05_disk/fsq/gui_agent_data/Mind2Web/metadata/hf_train_ws_sim_0.9.json \
    data.val_files=/mnt/Shared_05_disk/fsq/gui_agent_data/Mind2Web/metadata/hf_test_task.json \
    data.system_prompt="${SYSTEM_PROMPT}" \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.enable_chunked_prefill=false \
    worker.reward.compute_score=r1ws \
    trainer.experiment_name=qwen2_5_vl_3b_mind2web_grpo_ws_105 \
    trainer.n_gpus_per_node=8 \
    trainer.save_checkpoint_path=${SAVE_PATH} \
    data.max_pixels=1258291 \
    data.max_prompt_length=2048 \
    data.max_response_length=1024 \
    data.val_batch_size=128 \
    data.img_dir=/mnt/Shared_05_disk/fsq/gui_agent_data/Mind2Web/images \
