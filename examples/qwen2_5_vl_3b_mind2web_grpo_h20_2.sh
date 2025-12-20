set -x

EXP_NAME=mind2web_train_new_gt_history_r1gui_v2_grpo_qwen2_5_vl_3b_h20
MODEL_PATH=/root/cache/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3  # replace it with your local file path
SAVE_PATH=/root/datasets/fsq/gui_r1_exp/${EXP_NAME}
CONFIG_PATH=examples/config_mind2web_h20.yaml
DATA_DIR=/root/cache/hub/datasets--fansunqi--Mind2Web_R1/snapshots/70f9286e9c22b585b28c2fe6e766fd57977df18b

# Create SAVE_PATH directory if it doesn't exist
mkdir -p "${SAVE_PATH}"

# Copy this script to SAVE_PATH
cp "$0" "${SAVE_PATH}/$(basename "$0")"

# Copy CONFIG_PATH file to SAVE_PATH
cp "${CONFIG_PATH}" "${SAVE_PATH}/$(basename "${CONFIG_PATH}")"

SYSTEM_PROMPT=""""""

export TENSORBOARD_DIR="/apdcephfs_private/qy/projects/fsq/tensorboard_log/${EXP_NAME}"
mkdir -p "${TENSORBOARD_DIR}"

python3 -m verl.trainer.main \
    config=${CONFIG_PATH} \
    data.train_files=${DATA_DIR}/metadata/hf_train_new.json \
    data.val_files=${DATA_DIR}/metadata/hf_test_task.json \
    data.img_dir=${DATA_DIR}/images \
    data.system_prompt="${SYSTEM_PROMPT}" \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.enable_chunked_prefill=false \
    worker.reward.compute_score=r1gui_v2 \
    trainer.experiment_name=${EXP_NAME} \
    trainer.n_gpus_per_node=8 \
    trainer.save_checkpoint_path=${SAVE_PATH} \
    data.max_pixels=1258291 \
    data.max_prompt_length=2048 \
    data.max_response_length=1024 \
    data.val_batch_size=1024 \
    trainer.save_freq=50 \
    data.use_history=true \
    data.history_num=4 \
    trainer.load_checkpoint_path=${SAVE_PATH}/global_step_500

    
