set -x

MODEL_PATH=/data/fsq/hf_home/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3  # replace it with your local file path

SYSTEM_PROMPT=""""""

python3 -m verl.trainer.main \
    config=examples/config_mind2web.yaml \
    data.train_files=/data/fsq/gui_agent_data/Mind2Web/metadata/hf_train.json \
    data.val_files=/data/fsq/gui_agent_data/Mind2Web/metadata/hf_test_task.json \
    data.system_prompt="${SYSTEM_PROMPT}" \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.enable_chunked_prefill=false \
    worker.reward.compute_score=r1gui \
    trainer.experiment_name=qwen2_5_vl_3b_mind2web_grpo \
    trainer.n_gpus_per_node=8 \
    trainer.save_checkpoint_path=/data/fsq/GUI-R1_exp/mind2web_gt_ckpt_episodes_30_try2 \
    data.max_pixels=1258291 \
    data.max_prompt_length=2048 \
    data.max_response_length=1024 \
    data.val_batch_size=16