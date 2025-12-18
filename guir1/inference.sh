# python inference/inference_vllm_mind2web.py \
#     --model_path /data/fsq/hf_home/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3 \
#     --data_path /data/fsq/gui_agent_data/Mind2Web/metadata/hf_test_task.json \
#     --output_name test \
#     --num_actor 1 \
#     --use_history \
#     --history_num 4 \

# export TORCH_COMPILE_CACHE=/data/fsq/vllm_cache
# python inference/inference_vllm_mind2web.py \
#     --model_path /data/fsq/GUI-R1_exp/mind2web_ws_grpo_qwen2_5_vl_3b/global_step_250/actor/huggingface \
#     --data_path /data/fsq/gui_agent_data/Mind2Web/metadata/hf_test_task.json \
#     --output_name mind2web_ws_grpo_qwen2_5_vl_3b_global_step_250 \
#     --num_actor 2

# accelerate launch inference/inference_vllm_mind2web_accelerate.py \
#     --model_path /data/fsq/GUI-R1_exp/mind2web_ws_grpo_qwen2_5_vl_3b/global_step_50/actor/huggingface \
#     --data_path /data/fsq/gui_agent_data/Mind2Web/metadata/hf_test_domain.json \
#     --output_name mind2web_ws_grpo_qwen2_5_vl_3b_global_step_50

# 105 ui-tars-2b-sft
# python inference/inference_vllm_mind2web.py \
#     --model_path /mnt/Shared_05_disk/fsq/hf_home/hub/models--ByteDance-Seed--UI-TARS-2B-SFT/snapshots/f366a1db3e7f29635f5b236d6a71dea367a0a700 \
#     --data_path /mnt/Shared_05_disk/fsq/gui_agent_data/Mind2Web/metadata/hf_test_task.json \
#     --image_dir /mnt/Shared_05_disk/fsq/gui_agent_data/Mind2Web/images \
#     --output_name ui_tars_2b_sft \
#     --num_actor 2

# 103 ui-tars-7b
# python inference/inference_vllm_mind2web.py \
#     --model_path /mnt/Shared_06_disk1/fsq/hf_home/hub/models--ByteDance-Seed--UI-TARS-1.5-7B/snapshots/683d002dd99d8f95104d31e70391a39348857f4e \
#     --data_path /mnt/Shared_06_disk1/fsq/data/Mind2Web/metadata/hf_test_task.json \
#     --image_dir /mnt/Shared_06_disk1/fsq/data/Mind2Web/images \
#     --output_name ui_tars_1_5_7b \
#     --num_actor 6

# 103 ui-tars-7b only point
# python inference/inference_vllm_mind2web_only_point.py \
#     --model_path /mnt/Shared_06_disk1/fsq/hf_home/hub/models--ByteDance-Seed--UI-TARS-1.5-7B/snapshots/683d002dd99d8f95104d31e70391a39348857f4e \
#     --data_path /mnt/Shared_06_disk1/fsq/data/Mind2Web/metadata/hf_test_task.json \
#     --image_dir /mnt/Shared_06_disk1/fsq/data/Mind2Web/images \
#     --output_name ui_tars_1_5_7b_only_point \
#     --num_actor 6

# python inference/inference_vllm_screenspot_only_point.py \
#     --model_path /mnt/Shared_06_disk1/fsq/hf_home/hub/models--ByteDance-Seed--UI-TARS-1.5-7B/snapshots/683d002dd99d8f95104d31e70391a39348857f4e \
#     --data_path /mnt/Shared_06_disk1/fsq/hf_home/hub/datasets--ritzzai--GUI-R1/snapshots/ca55ddaa180c5e8f8b27003221c391efa10a1f52/screenspot_pro_test.parquet \
#     --num_actor 7

# h20 qwen2.5-vl-3b
export ray_init_num_cpus=32
DATA_DIR=/root/cache/hub/datasets--fansunqi--Mind2Web_R1/snapshots/762e2f2708c887222a07179bb847affd3e23e6f5
LOCAL_HF_DIR=/root/cache/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3
python inference/inference_vllm_mind2web.py \
    --model_path $LOCAL_HF_DIR \
    --data_path ${DATA_DIR}/metadata/hf_train.json \
    --image_dir ${DATA_DIR}/images \
    --output_name qwen_2_5_vl_3b_h20 \
    --num_actor 2
