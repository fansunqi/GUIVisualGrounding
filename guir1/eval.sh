# python eval/eval_mind2web_action.py \
#     --pred_path /home/fsq/gui_agent/GUI-R1-Evol-2/guir1/outputs/ui_tars_1_5_7b/hf_test_task.json \
#     --gt_path 
    # --gt_path /mnt/Shared_05_disk/fsq/gui_agent_data/Mind2Web/metadata/hf_test_task.json \

# python eval/eval_mind2web_point.py --pred_path /home/fsq/gui_agent/GUI-R1-Evol-2/guir1/outputs/ui_tars_1_5_7b_only_point/hf_test_task.json \

OUTPUT_NAME=GUI-R1_3B_org_prompt
OUTPUT_DIR=/apdcephfs_private/qy/projects/fsq/GUI-R1-Evol-2/guir1/outputs
DATA_DIR=/root/cache/hub/datasets--fansunqi--Mind2Web_R1/snapshots/70f9286e9c22b585b28c2fe6e766fd57977df18b
python eval/eval_mind2web_reformat.py \
    --pred_path $OUTPUT_DIR/${OUTPUT_NAME}/hf_test_full.json \
    --gt_path ${DATA_DIR}/metadata/hf_test_full.json