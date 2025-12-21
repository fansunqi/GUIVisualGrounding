import os
import pdb
import glob

exp_dir="/root/datasets/fsq/gui_r1_exp"

# exp_name_list = [
#     "mind2web_gt_history_fix_norm_grpo_qwen2_5_vl_3b_h20_try2",
#     "mind2web_train_new_gt_history_r1gui_org_grpo_qwen2_5_vl_3b_h20",
#     "mind2web_train_new_gt_history_r1gui_v2_grpo_qwen2_5_vl_3b_h20",
#     "mind2web_ws_org_sim_0_9_grpo_qwen2_5_vl_3b_h20",
#     "mind2web_ws_sim_0_7_grpo_qwen2_5_vl_3b_h20",
#     "mind2web_ws_sim_0_9_grpo_qwen2_5_vl_3b_h20",
#     "mind2web_ws_v2_sim_0_9_grpo_qwen2_5_vl_3b_h20"
# ]

# exp_name_list = [
#     "gui_r1gui_org_grpo_qwen2_5_vl_3b_h20",
#     "mind2web_meta_task_ws_org_sim_0_9_grpo_qwen2_5_vl_3b_h20",
#     "mind2web_phase3_from1_train_new_gt_history_r1gui_org_grpo_qwen2_5_vl_3b_h20"
# ]

exp_name_list = [
    "gui_phase3_from_mind2web_phase1_r1gui_org_grpo_qwen2_5_vl_3b_h20"
]


def find_global_step_folders_recursive(root_path):
    global_step_folders = []
    
    for dirpath, dirnames, filenames in os.walk(root_path):
        for dirname in dirnames:
            if dirname.startswith("global_step"):
                full_path = os.path.join(dirpath, dirname)
                global_step_folders.append(full_path)
    
    return global_step_folders


if __name__ == "__main__":
    
    for exp_name in exp_name_list:
        exp_path=os.path.join(exp_dir, exp_name)
        print(f"Processing experiment: {exp_name}")
        
        # 遍历 ckpt_dir 下的 global_step_* 文件夹
        global_step_folders = find_global_step_folders_recursive(exp_path)
        print(global_step_folders)
        
        for dir in global_step_folders:
            
            actor_dir = os.path.join(dir, "actor")
            hf_ckpt_path = os.path.join(actor_dir, "huggingface/model-00002-of-00002.safetensors")
            pt_files = glob.glob(os.path.join(actor_dir, "*.pt"))
            
            if os.path.exists(hf_ckpt_path):
                
                dataloader_pt_path = os.path.join(dir, "dataloader.pt")
                print(f"delete dataloader pt: {dataloader_pt_path}")
                os.remove(dataloader_pt_path)
                
                if pt_files:
                    print(f"在 {actor_dir} 中找到 {len(pt_files)} 个.pt文件，准备删除...")
                    # 逐个删除文件
                    for pt_file in pt_files:
                        try:
                            os.remove(pt_file)
                            print(f"已删除: {pt_file}")
                        except Exception as e:
                            print(f"删除 {pt_file} 失败: {e}")
   
        
    