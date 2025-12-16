from huggingface_hub import HfApi
import logging

api = HfApi()
api.upload_file(
    path_or_fileobj="/mnt/Shared_05_disk/fsq/gui_agent_data/Mind2Web/metadata/hf_train_ws_sim_0.9.json",  # 本地数据路径
    path_in_repo="metadata/hf_train_ws_sim_0.9.json",
    repo_id="fansunqi/mind2web_adapted",  # 数据集仓库名
    repo_type="dataset",
    commit_message="Upload dataset directly"
)

