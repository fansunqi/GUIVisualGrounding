from huggingface_hub import HfApi
import logging

api = HfApi()
api.upload_file(
    path_or_fileobj="/data/fsq/gui_agent_data/Mind2Web/metadata/hf_test_full.json",  # 本地数据路径
    path_in_repo="metadata/hf_test_full.json",
    repo_id="fansunqi/Mind2Web_R1",  # 数据集仓库名
    repo_type="dataset",
    commit_message="fix hf_test_full.json"
)

# api.upload_file(
#     path_or_fileobj="/data/fsq/gui_agent_data/Mind2Web/images.tar.gz",  # 本地数据路径
#     path_in_repo="images.tar.gz",
#     repo_id="fansunqi/Mind2Web_R1",  # 数据集仓库名
#     repo_type="dataset",
#     commit_message="Upload dataset directly"
# )
