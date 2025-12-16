from huggingface_hub import HfApi
import logging

api = HfApi()
api.upload_file(
    path_or_fileobj="/home/fsq/data/Mind2Web.tar.gz",  # 本地数据路径
    path_in_repo="Mind2Web.tar.gz",
    repo_id="fansunqi/mind2web_adapted",  # 数据集仓库名
    repo_type="dataset",
    commit_message="Upload dataset directly"
)

