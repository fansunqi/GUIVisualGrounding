from huggingface_hub import hf_hub_download

dataset_name = "fansunqi/HTML"  # 数据集标识（如 "imdb"、"facebook/bart-base"）
file_path_in_repo = "interact_results_test_wo_href.zip"  # 目标文件在数据集中的路径（从 Files 页面复制）
save_dir = "../../GUI-agent-data-process/"  # 本地保存目录（自动创建不存在的目录）
token = "xxx"  # 若为私有数据集，传入你的 HF token（从 https://huggingface.co/settings/tokens 获取）

file_path = hf_hub_download(
    repo_id=dataset_name,
    repo_type="dataset",  # 明确是数据集
    filename=file_path_in_repo,
    local_dir=save_dir,  # 本地目录（自动创建）
    local_dir_use_symlinks=False,  # Windows 建议设为 False，避免软链接问题
    token=token
)
print("文件保存路径：", file_path)  # 输出：./bert_config/config.json