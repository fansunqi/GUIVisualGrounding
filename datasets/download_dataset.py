from huggingface_hub import snapshot_download, HfFileSystem

# 配置参数
dataset_name = "fansunqi/HTML"  # 数据集标识（如 "imdb"、"facebook/bart-base"）
file_path_in_repo = "screenshot_rendered_test_mhtml.zip"  # 目标文件在数据集中的路径（从 Files 页面复制）
save_dir = "../../GUI-agent-data-process/"  # 本地保存目录（自动创建不存在的目录）
token = "xxx"  # 若为私有数据集，传入你的 HF token（从 https://huggingface.co/settings/tokens 获取）

# 方法 1：使用 snapshot_download（简洁，自动处理目录结构）
snapshot_download(
    repo_id=dataset_name,
    repo_type="dataset",  # 明确是数据集（默认也是 dataset，可省略）
    allow_patterns=file_path_in_repo,  # 仅下载匹配的文件（支持通配符，如 "*.csv"）
    ignore_patterns="*",  # 忽略其他所有文件
    local_dir=save_dir,
    local_dir_use_symlinks=False,  # 禁用符号链接，直接复制文件到本地
    token=token
)