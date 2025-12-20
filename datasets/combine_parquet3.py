from datasets import load_dataset, concatenate_datasets
import random

# 1. 读取两个 Parquet 文件（替换为你的文件路径）
file1_path = "/root/cache/hub/datasets--fansunqi--Mind2Web_R1/snapshots/70f9286e9c22b585b28c2fe6e766fd57977df18b/metadata/hf_train_new_image_bytes.parquet"
file2_path = "/root/cache/hub/datasets--ritzzai--GUI-R1/snapshots/ca55ddaa180c5e8f8b27003221c391efa10a1f52/train.parquet"
output_path = "/root/cache/hub/datasets--fansunqi--Mind2Web_R1/snapshots/70f9286e9c22b585b28c2fe6e766fd57977df18b/metadata/hf_train_new_gui_r1_train_merge.parquet"

# 使用 load_dataset 读取，格式指定为 "parquet"
dataset1 = load_dataset("parquet", data_files=file1_path, split="train")
dataset2 = load_dataset("parquet", data_files=file2_path, split="train")

# 2. 合并两个数据集（要求两者的特征结构完全一致）
merged_dataset = concatenate_datasets([dataset1, dataset2])

# 3. 打乱数据集顺序（设置随机种子确保可复现）
shuffled_dataset = merged_dataset.shuffle(seed=42)  # seed 可选，用于固定随机结果

# 4. 存储为新的 Parquet 文件（替换为你的输出路径）
shuffled_dataset.to_parquet(output_path)

print(f"合并并打乱后的数据集已存储至：{output_path}")
print(f"总样本数：{len(shuffled_dataset)}")