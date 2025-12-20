import pandas as pd

file1_path = "/root/cache/hub/datasets--fansunqi--Mind2Web_R1/snapshots/70f9286e9c22b585b28c2fe6e766fd57977df18b/metadata/hf_train_new.parquet"
file2_path = "/root/cache/hub/datasets--ritzzai--GUI-R1/snapshots/ca55ddaa180c5e8f8b27003221c391efa10a1f52/train.parquet"

# 读取文件（只读取gt_bbox列和索引，减少内存占用）
df = pd.read_parquet(file2_path, columns=['gt_bbox'])

# 记录字符串类型的行索引和值
str_records = []
for idx, val in df['gt_bbox'].items():
    if isinstance(val, str):
        str_records.append((idx, val))  # (行索引, 值)

# 输出结果
if str_records:
    print(f"共发现{len(str_records)}条字符串类型的gt_bbox数据：")
    for idx, val in str_records:
        print(f"行索引 {idx}：值={val}，类型={type(val)}")
else:
    print("gt_bbox列中没有字符串类型的数据")