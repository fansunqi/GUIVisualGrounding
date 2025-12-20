import pandas as pd
import json

file1_path = "/root/cache/hub/datasets--fansunqi--Mind2Web_R1/snapshots/70f9286e9c22b585b28c2fe6e766fd57977df18b/metadata/hf_train_new.parquet"
file2_path = "/root/cache/hub/datasets--ritzzai--GUI-R1/snapshots/ca55ddaa180c5e8f8b27003221c391efa10a1f52/train.parquet"
save_path = "/root/cache/hub/datasets--fansunqi--Mind2Web_R1/snapshots/70f9286e9c22b585b28c2fe6e766fd57977df18b/metadata/hf_train_new_gui_r1_train_merge.parquet"

import dask.dataframe as dd

ddf1 = dd.read_parquet(file1_path)
ddf2 = dd.read_parquet(file2_path)

count1 = ddf1.shape[0].compute()  # Dask需要compute()获取实际值
count2 = ddf2.shape[0].compute()
print(f"file1.parquet 数据条数：{count1}")
print(f"file2.parquet 数据条数：{count2}")

merged_ddf = dd.concat([ddf1, ddf2])
merged_count = merged_ddf.shape[0].compute()
print(f"合并后的数据条数：{merged_count}")

# 关键：Dask中shuffle（打乱顺序）
shuffled_ddf = merged_ddf.sample(frac=1, random_state=42)  # frac=1表示保留所有数据

# 保存结果
shuffled_ddf.to_parquet(save_path)
