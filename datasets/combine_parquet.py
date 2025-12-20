import pandas as pd
import json

file1_path = "/root/cache/hub/datasets--fansunqi--Mind2Web_R1/snapshots/70f9286e9c22b585b28c2fe6e766fd57977df18b/metadata/hf_train_new.parquet"
file2_path = "/root/cache/hub/datasets--ritzzai--GUI-R1/snapshots/ca55ddaa180c5e8f8b27003221c391efa10a1f52/train.parquet"
save_path = "/root/cache/hub/datasets--fansunqi--Mind2Web_R1/snapshots/70f9286e9c22b585b28c2fe6e766fd57977df18b/metadata/hf_train_new_gui_r1_train_merge.parquet"

# 1. 读取两个Parquet文件
df1 = pd.read_parquet(file1_path)
df2 = pd.read_parquet(file2_path)

# 2. 统计并输出两个文件的行数
count1 = len(df1)
count2 = len(df2)
print(f"file1.parquet 数据条数：{count1}")
print(f"file2.parquet 数据条数：{count2}")

# 3. 合并数据
merged_df = pd.concat([df1, df2], ignore_index=True)


# 关键：将字典列（如'image'）转换为JSON字符串
# 关键：处理image列的多种类型（dict、bytes等）
# if 'image' in merged_df.columns:
#     def process_image(x):
#         if isinstance(x, dict):
#             # 字典类型：序列化为JSON字符串
#             return json.dumps(x)
#         elif isinstance(x, bytes):
#             # bytes类型：转换为Base64字符串（可JSON序列化）
#             # return base64.b64encode(x).decode('utf-8')
#             return x
#         else:
#             # 其他类型（如字符串、None等）：直接返回
#             return x
    
#     # 应用处理函数
#     merged_df['image'] = merged_df['image'].apply(process_image)
    
    
# 4. 统计并输出合并后的行数
merged_count = len(merged_df)
print(f"合并后的数据条数：{merged_count}")  # 理论上等于 count1 + count2

# 5. Shuffle 打乱顺序
shuffled_df = merged_df.sample(frac=1, random_state=42)

# 6. 保存结果
shuffled_df.to_parquet(save_path, index=False)