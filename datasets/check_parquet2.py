import pandas as pd

# 读取其中一个原文件（如file1.parquet）
file_path = "/root/cache/hub/datasets--ritzzai--GUI-R1/snapshots/ca55ddaa180c5e8f8b27003221c391efa10a1f52/train.parquet"
df = pd.read_parquet(file_path)

# 检查是否存在'image'列
if 'image' in df.columns:
    # 打印'image'列的前5条数据（查看字典结构）
    print("原文件中'image'列的部分数据（字典格式）：")
    for i, val in enumerate(df['image'].head(5)):  # head(5)取前5条
        print(f"第{i+1}条：类型：{type(val['bytes'])}, 内容预览：{str(val)[:100]}")  # 只打印前100字符
else:
    print("原文件中不存在'image'列，请检查列名是否正确")