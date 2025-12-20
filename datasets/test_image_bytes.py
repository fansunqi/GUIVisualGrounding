from PIL import Image
from io import BytesIO
import pdb

# 假设已有一个PIL图像对象（例如从文件打开的）
img = Image.open("image.jpg")

# 创建一个BytesIO缓冲区
buffer = BytesIO()

# 将图像写入缓冲区（需指定格式，如"PNG"、"JPEG"）
img.save(buffer, format="JPEG")  # 格式需与原图像一致或兼容

# 从缓冲区获取字节数据
image_bytes = buffer.getvalue()  # 得到bytes类型数据
bytes_list = list(image_bytes)
pdb.set_trace()

# 验证：用PIL重新读取字节数据
image = Image.open(BytesIO(image_bytes))
image.save("reconstructed_image.jpg")  # 保存为新文件以验证正确性