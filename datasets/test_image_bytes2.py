
from PIL import Image
from io import BytesIO
import pdb

img_path = "image.jpg"

with open(img_path, "rb") as f:
    image_bytes = f.read()  # image_bytes 类型为 <class 'bytes'>

pdb.set_trace()
    
image = Image.open(BytesIO(image_bytes))
image.save("reconstructed_image2.jpg")  # 保存为新文件以验证正确性