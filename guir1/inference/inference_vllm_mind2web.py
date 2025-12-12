import os
# os.environ["TORCH_COMPILE_CACHE"] = "/data/fsq/vllm_cache"
import re
import ray
import json
import argparse
from tqdm import tqdm
from datasets import load_dataset
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from datasets import Dataset as hf_dataset
from qwen_vl_utils import process_vision_info
from torch.utils.data import Dataset, DataLoader


import math
from PIL import Image
from io import BytesIO
from PIL.Image import Image as ImageObject
from typing import Any, Dict, List, Optional, Union

# 初始化 Ray
ray.init()

# 模型路径
MODEL_PATH = ""

# 推理参数
SAMPLING_PARAMS = SamplingParams(
    temperature=0.0,
    top_p=0.001,
    repetition_penalty=1.05,
    max_tokens=1024,  # 根据需要调整最大生成长度
    stop_token_ids=[],  # 停止标志
)

# 数据路径
DATA_PATH = ""

# 微批大小
MICRO_BATCH = 4


def extract_coord(content):
    # Try to find the bbox within <answer> tags, if can not find, return [0, 0, 0, 0]
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    bbox_pattern = r'\{.*\[(\d+),\s*(\d+)]\s*.*\}'
    content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
    try:
        if content_answer_match:
            content_answer = content_answer_match.group(1).strip()
            coord_match = re.search(bbox_pattern, content_answer)
            if coord_match:
                coord = [int(coord_match.group(1)), int(coord_match.group(2))]
                return coord, True
        else:
            coord_pattern = r'\{.*\((\d+),\s*(\d+))\s*.*\}'
            coord_match = re.search(coord_pattern, content)
            if coord_match:
                coord = [int(coord_match.group(1)), int(coord_match.group(2))]
                return coord, True
        return [0, 0, 0, 0], False
    except:
        return [0, 0, 0, 0], False
    

# NOTE 看一下这里的 resize 对坐标有没有影响
def process_image(image: Union[Dict[str, Any], ImageObject], max_pixels: int, min_pixels: int) -> ImageObject:
    if isinstance(image, dict):
        image = Image.open(BytesIO(image["bytes"]))
    if isinstance(image, str):
        image = Image.open(image)
    if (image.width * image.height) > max_pixels:
        print("Image size (in pixels) exceeds the maximum limit. Resizing the image.")
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if (image.width * image.height) < min_pixels:
        print("Image size (in pixels) is below the minimum limit. Resizing the image.")
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image




class MultiModalDataset(Dataset):
    def __init__(self, data, processor, image_dir):
        self.data = data
        self.processor = processor
        self.processor.max_pixels=2097152
        self.processor.min_pixels=262144
        self.image_dir = image_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """返回单个样本，包含预处理后的数据"""
        sample = self.data[idx]
        
        task = sample['task']
        
        image_url = sample["img_url"]
        image_path = os.path.join(self.image_dir, image_url)
        image=process_image(image_path, self.processor.max_pixels, self.processor.min_pixels)

        # sys_prompt='''A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> nd <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>'''
        text = (
            f"You are GUI-R1, a reasoning GUI Agent Assistant. In this UI screenshot <image>, I want you to continue executing the command '{task}' on the current screenshot.\n"
            "Please provide the action to perform (enumerate from ['click', 'type', 'select']), the point where the cursor is moved to (integer) if a click is performed, and any input text required to complete the action.\n"
            "Output the thinking process in <think> </think> tags, and the final answer in <answer> </answer> tags as follows:\n"
            "<think> ... </think> <answer>[{'action': enum[ 'click', 'type', 'select'], 'point': [x, y], 'input_text': 'no input text'}]</answer>\n"
            "Note:\n specific input text (no default) is necessary for actions enum['type', 'select'] \n Example:\n"
            "[{'action': enum['click'], 'point': [123, 300], 'input_text': 'no input text'}]\n"
            "[{'action': enum['type', 'select'], 'point': [-100, -100], 'input_text': 'shanghai shopping mall'}]\n"
        )
        text = '<image>\n' + text
        message = [
            # {"role":"system", "content": sys_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": text},
                ],
            }
        ]

        # 生成推理所需的 prompt 和多模态输入
        prompt = self.processor.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True,
        )

        # prompt.replace("<|vision_start|><|image_pad|><|vision_end|>","")
        # prompt.replace("<image>","<|vision_start|><|image_pad|><|vision_end|>")

        image_inputs, video_inputs, video_kwargs = process_vision_info(message, return_video_kwargs=True)


        inputs = self.processor(
                    text=[prompt],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
        
        resized_height = inputs['image_grid_thw'][0][1] * self.processor.image_processor.patch_size
        resized_width = inputs['image_grid_thw'][0][2] * self.processor.image_processor.patch_size
              
        origin_height = image_inputs[0].size[1]
        origin_width = image_inputs[0].size[0]
        scale_x = origin_width / resized_width
        scale_y = origin_height / resized_height

        del inputs

        sample["scale"]=[scale_x.item(),scale_y.item()]
        sample["image_size"]=[origin_width,origin_height]

        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs
        if video_inputs is not None:
            mm_data["video"] = video_inputs

        return {
            "prompt": prompt,
            "multi_modal_data": mm_data,
            "mm_processor_kwargs": video_kwargs,
            "original_sample": sample,
        }


def custom_collate_fn(batch):
    collated_batch = {
        "prompts": [],
        "multi_modal_data": [],
        "mm_processor_kwargs": [],
        "original_samples": [],
    }
    for item in batch:
        collated_batch["prompts"].append(item["prompt"])
        collated_batch["multi_modal_data"].append(item["multi_modal_data"])
        collated_batch["mm_processor_kwargs"].append(item["mm_processor_kwargs"])
        collated_batch["original_samples"].append(item["original_sample"])
    return collated_batch



@ray.remote(num_gpus=1)
class Worker:
    def __init__(self, model_path, sampling_params):
        
        # 每个 Ray Worker 拥有独立的缓存目录
        # 获取 Ray 的 worker ID 来区分
        # worker_id = ray.get_runtime_context().get_node_id()  # 唯一ID
        # pid = os.getpid()  # 或者用进程号
        # cache_dir = f"/data/fsq/torchinductor_ray/worker_{worker_id}_{pid}"
        # os.makedirs(cache_dir, exist_ok=True)
        # os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache_dir
        # print(f"[Worker Init] Using TORCHINDUCTOR_CACHE_DIR={cache_dir}")
        
        self.llm = LLM(
            model=model_path,
            limit_mm_per_prompt={"image": 1, "video": 1},
        )
        self.sampling_params = sampling_params

    def process_data(self, dataloader):
        results = []

        for batch in tqdm(dataloader):
            prompts = batch["prompts"]
            multi_modal_data = batch["multi_modal_data"]
            mm_processor_kwargs = batch["mm_processor_kwargs"]
            original_samples = batch["original_samples"]

            llm_inputs = [
                {
                    "prompt": prompt,
                    "multi_modal_data": mm_data,
                    "mm_processor_kwargs": mm_kwargs,
                }
                for prompt, mm_data, mm_kwargs in zip(prompts, multi_modal_data, mm_processor_kwargs)
            ]

            # 执行推理
            outputs = self.llm.generate(llm_inputs, sampling_params=self.sampling_params)

            # 保存结果
            for original_sample, output in zip(original_samples, outputs):
                result_item = {}
                result_item["id"] = original_sample["id"]
                result_item["action_id"] = original_sample["action_uid"]
                
                generated_text = output.outputs[0].text
                result_item["pred"] = generated_text
                
                gt_bbox = original_sample["step"]["bbox"]
                gt_op = original_sample["step"]["operation"]
                result_item["gt_coord"] = gt_bbox
                result_item["gt_op"] = gt_op
                
                pred_coord, _ = extract_coord(generated_text)
                result_item["pred_coord"] = pred_coord

                results.append(result_item)

        return results
    
    

def main(args):
    # 将数据分成 8 份
    MODEL_PATH=args.model_path
    DATA_PATH=args.data_path
    
    # 打开 parquet, json 或者 jsonl 文件
    if DATA_PATH.endswith('parquet'):
        data=load_dataset("parquet", data_files=DATA_PATH, split="train")
    else:
        data = [json.loads(s) for s in open(DATA_PATH, "r")] if DATA_PATH.endswith(".jsonl") else json.load(open(DATA_PATH,"r"))

    # 输出路径
    OUTPUT_DIR = os.path.join(args.output_path, args.output_name)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    NEW_FILE = os.path.join(OUTPUT_DIR, DATA_PATH.split("/")[-1].replace(".jsonl", "_pred.jsonl").replace('.parquet','.json'))
    
    num_actors = args.num_actor
    
    data_chunks = [hf_dataset.from_list(data[i::num_actors]) for i in range(num_actors)]
    
    # 加载处理器
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    
    # 创建 8 个 Actor，每个 Actor 分配到一个 GPU
    workers = [Worker.remote(MODEL_PATH, SAMPLING_PARAMS) for _ in range(num_actors)]
    
    # 使用 PyTorch Dataset 和 DataLoader
    futures = []
    for i, chunk in enumerate(data_chunks):
        dataset = MultiModalDataset(chunk, processor, args.image_dir)
        dataloader = DataLoader(dataset, batch_size=MICRO_BATCH, shuffle=False, num_workers=4, collate_fn=custom_collate_fn)
        futures.append(workers[i].process_data.remote(dataloader))

    # 收集所有结果
    all_results = ray.get(futures)

    # 将结果写入文件
    with open(NEW_FILE, "w") as ans_file:
        for worker_results in all_results:
            for sample in worker_results:
                ans_file.write(json.dumps(sample) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='<model_path>')
    parser.add_argument('--data_path', type=str, default="<data_path>")
    parser.add_argument('--image_dir', type=str, default="/data/fsq/gui_agent_data/Mind2Web/images")
    parser.add_argument('--output_name', type=str, default="<output_name>")
    parser.add_argument('--output_path', type=str, default='./outputs')
    parser.add_argument('--num_actor', type=int, default=8)
    args = parser.parse_args()
    main(args)