import torch
from PIL import Image
import requests
import os
import base64
from io import BytesIO
import json
import numpy as np

from transformers import AutoProcessor, Gemma3ForConditionalGeneration

def load_image_from_url(url):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
        return image
    except Exception as e:
        print(f"从URL加载图片失败: {e}")
        return None

def decode_base64_image(base64_string):
    try:
        if "," in base64_string:
            _, encoded_data = base64_string.split(",", 1)
        else:
            encoded_data = base64_string
        decoded_image_data = base64.b64decode(encoded_data)
        image = Image.open(BytesIO(decoded_image_data)).convert("RGB")
        return image
    except Exception as e:
        print(f"解码 Base64 图像失败: {e}")
        return None

def comfy_image_to_pil(comfy_image_tensor):
    if comfy_image_tensor is None:
        return None
    try:
        image_np = comfy_image_tensor.squeeze(0).cpu().numpy() * 255
        return Image.fromarray(image_np.astype('uint8')).convert("RGB")
    except Exception as e:
        print(f"转换 ComfyUI 图像张量到 PIL 图像失败: {e}")
        return None

class Gemma3MultiModalChatNode:
    @classmethod
    def IS_CHANGED(s, model_path, model_id="google/gemma-3-4b-it", *args, **kwargs):
        if os.path.exists(os.path.join(model_path, "config.json")):
            return False
        return True

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text_input": ("STRING", {"multiline": True, "default": ""}),
                "model_path": ("STRING", {"default": "models/gemma3"}), # 模型本地路径
                "model_id": ("STRING", {"default": "google/gemma-3-4b-it"}), # Hugging Face 模型ID
                "max_new_tokens": ("INT", {"default": 100, "min": 1, "max": 8192}), # Gemma 3 输出上下文最大为8192
                "do_sample": ("BOOLEAN", {"default": False}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 2.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 1.0, "step": 0.01}),
                "repetition_penalty": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01}),
                "num_beams": ("INT", {"default": 1, "min": 1, "max": 10}), # 1表示贪婪搜索或抽样
                "load_from_hf_if_not_local": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "content_part_1": ("STRING", {"forceInput": True, "optional": True}),
                "content_part_2": ("STRING", {"forceInput": True, "optional": True}),
                "content_part_3": ("STRING", {"forceInput": True, "optional": True}),

                "system_prompt": ("STRING", {"multiline": True, "default": "You are a helpful assistant."}),
                "chat_history_json": ("STRING", {"multiline": True, "default": "[]"}), # 聊天历史的JSON字符串
                "add_generation_prompt": ("BOOLEAN", {"default": True}), # 控制是否添加模型特定的生成提示
                "skip_special_tokens": ("BOOLEAN", {"default": True}), # 控制解码时是否跳过特殊token
                "pad_token_id": ("INT", {"default": None, "placeholder": "Leave empty for default", "optional": True}), # pad_token_id
                "eos_token_id": ("INT", {"default": None, "placeholder": "Leave empty for default", "optional": True}), # eos_token_id
            }
        }

    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("generated_text", "updated_chat_history_json",)
    FUNCTION = "execute"
    CATEGORY = "Gemma3/Multi-Modal Chat"

    def __init__(self):
        self.model = None
        self.processor = None
        self.current_model_path = None
        self.current_model_id = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Determine device

    def load_model(self, model_path, model_id, load_from_hf_if_not_local):
        if self.model and self.processor and \
           self.current_model_path == model_path and \
           self.current_model_id == model_id:
            return

        print(f"尝试加载 Gemma 3 模型. 本地路径: {model_path}, Hugging Face ID: {model_id}")

        # 优先从本地路径加载
        if os.path.exists(model_path) and os.path.isdir(model_path):
            try:
                self.model = Gemma3ForConditionalGeneration.from_pretrained(
                    model_path, torch_dtype=torch.bfloat16
                ).to(self.device).eval() # Load to determined device, no device_map
                self.processor = AutoProcessor.from_pretrained(model_path)
                print(f"成功从本地路径加载模型: {model_path}")
                self.current_model_path = model_path
                self.current_model_id = model_id
                return
            except Exception as e:
                print(f"从本地路径 {model_path} 加载模型失败: {e}")
                if not load_from_hf_if_not_local:
                    raise FileNotFoundError(f"未在指定本地路径 {model_path} 找到模型，且不允许从Hugging Face下载。")
        
        # 如果本地加载失败或不允许本地加载，尝试从Hugging Face下载
        if load_from_hf_if_not_local:
            print(f"尝试从 Hugging Face 下载模型: {model_id} 到 {model_path}")
            os.makedirs(model_path, exist_ok=True) 
            try:
                self.model = Gemma3ForConditionalGeneration.from_pretrained(
                    model_id, cache_dir=model_path, torch_dtype=torch.bfloat16
                ).to(self.device).eval() # Load to determined device, no device_map
                self.processor = AutoProcessor.from_pretrained(model_id, cache_dir=model_path)
                print(f"成功从 Hugging Face 下载并加载模型: {model_id}, 缓存到 {model_path}")
                self.current_model_path = model_path
                self.current_model_id = model_id
            except Exception as e:
                print(f"从 Hugging Face 下载/加载模型失败: {e}")
                raise
        else:
            raise FileNotFoundError(f"未在指定路径 {model_path} 找到 Gemma 3 模型，且不允许从 Hugging Face 下载。")

    def process_generic_content_part(self, content_string):
        content_items = []
        if not content_string:
            return content_items

        try:
            parsed_content = json.loads(content_string)
            if isinstance(parsed_content, dict) and "type" in parsed_content:
                if parsed_content["type"] == "image" and "image" in parsed_content and isinstance(parsed_content["image"], str) and parsed_content["image"].startswith("data:image/"):
                    content_items.append(parsed_content)
                    print("Processed content part as JSON-encoded image.")
                elif parsed_content["type"] == "text" and "text" in parsed_content and isinstance(parsed_content["text"], str):
                    content_items.append(parsed_content)
                    print("Processed content part as JSON-encoded text.")
                else:
                    content_items.append({"type": "text", "text": content_string})
                    print("Processed unknown JSON content part as plain text.")
            elif isinstance(parsed_content, list):
                for item in parsed_content:
                    if isinstance(item, dict) and "type" in item:
                        if item["type"] == "image" and "image" in item and isinstance(item["image"], str) and item["image"].startswith("data:image/"):
                            content_items.append(item)
                        elif item["type"] == "text" and "text" in item and isinstance(item["text"], str):
                            content_items.append(item)
                        else:
                            print(f"Warning: Unrecognized item in JSON list: {item}")
                print("Processed content part as JSON list.")
            else:
                content_items.append({"type": "text", "text": content_string})
                print("Processed content part as plain text (unrecognized JSON structure).")
        except json.JSONDecodeError:
            content_items.append({"type": "text", "text": content_string})
            print("Processed content part as plain text.")
        return content_items

    def execute(self, text_input, model_path, model_id, max_new_tokens, do_sample, temperature, top_p,
                repetition_penalty, num_beams, load_from_hf_if_not_local,
                content_part_1="", content_part_2="", content_part_3="", 
                system_prompt="You are a helpful assistant.", chat_history_json="[]",
                add_generation_prompt=True, skip_special_tokens=True,
                pad_token_id=None, eos_token_id=None):

        self.load_model(model_path, model_id, load_from_hf_if_not_local)

        try:
            chat_history = json.loads(chat_history_json)
            if not isinstance(chat_history, list):
                print("警告: chat_history_json 不是有效的列表，将重置为 []。")
                chat_history = []
        except json.JSONDecodeError:
            print("警告: chat_history_json 解析失败，将重置为 []。")
            chat_history = []

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}]
            }
        ]
        messages.extend(chat_history)
        user_content = []

        if text_input:
            user_content.append({"type": "text", "text": text_input})

        user_content.extend(self.process_generic_content_part(content_part_1))
        user_content.extend(self.process_generic_content_part(content_part_2))
        user_content.extend(self.process_generic_content_part(content_part_3))

        if not user_content:
            print("当前轮次未提供有效内容")
            return ("无内容可处理。", json.dumps(chat_history, ensure_ascii=False, indent=2))

        messages.append({"role": "user", "content": user_content})

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.model.device, dtype=torch.bfloat16)

        input_len = inputs["input_ids"].shape[-1]

        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "repetition_penalty": repetition_penalty,
            "num_beams": num_beams,
        }

        if do_sample:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = top_p
        
        if pad_token_id is not None:
            generation_kwargs["pad_token_id"] = pad_token_id
        if eos_token_id is not None:
            generation_kwargs["eos_token_id"] = eos_token_id
        
        with torch.inference_mode():
            generation = self.model.generate(**inputs, **generation_kwargs)
            generation = generation[0][input_len:]

        decoded_text = self.processor.decode(generation, skip_special_tokens=skip_special_tokens)
        
        serializable_chat_history = []
        for msg in messages[1:]:
            if msg["role"] == "user":
                serializable_content = []
                for item in msg["content"]:
                    if item["type"] == "text":
                        serializable_content.append(item)
                    elif item["type"] == "image" and isinstance(item["image"], str) and item["image"].startswith("data:image/"):
                        serializable_content.append(item)
                if serializable_content:
                    serializable_chat_history.append({"role": "user", "content": serializable_content})
            elif msg["role"] == "model":
                serializable_chat_history.append(msg)

        serializable_chat_history.append({"role": "model", "content": [{"type": "text", "text": decoded_text}]})

        updated_chat_history_json = json.dumps(serializable_chat_history, ensure_ascii=False, indent=2)

        return (decoded_text, updated_chat_history_json,)

class ImageEncoderContentPartNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_input": ("IMAGE", ),
                "encoding_format": (["JPEG", "PNG", "WEBP"], {"default": "JPEG"}),
                "quality": ("INT", {"default": 90, "min": 0, "max": 100, "step": 1}),
            },
            "optional": {
                "image_url_input": ("STRING", {"default": "", "multiline": False, "placeholder": "或输入图片URL", "optional": True}),
                "image_base64_input": ("STRING", {"multiline": True, "default": "", "placeholder": "或输入Base64编码的图片", "optional": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("content_part_json",)
    FUNCTION = "encode_image"
    CATEGORY = "Gemma3/Utils"

    def encode_image(self, image_input, encoding_format, quality, image_url_input="", image_base64_input=""):
        pil_image = None

        if image_input is not None:
            pil_image = comfy_image_to_pil(image_input)
            if pil_image:
                print("已处理 ComfyUI 图像输入。")
        elif image_url_input:
            pil_image = load_image_from_url(image_url_input)
            if pil_image:
                print(f"已从URL '{image_url_input}' 加载图像。")
        elif image_base64_input:
            pil_image = decode_base64_image(image_base64_input)
            if pil_image:
                print("已成功解码 Base64 图像。")

        if pil_image is None:
            return ("无法处理图像输入，请检查输入。",)

        buffered = BytesIO()
        try:
            if encoding_format == "JPEG":
                pil_image.save(buffered, format="JPEG", quality=quality)
                mime_type = "image/jpeg"
            elif encoding_format == "PNG":
                pil_image.save(buffered, format="PNG")
                mime_type = "image/png"
            elif encoding_format == "WEBP":
                pil_image.save(buffered, format="WEBP", quality=quality)
                mime_type = "image/webp"
            else:
                raise ValueError("不支持的编码格式。")

            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

            content_part_data = {
                "type": "image",
                "image": f"data:{mime_type};base64,{img_base64}"
            }

            json_output = json.dumps(content_part_data, ensure_ascii=False)
            return (json_output,)

        except Exception as e:
            print(f"图像编码或格式化失败: {e}")
            return (f"图像编码或格式化失败: {e}",)


NODE_CLASS_MAPPINGS = {
    "Gemma3MultiModalChatNode": Gemma3MultiModalChatNode,
    "ImageEncoderContentPartNode": ImageEncoderContentPartNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Gemma3MultiModalChatNode": "Gemma 3",
    "ImageEncoderContentPartNode": "图像编码器 (Gemma3)"
}
