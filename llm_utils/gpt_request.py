import os
import requests
import json
import base64
import cv2
import numpy as np
from mimetypes import guess_type

# 配置 - 使用环境变量或默认值
API_BASE = os.environ.get('GPT4_API_BASE', 'https://api.gptplus5.com/v1')
API_KEY = os.environ.get('GPT4_API_KEY', 'sk-YHRe2zkbnEdU8lBXYbLkplYP5PjhIhSESOmlMWIdvnLjRCA8')
GPT4_MODEL = os.environ.get('GPT4_API_DEPLOY', 'gpt-4o')
GPT4V_MODEL = os.environ.get('GPT4V_API_DEPLOY', 'gpt-4o')

def local_image_to_data_url(image):
    """将本地图片转换为base64 data URL"""
    if isinstance(image, str):
        # 文件路径
        mime_type, _ = guess_type(image)
        if not mime_type:
            mime_type = 'image/jpeg'
        with open(image, "rb") as image_file:
            base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:{mime_type};base64,{base64_encoded_data}"
    elif isinstance(image, np.ndarray):
        # OpenCV图像数组
        base64_encoded_data = base64.b64encode(cv2.imencode('.jpg', image)[1]).decode('utf-8')
        return f"data:image/jpeg;base64,{base64_encoded_data}"
    else:
        raise ValueError("Unsupported image type")

def gptv_response(text_prompt, image_prompt, system_prompt=""):
    """调用视觉模型"""
    url = f"{API_BASE}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    
    # 构建消息
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": text_prompt},
            {
                "type": "image_url",
                "image_url": {
                    "url": local_image_to_data_url(image_prompt)
                }
            }
        ]
    })
    
    data = {
        "model": GPT4V_MODEL,
        "messages": messages,
        "max_tokens": 1000
    }
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        print(f"API请求错误: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"响应内容: {e.response.text}")
        return ""

def gpt_response(text_prompt, system_prompt=""):
    """调用文本模型"""
    url = f"{API_BASE}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    
    # 构建消息
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    messages.append({
        "role": "user",
        "content": text_prompt
    })
    
    data = {
        "model": GPT4_MODEL,
        "messages": messages,
        "max_tokens": 1000
    }
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        print(f"API请求错误: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"响应内容: {e.response.text}")
        return ""

# 测试函数
if __name__ == "__main__":
    # 测试文本API
    print("测试文本API...")
    response = gpt_response("Hello, how are you?")
    print(f"响应: {response}")
    
    # 测试环境变量
    print(f"\n当前配置:")
    print(f"API_BASE: {API_BASE}")
    print(f"API_KEY: {API_KEY[:10]}...")
    print(f"GPT4_MODEL: {GPT4_MODEL}")
    print(f"GPT4V_MODEL: {GPT4V_MODEL}")