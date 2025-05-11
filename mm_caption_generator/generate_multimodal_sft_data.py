#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
多模态SFT数据生成工具

该脚本用于基于图片目录生成多模态问答对，将结果保存为JSONL格式的训练数据。
使用Claude模型通过图片生成问答对，格式参考vlm_train_format.jsonl文件。

简单使用示例:
    # 设置API密钥
    export OPENAI_API_KEY="your-api-key"
    
    # 运行脚本处理images目录中的所有图片
    python generate_multimodal_sft_data.py
    
    # 指定参数运行
    python generate_multimodal_sft_data.py --image-dir=images --output=sft_data.jsonl --model=claude-3-7-sonnet-20250219
"""

import os
import json
import base64
import re
from pathlib import Path
from typing import List, Dict, Any
import argparse
import io
from PIL import Image

from jsonflow.core import Pipeline, JsonOperator
from jsonflow.io import JsonLoader, JsonSaver
from jsonflow.operators.json_ops import JsonTransformer, TextNormalizer
from jsonflow.operators.model import MultimodalInvoker

# 支持的图片格式
SUPPORTED_IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']

def get_image_files(image_dir: str) -> List[str]:
    """
    获取目录下所有支持的图片文件
    
    Args:
        image_dir: 图片目录
    
    Returns:
        图片文件路径列表
    """
    image_files = []
    for ext in SUPPORTED_IMAGE_EXTENSIONS:
        image_files.extend(list(Path(image_dir).glob(f'*{ext}')))
    
    # 也包括大写扩展名
    for ext in [e.upper() for e in SUPPORTED_IMAGE_EXTENSIONS]:
        image_files.extend(list(Path(image_dir).glob(f'*{ext}')))
    
    return [str(p) for p in image_files]

class ImageEncoder(JsonOperator):
    """
    图片编码操作符
    
    将图片文件编码为base64字符串，并添加到JSON数据中
    """
    
    def __init__(self, 
                 image_path_field: str = "image_path", 
                 base64_output_field: str = "image_base64",
                 max_width: int = 800,
                 max_height: int = 800,
                 quality: int = 85,
                 name: str = None,
                 description: str = None):
        """
        初始化图片编码操作符
        
        Args:
            image_path_field: 包含图片路径的字段名
            base64_output_field: 存储base64编码结果的字段名
            max_width: 图片最大宽度，超过会被缩放
            max_height: 图片最大高度，超过会被缩放
            quality: JPEG质量压缩参数（1-100）
            name: 操作符名称
            description: 操作符描述
        """
        super().__init__(
            name or "ImageEncoder",
            description or "将图片编码为base64字符串"
        )
        self.image_path_field = image_path_field
        self.base64_output_field = base64_output_field
        self.max_width = max_width
        self.max_height = max_height
        self.quality = quality
    
    def _compress_image(self, image_path: str) -> bytes:
        """
        压缩图片，降低大小
        
        Args:
            image_path: 图片路径
            
        Returns:
            压缩后的图片数据
        """
        try:
            # 打开图片
            img = Image.open(image_path)
            
            # 检查是否需要调整大小
            width, height = img.size
            if width > self.max_width or height > self.max_height:
                # 计算缩放比例
                ratio = min(self.max_width / width, self.max_height / height)
                new_width = int(width * ratio)
                new_height = int(height * ratio)
                
                # 调整大小
                img = img.resize((new_width, new_height), Image.LANCZOS)
                print(f"  调整图片大小: {width}x{height} -> {new_width}x{new_height}")
            
            # 转换为RGB模式(如果是RGBA或其他模式)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # 将图片保存为JPEG到内存缓冲区
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=self.quality)
            
            # 获取压缩后的大小
            compressed_size = buffer.tell()
            buffer.seek(0)
            
            # 获取原始文件大小
            original_size = os.path.getsize(image_path)
            print(f"  压缩图片: {original_size / 1024:.1f}KB -> {compressed_size / 1024:.1f}KB")
            
            return buffer.getvalue()
            
        except Exception as e:
            raise RuntimeError(f"压缩图片失败: {str(e)}")
    
    def process(self, json_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理JSON数据，编码图片
        
        Args:
            json_data: 输入的JSON数据
            
        Returns:
            处理后的JSON数据
        """
        result = json_data.copy()
        
        # 检查输入字段
        if self.image_path_field not in result:
            raise ValueError(f"输入数据中缺少图片路径字段: {self.image_path_field}")
        
        image_path = result[self.image_path_field]
        
        # 编码图片
        try:
            # 压缩并编码图片
            image_data = self._compress_image(image_path)
            base64_data = base64.b64encode(image_data).decode('utf-8')
            result[self.base64_output_field] = base64_data
            
            # 估计token数量（大约每4个字符1个token）
            estimated_tokens = len(base64_data) // 4
            result["estimated_image_tokens"] = estimated_tokens
            print(f"  估计图片token数: 约 {estimated_tokens}")
            
        except Exception as e:
            raise RuntimeError(f"编码图片失败: {str(e)}")
        
        return result

class MessageConstructor(JsonOperator):
    """
    消息构造操作符
    
    构造发送给模型的多模态消息
    """
    
    def __init__(self, 
                 base64_field: str = "image_base64",
                 output_field: str = "message",
                 prompt_template: str = None,
                 name: str = None,
                 description: str = None):
        """
        初始化消息构造操作符
        
        Args:
            base64_field: 包含base64编码图片的字段名
            output_field: 存储构造消息的字段名
            prompt_template: 提示模板
            name: 操作符名称
            description: 操作符描述
        """
        super().__init__(
            name or "MessageConstructor",
            description or "构造发送给模型的多模态消息"
        )
        self.base64_field = base64_field
        self.output_field = output_field
        
        # 默认提示模板
        self.prompt_template = prompt_template or """
        请基于以下图片生成多个高质量的问答对，可用于多模态大模型的训练。
        
        要求：
        1. 问题应是关于图片内容的开放性或具体细节的提问
        2. 答案应该准确、简洁，直接回答问题
        3. 问题应该让模型需要理解图片才能回答
        4. 避免生成需要外部知识才能回答的问题
        
        按照以下JSON格式返回问答对：
        {"question": "图片中的问题...", "answer": "对应的答案..."}
        {"question": "second question", "answer": "second answer"}
        
        只返回JSON内容，不要有其他任何文字。
        """
    
    def process(self, json_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理JSON数据，构造多模态消息
        
        Args:
            json_data: 输入的JSON数据
            
        Returns:
            处理后的JSON数据
        """
        result = json_data.copy()
        
        # 检查输入字段
        if self.base64_field not in result:
            raise ValueError(f"输入数据中缺少base64编码字段: {self.base64_field}")
        
        base64_data = result[self.base64_field]
        
        # 构造多模态消息
        image_message = {
            "role": "user",
            "content": [
                {"type": "text", "text": self.prompt_template},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_data}"}}
            ]
        }
        
        # 存储消息
        result[self.output_field] = json.dumps(image_message)
        
        return result

class ResponseParser(JsonOperator):
    """
    响应解析操作符
    
    解析模型返回的响应，提取JSON数据
    """
    
    def __init__(self, 
                 response_field: str = "response",
                 output_field: str = "qa_data",
                 name: str = None,
                 description: str = None):
        """
        初始化响应解析操作符
        
        Args:
            response_field: 包含模型响应的字段名
            output_field: 存储解析结果的字段名
            name: 操作符名称
            description: 操作符描述
        """
        super().__init__(
            name or "ResponseParser",
            description or "解析模型返回的响应，提取JSON数据"
        )
        self.response_field = response_field
        self.output_field = output_field
    
    def process(self, json_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理JSON数据，解析模型响应
        
        Args:
            json_data: 输入的JSON数据
            
        Returns:
            处理后的JSON数据
        """
        result = json_data.copy()
        result[self.output_field] = []
        
        # 检查输入字段
        if self.response_field not in result:
            raise ValueError(f"输入数据中缺少响应字段: {self.response_field}")
        
        response_text = result[self.response_field]
        
        try:
            # 尝试直接解析为JSON
            for line in response_text.split("\n"):
                qa_data = json.loads(line)
                result[self.output_field].append(qa_data)
            return result
        except json.JSONDecodeError:
            # 如果解析失败，尝试从文本中提取JSON部分
            json_pattern = r'({.*})'
            match = re.search(json_pattern, response_text, re.DOTALL)
            if match:
                try:
                    qa_data = json.loads(match.group(1))
                    result[self.output_field] = qa_data
                    return result
                except json.JSONDecodeError:
                    pass
        
        # 如果所有解析都失败，返回默认值
        result[self.output_field] = [{"question": "图片中显示了什么?", 
                                      "answer": "无法解析模型响应"}]
        return result

class SftFormatter(JsonOperator):
    """
    SFT格式化操作符
    
    将问答数据格式化为SFT训练数据格式
    """
    
    def __init__(self, 
                 qa_field: str = "qa_data",
                 image_path_field: str = "image_path",
                 id_field: str = "id",
                 name: str = None,
                 description: str = None):
        """
        初始化SFT格式化操作符
        
        Args:
            qa_field: 包含问答数据的字段名
            image_path_field: 包含图片路径的字段名
            id_field: 包含ID的字段名
            name: 操作符名称
            description: 操作符描述
        """
        super().__init__(
            name or "SftFormatter",
            description or "将问答数据格式化为SFT训练数据格式"
        )
        self.qa_field = qa_field
        self.image_path_field = image_path_field
        self.id_field = id_field
    
    def process(self, json_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理JSON数据，格式化为SFT训练数据
        
        Args:
            json_data: 输入的JSON数据
            
        Returns:
            处理后的SFT训练数据
        """
        # 检查输入字段
        if self.qa_field not in json_data:
            raise ValueError(f"输入数据中缺少问答数据字段: {self.qa_field}")
        if self.image_path_field not in json_data:
            raise ValueError(f"输入数据中缺少图片路径字段: {self.image_path_field}")
        if self.id_field not in json_data:
            raise ValueError(f"输入数据中缺少ID字段: {self.id_field}")

        sft_example = []
        qa_data = json_data[self.qa_field]
        print(qa_data)
        image_path = json_data[self.image_path_field]
        image_id = json_data[self.id_field]
        
        # 获取图片的相对路径
        relative_path = os.path.basename(image_path)

        for qa in qa_data:
            # 构建对话
            conversations = [
                {
                    "from": "human", 
                    "value": f"<image>\n{qa['question']}"
                },
                {
                    "from": "gpt", 
                    #"value": qa_data['answer']
                    "value": qa['answer']
                }
            ]

            # 构建SFT示例
            sft_example.append({
                "id": image_id,
                "image": f"images/{relative_path}",
                "conversations": conversations
            })
        
        return sft_example

def create_multimodal_pipeline(model_name: str, base_url: str, api_key: str, max_width: int = 800, max_height: int = 800) -> Pipeline:
    """
    创建多模态数据生成pipeline
    
    Args:
        model_name: 模型名称
        base_url: API基础URL
        api_key: API密钥
        max_width: 图片最大宽度
        max_height: 图片最大高度
        
    Returns:
        Pipeline: 处理pipeline
    """
    # 设置OpenAI兼容的参数
    openai_params = {
        "base_url": base_url,
        "api_key": api_key
    }
    
    # 创建处理pipeline
    pipeline = Pipeline([
        # 步骤1: 将图片编码为base64
        ImageEncoder(
            image_path_field="image_path",
            base64_output_field="image_base64",
            max_width=max_width,
            max_height=max_height
        ),
        
        # 步骤2: 构造发送给模型的多模态消息
        MessageConstructor(
            base64_field="image_base64",
            output_field="message"
        ),
        
        # 步骤3: 调用多模态模型 - 使用MultimodalInvoker处理多模态输入
        MultimodalInvoker(
            model=model_name,
            message_field="message",
            response_field="response",
            openai_params=openai_params
        ),
        
        # 步骤4: 解析模型响应
        ResponseParser(
            response_field="response",
            output_field="qa_data"
        ),
        
        # 步骤5: 格式化为SFT训练数据
        SftFormatter(
            qa_field="qa_data",
            image_path_field="image_path",
            id_field="id"
        )
    ])
    
    return pipeline

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="多模态SFT数据生成工具")
    parser.add_argument("--image-dir", type=str, default="images", help="图片目录")
    parser.add_argument("--output", type=str, default="multimodal_sft_data.jsonl", help="输出JSONL文件")
    parser.add_argument("--model", type=str, default="qianfan-llama-vl-8b", help="模型名称")
    parser.add_argument("--api-key", type=str, required=True, help="API密钥")
    parser.add_argument("--base-url", type=str, default="https://qianfan.baidubce.com/v2", help="API基础URL")
    parser.add_argument("--num-samples", type=int, default=-1, help="处理的图片数量，-1表示全部")
    parser.add_argument("--max-width", type=int, default=800, help="图片最大宽度")
    parser.add_argument("--max-height", type=int, default=800, help="图片最大高度")
    parser.add_argument("--quality", type=int, default=85, help="JPEG压缩质量(1-100)")
    
    args = parser.parse_args()
    
    # 验证API密钥
    if not args.api_key:
        print("错误: 未提供API密钥。请通过--api-key参数提供API密钥。")
        return 1
    
    # 获取图片文件
    image_files = get_image_files(args.image_dir)
    if not image_files:
        print(f"错误: 在'{args.image_dir}'目录中未找到支持的图片文件。")
        return 1
    
    print(f"找到{len(image_files)}个图片文件。")
    
    # 限制处理的图片数量
    if args.num_samples > 0:
        image_files = image_files[:args.num_samples]
        print(f"将处理前{len(image_files)}个图片。")
    
    # 创建pipeline
    pipeline = create_multimodal_pipeline(
        model_name=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        max_width=args.max_width,
        max_height=args.max_height
    )
    
    # 初始化JsonSaver
    saver = JsonSaver(args.output)
    
    # 处理每个图片
    for i, image_path in enumerate(image_files):
        print(f"处理图片 {i+1}/{len(image_files)}: {image_path}")
        
        try:
            # 准备输入数据
            input_data = {
                "id": i,
                "image_path": image_path
            }
            
            # 使用pipeline处理数据
            result = pipeline.process(input_data)

            # 保存结果
            for item in result:
                saver.write(item)
            
            if "conversations" in result and len(result["conversations"]) > 0:
                question = result["conversations"][0]["value"].replace("<image>\n", "")
                print(f"  成功生成问答对: {question[:50]}...")
            
        except Exception as e:
            print(f"  处理图片时出错: {e}")
    
    print(f"完成! 生成的数据已保存到 {args.output}")
    return 0

if __name__ == "__main__":
    exit(main()) 
