#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
图像标注示例

这个示例演示如何扩展ModelInvoker创建一个专门用于图像标注的操作符。
ImageCaptioningInvoker接收包含图像路径的JSON输入，
并返回添加了图像标注的JSON数据。
"""

import os
import base64
from typing import Dict, Any, Optional
from jsonflow.operators.model import ModelInvoker
from jsonflow.core import Pipeline
from jsonflow.io import JsonLoader, JsonSaver
import argparse


class ImageCaptioningInvoker(ModelInvoker):
    """图像标注操作符"""
    
    def __init__(self, 
                 model: str,
                 image_field: str = "image_path",
                 caption_field: str = "caption",
                 caption_prompt: str = "请简要描述这张图片的内容。",
                 **kwargs):
        """
        初始化图像标注操作符
        
        Args:
            model: 模型名称（需要支持图像处理，如gpt-4-vision-preview）
            image_field: 输入图像路径的字段名，默认为"image_path"
            caption_field: 输出标注的字段名，默认为"caption"
            caption_prompt: 向模型发送的提示文本，默认为简单的描述请求
            **kwargs: 其他传递给ModelInvoker的参数
        """
        super().__init__(model=model, **kwargs)
        self.image_field = image_field
        self.caption_field = caption_field
        self.caption_prompt = caption_prompt
    
    def process(self, json_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理包含图像路径的JSON数据，生成图像标注
        
        Args:
            json_data: 包含图像路径的JSON数据
            
        Returns:
            dict: 添加了图像标注的JSON数据
        """
        if not json_data or self.image_field not in json_data:
            return json_data
            
        result = json_data.copy()
        image_path = result[self.image_field]
        
        # 读取并编码图像
        try:
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            result[self.caption_field] = f"图像读取错误: {str(e)}"
            return result
        
        # 构建包含图像的消息
        messages = [
            {"role": "system", "content": self.system_prompt or "你是一个图像标注专家，善于准确描述图像内容。"},
            {
                "role": "user", 
                "content": [
                    {"type": "text", "text": self.caption_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
                ]
            }
        ]
        
        # 调用支持图像的模型
        try:
            print(self.base_url)
            response = self.call_llm(messages)
            result[self.caption_field] = response
        except Exception as e:
            result[self.caption_field] = f"模型调用错误: {str(e)}"
        
        return result


def main():
    """
    运行图像标注示例。
    
    这个函数:
    1. 创建一个ImageCaptioningInvoker实例
    2. 处理一组包含图像路径的JSON数据
    3. 保存标注结果
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="图像标注示例")
    parser.add_argument("--image", required=True, help="输入图像的路径")
    parser.add_argument("--output", default="image_captions.jsonl", help="输出文件路径")
    parser.add_argument("--model", default="qianfan-llama-vl-8b", help="使用的模型名称")
    parser.add_argument("--base-url", default="https://qianfan.baidubce.com/v2", help="API基础URL")
    parser.add_argument("--api-key", required=True, help="API密钥")
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = os.path.dirname(os.path.abspath(args.output))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 检查图像文件
    if not os.path.exists(args.image):
        print(f"错误: 图像文件不存在: {args.image}")
        return
    
    # 创建示例数据
    sample_data = {
        "id": "img001",
        "image_path": args.image,
        "metadata": {"type": "image"}
    }
    
    # 创建图像标注管道
    pipeline = Pipeline([
        ImageCaptioningInvoker(
            model=args.model,  # 使用命令行参数指定的模型
            base_url=args.base_url,  # 使用命令行参数指定的基础URL
            api_key=args.api_key,  # 使用命令行参数指定的API密钥
            image_field="image_path",
            caption_field="caption",
            caption_prompt="请详细描述这张图片，包括主要对象、场景和可能的含义。",
            system_prompt="你是一个专业的图像描述专家，善于捕捉图像的细节和核心内容。",
            max_tokens=300
        )
    ])
    
    # 处理图像
    print("\n=== JSONFlow 图像标注示例 ===")
    print(f"\n处理图像: {args.image}...")
    
    try:
        result = pipeline.process(sample_data)
        print(f"✓ 标注完成")
        
        # 显示结果
        print("\n=== 标注结果 ===")
        print(f"图像: {result['image_path']}")
        
        caption = result.get('caption', '处理失败')
        if len(caption) > 100:
            print(f"标注: {caption[:100]}...")
        else:
            print(f"标注: {caption}")
        
        # 保存结果
        saver = JsonSaver(args.output)
        saver.write(result)
        print(f"\n结果已保存到 {args.output}")
        
    except Exception as e:
        print(f"✗ 处理失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 