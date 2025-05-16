#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
文本语言模型调用示例

这个示例演示如何使用JSONFlow的ModelInvoker调用文本语言模型。
TextLLMProcessor接收包含文本的JSON输入，
并返回添加了模型回复的JSON数据。
"""

import os
import argparse
from typing import Dict, Any, Optional, List
from jsonflow.operators.model import ModelInvoker
from jsonflow.core import Pipeline, JsonOperator
from jsonflow.io import JsonLoader, JsonSaver


class TextProcessor(JsonOperator):
    """文本预处理操作符"""
    
    def __init__(self, 
                 input_field: str = "input_text",
                 output_field: str = "processed_text",
                 add_instruction: bool = True,
                 instruction: str = "请回答以下问题："):
        """
        初始化文本预处理操作符
        
        Args:
            input_field: 输入文本的字段名，默认为"input_text"
            output_field: 输出处理后文本的字段名，默认为"processed_text"
            add_instruction: 是否添加指令，默认为True
            instruction: 添加的指令文本，默认为"请回答以下问题："
        """
        super().__init__(name="TextProcessor", description="Text preprocessing operator")
        self.input_field = input_field
        self.output_field = output_field
        self.add_instruction = add_instruction
        self.instruction = instruction
    
    def process(self, json_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理包含文本的JSON数据
        
        Args:
            json_data: 包含文本的JSON数据
            
        Returns:
            dict: 添加了处理后文本的JSON数据
        """
        if not json_data or self.input_field not in json_data:
            return json_data
            
        result = json_data.copy()
        input_text = result[self.input_field]
        
        # 文本处理逻辑
        processed_text = input_text.strip()
        
        # 添加指令（如果需要）
        if self.add_instruction and processed_text:
            processed_text = f"{self.instruction}\n{processed_text}"
        
        result[self.output_field] = processed_text
        return result


class TextLLMProcessor(ModelInvoker):
    """文本语言模型处理操作符"""
    
    def __init__(self, 
                 model: str,
                 input_field: str = "processed_text",
                 output_field: str = "model_response",
                 system_prompt: Optional[str] = None,
                 **kwargs):
        """
        初始化文本语言模型处理操作符
        
        Args:
            model: 模型名称（如qianfan-llama2-7b）
            input_field: 输入文本的字段名，默认为"processed_text"
            output_field: 输出模型回复的字段名，默认为"model_response"
            system_prompt: 系统提示，默认为None
            **kwargs: 其他传递给ModelInvoker的参数
        """
        super().__init__(model=model, **kwargs)
        self.input_field = input_field
        self.output_field = output_field
        self.system_prompt = system_prompt
    
    def process(self, json_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理包含文本的JSON数据，调用语言模型生成回复
        
        Args:
            json_data: 包含文本的JSON数据
            
        Returns:
            dict: 添加了模型回复的JSON数据
        """
        if not json_data or self.input_field not in json_data:
            return json_data
            
        result = json_data.copy()
        input_text = result[self.input_field]
        
        # 构建消息
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        
        messages.append({"role": "user", "content": input_text})
        
        # 调用语言模型
        try:
            response = self.call_llm(messages)
            result[self.output_field] = response
        except Exception as e:
            result[self.output_field] = f"模型调用错误: {str(e)}"
        
        return result


class ResponseSummarizer(JsonOperator):
    """回复总结操作符"""
    
    def __init__(self, 
                 input_field: str = "model_response",
                 output_field: str = "summary",
                 max_length: int = 100):
        """
        初始化回复总结操作符
        
        Args:
            input_field: 输入模型回复的字段名，默认为"model_response"
            output_field: 输出总结的字段名，默认为"summary"
            max_length: 总结的最大长度，默认为100
        """
        super().__init__(name="ResponseSummarizer", description="Response summarization operator")
        self.input_field = input_field
        self.output_field = output_field
        self.max_length = max_length
    
    def process(self, json_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理包含模型回复的JSON数据，生成总结
        
        Args:
            json_data: 包含模型回复的JSON数据
            
        Returns:
            dict: 添加了总结的JSON数据
        """
        if not json_data or self.input_field not in json_data:
            return json_data
            
        result = json_data.copy()
        response = result[self.input_field]
        
        # 简单的总结逻辑：截取前max_length个字符
        if len(response) > self.max_length:
            summary = response[:self.max_length] + "..."
        else:
            summary = response
        
        result[self.output_field] = summary
        return result


def main():
    """
    运行文本语言模型处理示例。
    
    这个函数:
    1. 创建一个Pipeline包含TextProcessor和TextLLMProcessor
    2. 处理包含文本的JSON数据
    3. 保存处理结果
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="文本语言模型处理示例")
    parser.add_argument("--input", default=None, help="输入文本，如果不指定将使用示例文本")
    parser.add_argument("--output", default="llm_responses.jsonl", help="输出文件路径")
    parser.add_argument("--model", default="qianfan-llama2-7b", help="使用的模型名称")
    parser.add_argument("--base-url", default="https://qianfan.baidubce.com/v2", help="API基础URL")
    parser.add_argument("--api-key", required=True, help="API密钥")
    parser.add_argument("--system-prompt", default="你是一个知识渊博、乐于助人的AI助手。", help="系统提示")
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = os.path.dirname(os.path.abspath(args.output))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 创建示例数据
    input_text = args.input if args.input else "人工智能的发展历程是怎样的？请简要概述其主要阶段和里程碑。"
    
    sample_data = {
        "id": "query001",
        "input_text": input_text,
        "metadata": {"type": "question", "category": "AI"}
    }
    
    # 创建文本处理管道
    pipeline = Pipeline([
        TextProcessor(
            input_field="input_text",
            output_field="processed_text",
            add_instruction=True,
            instruction="请详细回答以下问题："
        ),
        TextLLMProcessor(
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            input_field="processed_text",
            output_field="model_response",
            system_prompt=args.system_prompt,
            max_tokens=800
        ),
        ResponseSummarizer(
            input_field="model_response",
            output_field="summary",
            max_length=100
        )
    ])
    
    # 处理文本
    print("\n=== JSONFlow 文本语言模型处理示例 ===")
    print(f"\n处理文本: {input_text[:50]}..." if len(input_text) > 50 else f"\n处理文本: {input_text}")
    
    try:
        result = pipeline.process(sample_data)
        print(f"✓ 处理完成")
        
        # 显示结果
        print("\n=== 处理结果 ===")
        
        response = result.get('model_response', '处理失败')
        summary = result.get('summary', '无总结')
        
        print(f"总结: {summary}")
        print("\n完整回复:")
        print(response)
        
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