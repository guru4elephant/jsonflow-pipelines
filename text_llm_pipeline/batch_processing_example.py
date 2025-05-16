#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
文本语言模型批量处理示例

这个示例演示如何使用JSONFlow的Pipeline和Executor批量处理多个文本输入，
并调用文本语言模型生成回复。
"""

import os
import json
import argparse
from typing import Dict, Any, List
from jsonflow.core import Pipeline, MultiThreadExecutor
from jsonflow.io import JsonLoader, JsonSaver

# 导入自定义操作符
from text_llm_example import TextProcessor, TextLLMProcessor, ResponseSummarizer


def load_sample_data() -> List[Dict[str, Any]]:
    """
    加载示例数据
    
    Returns:
        list: 包含示例问题的数据列表
    """
    return [
        {
            "id": "q1",
            "input_text": "人工智能可能带来哪些社会影响？",
            "metadata": {"type": "question", "category": "AI", "difficulty": "medium"}
        },
        {
            "id": "q2",
            "input_text": "简要介绍深度学习的原理。",
            "metadata": {"type": "question", "category": "AI", "difficulty": "hard"}
        },
        {
            "id": "q3",
            "input_text": "什么是自然语言处理？",
            "metadata": {"type": "question", "category": "NLP", "difficulty": "easy"}
        },
        {
            "id": "q4",
            "input_text": "机器学习和深度学习有什么区别？",
            "metadata": {"type": "question", "category": "ML", "difficulty": "medium"}
        },
        {
            "id": "q5",
            "input_text": "计算机视觉的主要应用领域有哪些？",
            "metadata": {"type": "question", "category": "CV", "difficulty": "medium"}
        }
    ]


def main():
    """
    运行文本语言模型批量处理示例。
    
    这个函数:
    1. 创建一个Pipeline包含TextProcessor和TextLLMProcessor
    2. 使用MultiThreadExecutor并行处理多个输入
    3. 保存处理结果
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="文本语言模型批量处理示例")
    parser.add_argument("--input", default=None, help="输入JSONL文件路径，不指定则使用内置样例")
    parser.add_argument("--output", default="batch_responses.jsonl", help="输出JSONL文件路径")
    parser.add_argument("--model", default="qianfan-llama2-7b", help="使用的模型名称")
    parser.add_argument("--base-url", default="https://qianfan.baidubce.com/v2", help="API基础URL")
    parser.add_argument("--api-key", required=True, help="API密钥")
    parser.add_argument("--system-prompt", default="你是一个知识渊博、乐于助人的AI助手。", help="系统提示")
    parser.add_argument("--threads", type=int, default=2, help="并行处理的线程数")
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = os.path.dirname(os.path.abspath(args.output))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 加载输入数据
    input_data = []
    if args.input and os.path.exists(args.input):
        # 从文件加载
        loader = JsonLoader(args.input)
        input_data = loader.load()
        print(f"已从 {args.input} 加载 {len(input_data)} 条数据")
    else:
        # 使用示例数据
        input_data = load_sample_data()
        print(f"使用 {len(input_data)} 条内置示例数据")
    
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
    
    # 创建多线程执行器
    executor = MultiThreadExecutor(pipeline, max_workers=args.threads)
    
    # 批量处理
    print(f"\n=== JSONFlow 文本语言模型批量处理示例 (线程数: {args.threads}) ===")
    print(f"开始处理 {len(input_data)} 条输入...")
    
    try:
        results = executor.execute_all(input_data)
        print(f"✓ 批量处理完成，共处理 {len(results)} 条数据")
        
        # 显示结果摘要
        print("\n=== 处理结果摘要 ===")
        for i, result in enumerate(results[:3], 1):  # 只显示前3条
            print(f"{i}. 问题: {result.get('input_text', '')[:30]}...")
            print(f"   摘要: {result.get('summary', '无摘要')[:50]}...\n")
        
        if len(results) > 3:
            print(f"... 共 {len(results)} 条结果")
        
        # 保存结果
        saver = JsonSaver(args.output)
        saver.write_all(results)
        print(f"\n结果已保存到 {args.output}")
        
    except Exception as e:
        print(f"✗ 处理失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 