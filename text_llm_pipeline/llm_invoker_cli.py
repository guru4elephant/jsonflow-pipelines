#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
文本语言模型调用CLI工具

这个工具可以基于YAML配置文件调用文本语言模型。
通过配置文件和命令行参数，可以灵活地调用不同的模型、设置不同的提示，并获取推理结果。
"""

import os
import sys
import yaml
import argparse
from typing import Dict, Any, List, Optional
from jsonflow.operators.model import ModelInvoker
from jsonflow.core import Pipeline
from jsonflow.io import JsonSaver


class ConfigurableModelInvoker(ModelInvoker):
    """可配置的模型调用操作符"""
    
    def __init__(self, 
                 model: str,
                 prompt_template: str,
                 system_prompt: Optional[str] = None,
                 **kwargs):
        """
        初始化可配置的模型调用操作符
        
        Args:
            model: 模型名称
            prompt_template: 提示模板，可以包含 {input} 占位符
            system_prompt: 系统提示，默认为None
            **kwargs: 其他传递给ModelInvoker的参数
        """
        super().__init__(model=model, **kwargs)
        self.prompt_template = prompt_template
        self.system_prompt = system_prompt
    
    def call_with_text(self, input_text: str) -> str:
        """
        使用文本调用模型
        
        Args:
            input_text: 输入文本
            
        Returns:
            str: 模型回复
        """
        # 填充提示模板
        prompt = self.prompt_template
        if "{input}" in prompt:
            prompt = prompt.replace("{input}", input_text)
        else:
            prompt = f"{prompt}\n{input_text}"
        
        # 构建消息
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        # 调用语言模型
        try:
            response = self.call_llm(messages)
            return response
        except Exception as e:
            return f"模型调用错误: {str(e)}"


def load_config(config_path: str) -> Dict[str, Any]:
    """
    加载YAML配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        dict: 配置信息
    """
    if not os.path.exists(config_path):
        print(f"错误: 配置文件不存在: {config_path}")
        sys.exit(1)
        
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"错误: 加载配置文件失败: {str(e)}")
        sys.exit(1)


def validate_config(config: Dict[str, Any]) -> bool:
    """
    验证配置文件是否包含必要的字段
    
    Args:
        config: 配置信息
        
    Returns:
        bool: 配置是否有效
    """
    required_fields = ['model', 'prompt']
    for field in required_fields:
        if field not in config:
            print(f"错误: 配置文件缺少必要字段: {field}")
            return False
    return True


def main():
    """
    CLI工具主函数
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="文本语言模型调用CLI工具")
    parser.add_argument("--config", "-c", required=True, help="YAML配置文件路径")
    parser.add_argument("--input", "-i", help="输入文本或文本文件路径")
    parser.add_argument("--output", "-o", help="输出文件路径")
    parser.add_argument("--api-key", "-k", help="API密钥（可覆盖配置文件中的设置）")
    parser.add_argument("--verbose", "-v", action="store_true", help="显示详细信息")
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    if not validate_config(config):
        sys.exit(1)
    
    # 设置模型参数
    model_params = {
        "model": config["model"],
        "prompt_template": config["prompt"],
        "system_prompt": config.get("system_prompt"),
        "max_tokens": config.get("max_tokens", 800),
    }
    
    # 设置API相关参数
    if "base_url" in config:
        model_params["base_url"] = config["base_url"]
    
    # 命令行API密钥覆盖配置文件
    api_key = args.api_key or config.get("api_key")
    if not api_key:
        print("错误: 未提供API密钥，请在配置文件中设置api_key或使用--api-key参数")
        sys.exit(1)
    
    model_params["api_key"] = api_key
    
    # 创建模型调用器
    invoker = ConfigurableModelInvoker(**model_params)
    
    # 获取输入文本
    input_text = ""
    if args.input:
        # 如果是文件路径，则读取文件内容
        if os.path.exists(args.input):
            try:
                with open(args.input, 'r', encoding='utf-8') as f:
                    input_text = f.read()
            except Exception as e:
                print(f"错误: 读取输入文件失败: {str(e)}")
                sys.exit(1)
        else:
            # 直接使用命令行提供的文本
            input_text = args.input
    else:
        # 从标准输入读取
        print("请输入文本 (输入完成后按Ctrl+D结束):")
        input_text = sys.stdin.read()
    
    if not input_text.strip():
        print("错误: 没有提供输入文本")
        sys.exit(1)
    
    # 执行模型调用
    if args.verbose:
        print(f"\n模型: {model_params['model']}")
        print(f"系统提示: {model_params.get('system_prompt', '无')}")
        print(f"输入文本: {input_text[:100]}..." if len(input_text) > 100 else f"输入文本: {input_text}")
        print("\n处理中...\n")
        
    result = invoker.call_with_text(input_text)
    
    # 输出结果
    if args.output:
        # 保存到文件
        try:
            output_dir = os.path.dirname(os.path.abspath(args.output))
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(result)
            print(f"结果已保存到: {args.output}")
        except Exception as e:
            print(f"错误: 保存输出文件失败: {str(e)}")
            print("\n结果:")
            print(result)
    else:
        # 打印到标准输出
        print("\n===== 模型回复 =====\n")
        print(result)
        print("\n====================")
    
    # 保存完整的调用记录
    if args.verbose and args.output:
        record_path = f"{os.path.splitext(args.output)[0]}_record.json"
        record = {
            "config": config,
            "input": input_text,
            "output": result
        }
        try:
            saver = JsonSaver(record_path)
            saver.write(record)
            print(f"完整记录已保存到: {record_path}")
        except Exception as e:
            if args.verbose:
                print(f"警告: 保存记录失败: {str(e)}")


if __name__ == "__main__":
    main() 