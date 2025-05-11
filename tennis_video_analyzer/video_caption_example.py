#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
视频标注示例

这个示例演示如何扩展ModelInvoker创建一个专门用于视频帧标注的操作符。
VideoCaptioningInvoker接收包含视频路径的JSON输入，提取关键帧，
并返回添加了每一帧标注的JSON数据。
"""

import os
import cv2
import base64
import tempfile
import argparse
import numpy as np
from PIL import Image
from typing import Dict, Any, Optional, List
from jsonflow.operators.model import ModelInvoker
from jsonflow.core import Pipeline
from jsonflow.io import JsonLoader, JsonSaver


class VideoCaptioningInvoker(ModelInvoker):
    """视频帧标注操作符"""
    
    def __init__(self, 
                 model: str,
                 video_field: str = "video_path",
                 frames_field: str = "frames",
                 captions_field: str = "captions",
                 caption_prompt: str = "请简要描述这个视频帧的内容。",
                 num_frames: int = 5,
                 **kwargs):
        """
        初始化视频帧标注操作符
        
        Args:
            model: 模型名称（需要支持图像处理，如qianfan-llama-vl-8b）
            video_field: 输入视频路径的字段名，默认为"video_path"
            frames_field: 输出帧信息的字段名，默认为"frames"
            captions_field: 输出标注的字段名，默认为"captions"
            caption_prompt: 向模型发送的提示文本，默认为简单的描述请求
            num_frames: 要从视频中提取的帧数量
            **kwargs: 其他传递给ModelInvoker的参数
        """
        super().__init__(model=model, **kwargs)
        self.video_field = video_field
        self.frames_field = frames_field
        self.captions_field = captions_field
        self.caption_prompt = caption_prompt
        self.num_frames = num_frames
    
    def process(self, json_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理包含视频路径的JSON数据，提取帧并生成图像标注
        
        Args:
            json_data: 包含视频路径的JSON数据
            
        Returns:
            dict: 添加了帧和标注的JSON数据
        """
        if not json_data or self.video_field not in json_data:
            return json_data
            
        result = json_data.copy()
        video_path = result[self.video_field]
        
        # 提取视频帧
        try:
            frames = self._extract_frames(video_path, self.num_frames)
            result[self.frames_field] = frames
        except Exception as e:
            result[self.captions_field] = [f"视频帧提取错误: {str(e)}"]
            return result
        
        # 为每个帧生成标注
        content = []
        for frame in frames:
            # 读取并编码图像
            try:
                with open(frame, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                    if base64_image:
                        image_content = {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }

                        content.append(image_content)
            except Exception as e:
                print(e)
                continue
        content.append({"type": "text", "text": self.caption_prompt})
        # 构建包含图像的消息
        messages = [{"role": "system", "content": "使用中文生成回复"}, {"role": "user", "content": content}]
        # 调用支持图像的模型
        try:
            response = self.call_llm(messages)
            result[self.captions_field] = response
        except Exception as e:
            result[self.captions_field] = f"模型调用错误: {str(e)}"

        return result
    
    def _extract_frames(self, video_path, num_frames: int) -> List[Dict[str, Any]]:
        """从视频中提取帧，使用OpenCV"""
        print(f"正在从视频中提取帧: {video_path}")
        
        
        # 使用OpenCV打开视频
        video = cv2.VideoCapture(video_path)
        
        # 获取视频的总帧数
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            print("无法读取视频帧")
            video.release()
            return []
        
        print(f"视频总帧数: {total_frames}")
        
        # 创建临时目录存储图像
        temp_dir = tempfile.mkdtemp()
        temp_image_paths = []
        
        # 均匀抽取帧
        if total_frames < num_frames:
            # 如果视频帧数少于指定帧数，则全部使用
            indices = list(range(total_frames))
        else:
            # 均匀抽取指定数量的帧
            indices = np.linspace(0, total_frames-1, num=num_frames, dtype=int)
        
        # 提取并保存帧
        for i, frame_idx in enumerate(indices):
            # 设置读取位置
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            success, frame = video.read()
            
            if success:
                # OpenCV以BGR格式读取，需要转换为RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                
                # 保存为JPEG文件
                temp_image_path = os.path.join(temp_dir, f"frame_{i+1}.jpg")
                img.save(temp_image_path, "JPEG", quality=90)
                temp_image_paths.append(temp_image_path)
            else:
                print(f"帧 {frame_idx} 读取失败")
        
        # 释放视频资源
        video.release()
        
        print(f"已提取 {len(temp_image_paths)} 帧")
        return temp_image_paths


def main():
    """
    运行视频帧标注示例。
    
    这个函数:
    1. 创建一个VideoCaptioningInvoker实例
    2. 处理一个视频路径输入
    3. 保存标注结果
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="视频帧标注示例")
    parser.add_argument("--video", required=True, help="输入视频的路径")
    parser.add_argument("--frames", type=int, default=4, help="要提取的帧数量")
    parser.add_argument("--output", default="video_captions.jsonl", help="输出文件路径")
    parser.add_argument("--model", default="qianfan-llama-vl-8b", help="使用的模型名称")
    parser.add_argument("--base-url", default="https://qianfan.baidubce.com/v2", help="API基础URL")
    parser.add_argument("--api-key", required=True, help="API密钥")
    parser.add_argument("--system-prompt", default="你是一个专业的视频分析专家，善于捕捉视频帧中的细节和动作要点。", 
                        help="系统提示词")
    parser.add_argument("--caption-prompt", 
                        default="请详细描述这个视频的内容，综合给你的多张图片的信息给出一个完整的视频描述，你不需要逐帧解释。", 
                        help="标注提示词")
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) if os.path.dirname(args.output) else ".", exist_ok=True)
    
    # 检查视频文件
    if not os.path.exists(args.video):
        print(f"错误: 视频文件不存在: {args.video}")
        return
    
    # 创建示例数据
    sample_data = {
        "id": "video001",
        "video_path": args.video,
        "metadata": {"type": "tennis_match"}
    }
    
    # 创建视频标注管道
    pipeline = Pipeline([
        VideoCaptioningInvoker(
            model=args.model,  # 使用命令行参数指定的模型
            base_url=args.base_url,  # 使用命令行参数指定的基础URL
            api_key=args.api_key,  # 使用命令行参数指定的API密钥
            video_field="video_path",
            frames_field="frames",
            captions_field="caption",
            caption_prompt=args.caption_prompt,
            system_prompt=args.system_prompt,
            max_tokens=300,
            num_frames=args.frames
        )
    ])
    
    # 处理视频
    print("\n=== JSONFlow 视频帧标注示例 ===")
    print(f"\n处理视频: {args.video}...")
    
    try:
        result = pipeline.process(sample_data)
        print(f"✓ 标注完成")
        
        # 显示结果
        print("\n=== 标注结果 ===")
        print(f"视频: {result['video_path']}")
        
        if 'frames' in result and 'caption' in result:
            frames = result['frames']
            caption = result['caption']
            
            print(caption)

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