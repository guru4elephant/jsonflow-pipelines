# JSONFlow Pipelines

基于JSONFlow构建的一系列可用的数据处理流水线。

## 可用的流水线

### 1. 多模态内容生成器 (mm_caption_generator)

基于图片生成多模态问答对，将结果保存为JSONL格式的训练数据。

### 2. 网球视频分析器 (tennis_video_analyzer)

使用千帆视觉语言模型(Qianfan-VL)分析网球视频，提供技术动作、姿势和策略的多维度分析。

- **功能**：视频关键帧提取、球员动作分析、技术评估、策略分析、综合评估和改进建议
- **使用方法**：`cd tennis_video_analyzer && python tennis_video_analysis.py --video-path=videos/match.mp4`
- **详情**：查看 [tennis_video_analyzer/README.md](tennis_video_analyzer/README.md)

## 使用方法

每个pipeline都是独立的，可以单独使用。请参考各个pipeline目录下的README.md文件获取详细使用说明。

## 依赖

- JSONFlow库：`pip install guru4elephant-jsonflow`
- 其他依赖请查看每个pipeline目录下的requirements.txt文件

## 贡献

欢迎贡献新的pipeline或对现有pipeline进行改进。请确保每个新pipeline都包含详细的README.md文件，说明pipeline的功能、依赖和使用方法。
