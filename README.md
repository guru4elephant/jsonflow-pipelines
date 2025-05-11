# JSONFlow Pipelines

基于JSONFlow构建的一系列可用的数据处理流水线。这个库的主要目的是基于jsonflow这个基础库开发各种LLM的pipeline，用于沉淀数据处理、数据合成和大模型交互的技术方案，提供可重用、可扩展的解决方案。

## 项目背景

JSONFlow是一个专为JSON数据处理而设计的高效库，支持构建灵活的数据处理流水线。本项目通过扩展JSONFlow的功能，针对大语言模型和多模态模型的应用场景，提供了一系列开箱即用的数据处理和模型交互方案。

## 可用的流水线

### 1. [图像标注器 (image_caption)](./image_caption)

**创建日期**: 2025年5月11日

**主要功能**:
- 基于多模态大语言模型的图像内容自动描述
- 支持配置不同的模型API（如千帆VL模型、GPT-4V等）
- 批量处理图像并生成结构化输出

**使用示例**:
```bash
python image_caption/image_caption_example.py --image /path/to/your/image.jpg --api-key YOUR_API_KEY
```

### 2. [多模态内容生成器 (mm_caption_generator)](./mm_caption_generator)

**创建日期**: 2025年5月11日

**主要功能**:
- 基于图片生成多模态问答对
- 自动压缩和编码图像以符合模型要求
- 输出JSONL格式的训练数据，可用于模型微调

**使用示例**:
```bash
python mm_caption_generator/generate_multimodal_sft_data.py --image-dir=images --api-key=YOUR_API_KEY
```

### 3. [网球视频分析器 (tennis_video_analyzer)](./tennis_video_analyzer)

**创建日期**: 2025年5月11日

**主要功能**:
- 视频关键帧提取
- 基于视觉语言模型的球员动作分析
- 技术评估、姿势分析、策略分析
- 综合评估和改进建议

**使用示例**:
```bash
python tennis_video_analyzer/video_caption_example.py --video=videos/match.mp4 --api-key=YOUR_API_KEY
```

### 4. [LLM验证器构建器 (llm_verifier_builder)](./llm_verifier_builder)

**创建日期**: 2025年5月11日

**主要功能**:
- 基于LLM的文本验证和校验
- 支持自定义验证规则和验证流程
- 错误检测和修正建议

**使用示例**: 请参考目录下的README.md文件

## 使用方法

每个pipeline都是独立的，可以单独使用。每个pipeline目录下都包含:
- README.md: 详细的使用说明和示例
- 示例代码文件
- 可选的requirements.txt: 列出特定依赖

## 依赖

- JSONFlow库：`pip install guru4elephant-jsonflow>=0.1.0`
- Python 3.7+
- 其他依赖请查看每个pipeline目录下的requirements.txt文件

## 特点

- **可扩展性**：每个组件都设计为可独立使用或与其他组件组合
- **标准化**：遵循统一的接口和数据格式约定
- **易用性**：详细的文档和示例代码
- **模块化**：可以根据需要只使用部分组件

## 贡献

欢迎贡献新的pipeline或对现有pipeline进行改进。请确保每个新pipeline都包含:
1. 详细的README.md文件，说明pipeline的功能、依赖和使用方法
2. 示例代码和使用说明
3. 必要的测试

## 许可证

本项目采用MIT许可证。
