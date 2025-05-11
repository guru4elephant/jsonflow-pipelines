# 图像标注 (Image Caption) Pipeline

## 简介

图像标注 Pipeline 是一个基于 JSONFlow 的数据处理流水线，专为图像描述自动化设计。该流水线借助多模态大语言模型，能够接收图像输入，并生成关于图像内容的详细、准确的文本描述。

该 Pipeline 特别适用于：
- 自动图像描述和内容标注
- 视觉内容数据库的内容索引
- 无障碍应用的图像描述生成
- 图像内容分析和识别

## 功能特点

- **扩展性强**：基于 JSONFlow 的 ModelInvoker 扩展，便于自定义和扩展
- **易于配置**：支持灵活配置模型参数、提示词和输出形式
- **多模态支持**：支持各种多模态大语言模型，如千帆的 VL 模型、GPT-4V 等
- **批量处理**：支持单张或批量图像处理
- **结构化输出**：输出 JSON 格式，便于后续处理和分析

## 安装要求

- Python 3.7+
- JSONFlow 0.1.0+
- 相关依赖包

```bash
pip install guru4elephant-jsonflow>=0.1.0
```

## 使用方法

### 命令行使用

```bash
# 使用默认参数
python image_caption_example.py --image /path/to/your/image.jpg --api-key YOUR_API_KEY

# 自定义参数
python image_caption_example.py \
  --image /path/to/your/image.jpg \
  --output results.jsonl \
  --model model-name \
  --base-url https://api-url \
  --api-key YOUR_API_KEY
```

### 作为模块使用

```python
from jsonflow.core import Pipeline
from image_caption.image_caption_example import ImageCaptioningInvoker

# 创建图像标注操作符
captioner = ImageCaptioningInvoker(
    model="您的模型名称",  # 使用支持图像的模型
    base_url="API基础URL", # 模型API基础URL
    api_key="您的API密钥",  # 替换为您的API密钥
    image_field="image_path",  # 输入JSON中的图像路径字段
    caption_field="caption",  # 输出JSON中的标注字段
    caption_prompt="请详细描述这张图片，包括主要对象、场景和可能的含义。",
    system_prompt="你是一个专业的图像描述专家，善于捕捉图像的细节和核心内容。",
    max_tokens=300
)

# 创建处理流水线
pipeline = Pipeline([captioner])

# 处理单个图像
result = pipeline.process({
    "id": "img001",
    "image_path": "/path/to/your/image.jpg",
    "metadata": {"type": "landscape"}
})

print(result["caption"])  # 打印生成的图像描述
```

### 完整示例

参考 `image_caption_example.py` 文件中的 `main()` 函数，展示了从加载图像到保存结果的完整流程。

## 示例输入与输出

### 输入示例

```json
{
  "id": "img001",
  "image_path": "/path/to/your/image.jpg",
  "metadata": {
    "type": "landscape",
    "source": "user_upload"
  }
}
```

### 输出示例

```json
{
  "id": "img001",
  "image_path": "/path/to/your/image.jpg",
  "metadata": {
    "type": "landscape",
    "source": "user_upload"
  },
  "caption": "这张图片展示了一片宁静的湖泊景观。湖面平静如镜，倒映着周围葱郁的山峦和蓝天。远处的山脉层层叠叠，与天空形成优美的轮廓线。前景中可以看到几块岩石和一小片沙滩，增添了画面的层次感。整体色调以蓝色和绿色为主，给人一种宁静祥和的感受。这是一幅典型的自然风景照，展现了大自然的壮丽和宁静。"
}
```

## 自定义与扩展

可以通过继承 `ImageCaptioningInvoker` 类来自定义更多功能，例如：

- 添加图像预处理步骤
- 支持更多图像格式
- 自定义响应格式
- 集成特定领域的图像分析

## 使用提示

1. 确保提供有效的图像路径和正确的模型API密钥
2. 针对不同类型的图像，可以调整提示词以获得更精确的描述
3. 对于大量图像的批处理，建议使用多线程执行器以提高效率
4. 可以通过环境变量设置API密钥，避免在代码中硬编码：
   ```bash
   export OPENAI_API_KEY=your_api_key
   ```

## 许可证

本项目遵循 MIT 许可证。 