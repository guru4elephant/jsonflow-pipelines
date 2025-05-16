# 文本语言模型处理流水线 (Text LLM Pipeline)

这个流水线提供了一个简单而灵活的方式来调用文本语言模型，对输入文本进行处理，并获取模型回复。流水线基于JSONFlow构建，支持灵活配置和扩展。

## 功能特点

- **文本预处理**：支持对输入文本进行标准化和指令添加
- **灵活的模型调用**：支持配置不同的文本语言模型（如千帆模型、OpenAI模型等）
- **结果后处理**：包含回复总结功能，便于快速获取回复摘要
- **可扩展架构**：基于JSONFlow的组件式设计，便于自定义和扩展
- **配置驱动**：支持通过YAML配置文件进行灵活配置和部署

## 核心组件

1. **TextProcessor**：文本预处理操作符，负责对输入文本进行清理和格式化
2. **TextLLMProcessor**：语言模型调用操作符，基于JSONFlow的ModelInvoker扩展，处理与模型的交互
3. **ResponseSummarizer**：回复处理操作符，对模型返回的内容进行摘要和格式化
4. **ConfigurableModelInvoker**：可通过配置文件驱动的模型调用组件，用于CLI工具

## 使用方法

### 安装依赖

```bash
pip install guru4elephant-jsonflow>=0.1.0 requests>=2.28.0 tqdm>=4.64.0 pyyaml>=6.0
```

### 基本用法

单条文本处理：
```bash
python text_llm_example.py --api-key YOUR_API_KEY
```

批量处理：
```bash
python batch_processing_example.py --api-key YOUR_API_KEY
```

基于配置文件的CLI工具：
```bash
python llm_invoker_cli.py --config config_samples/qianfan_default.yaml --input "人工智能的定义是什么？" --api-key YOUR_API_KEY
```

### 命令行工具参数说明

#### 单条文本处理 (text_llm_example.py)

- `--input`：输入文本，不指定则使用默认示例文本
- `--output`：输出JSONL文件路径，默认为"llm_responses.jsonl"
- `--model`：使用的模型名称，默认为"qianfan-llama2-7b"
- `--base-url`：API基础URL，默认为千帆API URL
- `--api-key`：API密钥（必需）
- `--system-prompt`：系统提示，默认为通用助手角色设定

#### 批量处理 (batch_processing_example.py)

- `--input`：输入JSONL文件路径，不指定则使用内置样例
- `--output`：输出JSONL文件路径，默认为"batch_responses.jsonl"
- `--model`：使用的模型名称，默认为"qianfan-llama2-7b"
- `--base-url`：API基础URL，默认为千帆API URL
- `--api-key`：API密钥（必需）
- `--system-prompt`：系统提示，默认为通用助手角色设定
- `--threads`：并行处理的线程数，默认为2

#### 配置文件驱动的CLI工具 (llm_invoker_cli.py)

- `--config`, `-c`：YAML配置文件路径（必需）
- `--input`, `-i`：输入文本或文本文件路径
- `--output`, `-o`：输出文件路径
- `--api-key`, `-k`：API密钥（可覆盖配置文件中的设置）
- `--verbose`, `-v`：显示详细信息

### 配置文件格式

```yaml
# 模型配置
model: "模型名称"
base_url: "API基础URL"
api_key: "API密钥" # 可选，也可通过命令行提供

# 提示配置
prompt: |
  提示模板，可以包含 {input} 占位符，
  用于替换实际输入文本
  
  {input}

# 系统提示
system_prompt: "系统提示内容"

# 生成参数
max_tokens: 500
temperature: 0.7
top_p: 0.9
```

### 示例

1. **使用默认参数处理示例文本**：
   ```bash
   python text_llm_example.py --api-key YOUR_API_KEY
   ```

2. **处理自定义文本**：
   ```bash
   python text_llm_example.py --input "量子计算的基本原理是什么？" --api-key YOUR_API_KEY
   ```

3. **批量处理多个问题**：
   ```bash
   python batch_processing_example.py --threads 4 --api-key YOUR_API_KEY
   ```

4. **使用摘要生成器配置**：
   ```bash
   python llm_invoker_cli.py -c config_samples/text_summarizer.yaml -i article.txt -o summary.txt -k YOUR_API_KEY
   ```

5. **使用代码助手配置**：
   ```bash
   python llm_invoker_cli.py -c config_samples/code_assistant.yaml -i "实现一个简单的Python Web服务器" -k YOUR_API_KEY
   ```

## 输入输出格式

### 输入JSON格式

```json
{
  "id": "query001",
  "input_text": "人工智能的发展历程是怎样的？",
  "metadata": {"type": "question", "category": "AI"}
}
```

### 输出JSON格式

```json
{
  "id": "query001",
  "input_text": "人工智能的发展历程是怎样的？",
  "metadata": {"type": "question", "category": "AI"},
  "processed_text": "请详细回答以下问题：\n人工智能的发展历程是怎样的？",
  "model_response": "人工智能的发展历程可以大致分为以下几个主要阶段...(完整的模型回复)",
  "summary": "人工智能的发展历程可以大致分为以下几个主要阶段...(截取的摘要)"
}
```

## 配置样例说明

### 1. 通用问答配置 (qianfan_default.yaml)

适用于一般性问答场景，使用千帆平台的LLaMA-2模型，设置了适中的回复长度和温度参数。

### 2. 代码助手配置 (code_assistant.yaml)

适用于编程和技术问题，使用更高版本的LLaMA-3模型，设置了较低的温度值以确保回复的确定性和准确性，并提供了编程相关的指令提示。

### 3. 文本摘要配置 (text_summarizer.yaml)

专为文本摘要设计，使用BLOOMZ模型，通过精心设计的提示模板指导模型生成简洁明了的摘要，适合处理长文章、新闻或报告。

## 自定义和扩展

### 添加新的预处理步骤

```python
class CustomTextProcessor(JsonOperator):
    # 自定义文本处理逻辑
    ...

# 在Pipeline中添加
pipeline = Pipeline([
    CustomTextProcessor(...),
    TextLLMProcessor(...),
    ...
])
```

### 创建自定义配置文件

创建新的YAML配置文件，指定不同的模型、提示模板和参数：

```yaml
model: "自定义模型名称"
base_url: "自定义API基础URL"
prompt: "自定义提示模板 {input}"
system_prompt: "自定义系统提示"
max_tokens: 1000
```

### 使用不同的模型API

修改`--model`和`--base-url`参数或在配置文件中指定：

```bash
python text_llm_example.py --model "openai/gpt-3.5-turbo" --base-url "https://api.openai.com/v1" --api-key YOUR_API_KEY
```

## 注意事项

- 确保您有足够的API访问权限和配额
- 对于长文本处理，请注意模型的上下文长度限制
- 处理敏感数据时，请遵循相关的数据保护和隐私政策
- 配置文件中的API密钥建议留空，通过命令行参数提供，以避免将密钥保存在配置文件中 