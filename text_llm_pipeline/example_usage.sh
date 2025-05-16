#!/bin/bash
# 文本语言模型CLI工具使用示例

# 设置API密钥
API_KEY="your_api_key_here"

echo "=== 文本语言模型CLI工具使用示例 ==="
echo

# 示例1: 基本问答
echo "示例1: 使用基本问答配置"
python3 llm_invoker_cli.py \
  --config config_samples/qianfan_default.yaml \
  --input "人工智能对未来社会的影响是什么？" \
  --api-key $API_KEY \
  --verbose

# 示例2: 代码生成
echo
echo "示例2: 使用代码助手配置"
python3 llm_invoker_cli.py \
  --config config_samples/code_assistant.yaml \
  --input "用Python编写一个简单的Web服务器，能够提供静态文件服务" \
  --api-key $API_KEY

# 示例3: 文本摘要
echo
echo "示例3: 使用文本摘要配置处理文件"
# 创建一个示例文本文件
cat > example_text.txt << EOF
人工智能（Artificial Intelligence，简称AI）是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。自从1956年在达特茅斯会议上提出"人工智能"的概念以来，人工智能已经走过了60多年的发展历程。

早期的人工智能主要采用符号逻辑推理方法，专注于解决如象棋等规则明确的问题。1980年代后期，机器学习方法，特别是神经网络开始受到重视。但由于计算能力限制和数据不足，这一时期的研究进展缓慢，被称为"AI寒冬"。

21世纪以来，随着计算能力的提升和大数据的出现，深度学习技术取得了突破性进展。2012年，深度神经网络在图像识别任务中首次超越了人类专家，揭开了人工智能新时代的序幕。自那时起，人工智能技术在图像识别、语音识别、自然语言处理等多个领域取得了显著成就。

如今，人工智能已经深入到我们生活的方方面面。从智能手机上的语音助手，到推荐系统，再到自动驾驶汽车，人工智能正在改变人们的生活和工作方式。未来，随着技术的进一步发展，人工智能有望在医疗健康、教育、环境保护等领域发挥更大的作用，帮助人类应对各种挑战。

然而，人工智能的发展也带来了一系列伦理和社会问题，如隐私保护、就业变化、算法偏见等。如何确保人工智能的发展方向与人类福祉一致，已成为学术界和产业界共同关注的重要议题。
EOF

python3 llm_invoker_cli.py \
  --config config_samples/text_summarizer.yaml \
  --input example_text.txt \
  --output example_summary.txt \
  --api-key $API_KEY \
  --verbose

# 显示生成的摘要
echo
echo "生成的摘要:"
cat example_summary.txt

# 清理示例文件
# rm example_text.txt example_summary.txt

echo
echo "===== 示例完成 =====" 