# AI 训练名词解释

## 基础概念

### 预训练 (Pre-training)
大规模无监督学习，让模型学习通用语言表示。使用海量文本数据训练预测下一个词元。

### 有监督微调 (SFT - Supervised Fine-Tuning)
使用人工标注的指令-响应对数据进行微调，让模型获得指令跟随能力。

### 人类反馈强化学习 (RLHF - Reinforcement Learning from Human Feedback)
使用人类反馈训练奖励模型，再用强化学习优化语言模型。

### 直接偏好优化 (DPO - Direct Preference Optimization)
无需显式奖励模型，直接从人类偏好数据优化模型，更简单稳定。

## 模型架构

### Transformer
目前主流的大语言模型架构，基于自注意力机制。

### MoE (Mixture of Experts)
混合专家模型，多个专家网络动态激活，提高参数效率。

### RoPE (Rotary Position Embedding)
旋转位置编码，一种位置编码方法。

### GQA (Grouped Query Attention)
分组查询注意力，减少 KV 缓存但保持效果。

## 训练技术

### 梯度累积 (Gradient Accumulation)
小 batch 无法容纳大模型时，分多步累积梯度再更新。

### 混合精度训练 (Mixed Precision)
使用 FP16/BF16 加速训练，减少显存。

### DeepSpeed ZeRO
微软的分布式训练优化技术，降低显存占用。

### Flash Attention
高效的注意力计算方法，显著降低显存和计算量。

## 数据相关

### Token
词元，文本处理的基本单位。

### 数据配比
训练数据中不同来源（网页、代码、书籍等）的比例。

### 数据清洗
去重、质量过滤、敏感内容过滤等预处理。

### Scale
数据规模和模型规模的度量。

## 对齐技术

### Constitutional AI
Anthropic 提出的 AI 对齐方法，基于行为准则自我约束。

### RLAIF (RL from AI Feedback)
使用 AI 反馈代替人类反馈进行强化学习。

### Red Teaming
红队测试，故意让模型产生有害输出以改进安全性。

## 评估指标

### Perplexity (PPL)
困惑度，衡量模型预测能力的指标，越低越好。

### BLEU / ROUGE
机器翻译和摘要的评估指标。

### MMLU / BIG-Bench
大模型综合能力评测基准。
