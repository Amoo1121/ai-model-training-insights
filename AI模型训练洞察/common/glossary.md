# AI 训练名词解释

## 基础概念

### 预训练 (Pre-training)
大规模无监督学习阶段，使用海量文本数据训练模型预测下一个词元。模型从互联网、书籍、代码等处学习语言的通用模式和知识。

**为什么重要**：是模型获得"语言能力"的基础，决定了模型的知识储备和推理能力。

**例子**：GPT-3 使用 45TB 文本训练，LLaMA-3 使用 15T tokens。

---

### 有监督微调 (SFT - Supervised Fine-Tuning)
在预训练模型基础上，使用人工标注的指令-响应对数据进行微调。

**为什么重要**：让模型学会理解人类指令，产生符合预期的回答。

**例子**：给模型输入"请总结这段文章"，训练模型输出正确的摘要。

**参见**：RLHF、Prompt Engineering

---

### 人类反馈强化学习 (RLHF - Reinforcement Learning from Human Feedback)
使用人类对模型输出的评分/排序训练奖励模型，再用强化学习（PPO算法）优化语言模型。

**为什么重要**：让模型输出更有帮助、更安全、更符合人类偏好。

**例子**：ChatGPT 通过 RLHF 显著提升对话质量。

**参见**：Reward Model、PPO、DPO

---

### 直接偏好优化 (DPO - Direct Preference Optimization)
一种简化的对齐训练方法，直接从人类偏好数据（哪个回答更好）优化模型，无需显式训练奖励模型。

**为什么重要**：比 RLHF 更简单、更稳定、训练成本更低。

**例子**：LLaMA 2 使用 DPO 进行对齐训练。

**参见**：RLHF、Preference Data

---

## 模型架构

### Transformer
2017年提出的深度学习架构，核心是自注意力机制（Self-Attention），能并行处理序列数据。

**为什么重要**：当前所有主流大模型的基础架构。

**例子**：GPT、BERT、LLaMA、Claude 都基于 Transformer。

**参见**：Self-Attention、Encoder-Decoder

---

### MoE (Mixture of Experts)
混合专家模型，包含多个"专家"网络，每次只激活部分专家处理输入。

**为什么重要**：在保持高质量的同时大幅降低推理成本。

**例子**：GPT-4 是 MoE 模型（8x2200亿参数），Switch Transformer。

**参见**：Sparse MoE、Expert Routing

---

### RoPE (Rotary Position Embedding)
旋转位置编码，通过旋转矩阵编码位置信息。

**为什么重要**：相比传统位置编码，效果更好，能处理更长上下文。

**例子**：LLaMA、Qwen 系列使用 RoPE。

**参见**：Position Embedding、ALiBi

---

### GQA (Grouped Query Attention)
分组查询注意力，K/V 头少于 Q 头，平衡质量和效率。

**为什么重要**：显著减少 KV 缓存，提升推理效率。

**例子**：LLaMA 2 70B 使用 GQA。

**参见**：Multi-Head Attention、KV Cache

---

## 训练技术

### 梯度累积 (Gradient Accumulation)
当 GPU 显存不够大时，将多个小 batch 的梯度累积后再更新参数。

**为什么重要**：让小显存也能训练大模型。

**例子**：batch_size=1，累积 32 次 = 实际 batch_size=32。

**参见**：Batch Size、Mixed Precision

---

### 混合精度训练 (Mixed Precision)
使用 FP16/BF16 而非 FP32 进行训练和推理。

**为什么重要**：提升训练速度 2-3 倍，减少 50% 显存。

**例子**：使用 NVIDIA Apex 或 PyTorch AMP。

**参见**：FP16、BF16、AMP

---

### DeepSpeed ZeRO
微软的分布式训练优化技术，分片存储优化器状态、梯度、参数。

**为什么重要**：让消费级 GPU 也能训练大模型。

**例子**：ZeRO-3 可在 8 张 A100 训练千亿参数模型。

**参见**：Data Parallelism、Model Parallelism

---

### Flash Attention
高效的注意力计算方法，使用 IO-aware 算法减少显存访问。

**为什么重要**：注意力计算从 O(N²) 显存降到 O(N)，显著加速。

**例子**：Flash Attention 2 比标准 Attention 快 2-4 倍。

**参见**：Attention、Sparse Attention

---

## 数据相关

### Token
词元，文本处理的基本单位。1 token ≈ 1-4 个字符。

**为什么重要**：大模型按 token 收费和训练。

**例子**："人工智能" = 4 tokens (人工智能/是/什么/？)

**参见**：Tokenizer、Vocab Size

---

### 数据配比 (Data Ratio)
训练数据中不同来源（网页、代码、书籍、对话等）的比例。

**为什么重要**：配比影响模型能力偏向。

**例子**：LLaMA 3 配比 ≈ 50% 网页 + 25% 代码 + 12.5% 书籍 + 12.5% 其他

**参见**：Common Crawl、GitHub、Wikipedia

---

### 数据清洗 (Data Filtering)
去除低质量、重复、敏感内容的数据预处理。

**为什么重要**：数据质量决定模型质量。

**方法**：
- 语言检测（去除非目标语言）
- 质量评分（Perplexity、LLM 评分）
- 去重（精确哈希、MinHash）
- 敏感词过滤

**参见**：CCNet、RefinedWeb

---

## 对齐技术

### Constitutional AI
Anthropic 提出的 AI 对齐方法，让模型根据一套"行为准则"自我约束行为。

**为什么重要**：比人工标注更 scalable 的安全对齐方案。

**参见**：RLHF、AI Safety、Red Teaming

---

### RLAIF (RL from AI Feedback)
使用 AI（而非人类）提供反馈进行强化学习。

**为什么重要**：解决 RLHF 的人类标注成本高问题。

**例子**：Anthropic 用 AI 反馈训练 Claude。

**参见**：RLHF、Constitutional AI

---

### Red Teaming
红队测试，故意让模型产生有害输出以发现和修复安全问题。

**为什么重要**：发现模型的潜在风险和漏洞。

**参见**：AI Safety、Constitutional AI

---

## 评估指标

### Perplexity (PPL)
困惑度，衡量模型预测下一个词的能力。数值越低越好。

**为什么重要**：语言模型的核心指标。

**参见**：Cross-Entropy、Bits Per Character

---

### BLEU / ROUGE
机器翻译和文本摘要的评估指标，比较生成文本与参考文本的重叠程度。

**为什么重要**：传统 NLP 任务的标准化评估指标。

**参见**：Text Generation、Summarization

---

### MMLU (Multi-task Language Understanding)
57 个学科的多选题测试，评估模型综合知识能力。

**为什么重要**：当前最权威的大模型评测基准之一。

**例子**：GPT-4 在 MMLU 得分约 86%，人类平均 34%。

**参见**：BIG-Bench、HumanEval

---

### BIG-Bench
Google 推出的综合性大模型评测基准，包含 200+ 任务。

**为什么重要**：测试模型在推理、编程、知识等多维度的能力。

**参见**：MMLU、HumanEval
