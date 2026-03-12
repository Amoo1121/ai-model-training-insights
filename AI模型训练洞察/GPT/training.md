# GPT 系列模型训练洞察

## 模型系列

- GPT-1 (117M)
- GPT-2 (1.5B)
- GPT-3 (175B)
- GPT-3.5 (InstructGPT, ChatGPT)
- GPT-4 (1.7T MoE)
- GPT-4o / GPT-4o mini
- GPT-5 (最新)

## 训练流程

### 1. 预训练 (Pre-training)

**目标**：让模型学习通用语言表示，构建世界知识

**数据来源与配比**：
| 数据集 | 占比 | 说明 |
|--------|------|------|
| Common Crawl | ~60% | 网页文本，经过清洗和去重 |
| WebText2 | ~22% | Reddit 链接文章，高质量 |
| Books1 | ~8% | 书籍 |
| Books2 | ~8% | 书籍 |
| Wikipedia | ~3% | 英文百科全书 |

**数据清洗方法**：
1. **CCNet 管道**：语言识别 → 质量过滤（Perplexity评分）→ 去重
2. **分词**：BPE (Byte Pair Encoding)，GPT-3 使用 50257 token 词表
3. **质量过滤**：使用 LightML 分类器过滤低质量文档
4. **去重**：精确去重（SHA256哈希）+ 近似去重（MinHash）

**训练方法**：
- **目标**：Next Token Prediction (NTP) / Casual Language Modeling
- **架构**：Transformer Decoder-only
- **优化器**：AdamW (β1=0.9, β2=0.95)
- **学习率**：预热 + Cosine Decay，峰值约 1.2e-4 (GPT-3)
- **批量大小**：3.2M tokens (GPT-3)
- **训练步数**：300B+ tokens

**分布式训练**：
- DeepSpeed ZeRO Stage 3
- Pipeline Parallelism
- 混合精度训练 (FP16/BF16)

### 2. 有监督微调 (SFT - Supervised Fine-Tuning)

**目标**：赋予模型指令跟随能力

**数据来源**：
| 数据集 | 数量 | 来源 |
|--------|------|------|
| InstructGPT SFT | ~13k | 人工标注 |
| Alpaca | ~52k | Self-Instruct 自动生成 |
| FLAN | ~1.8M | 多种任务集合 |

**数据格式**：
```json
{
  "messages": [
    {"role": "user", "content": "请解释量子计算"},
    {"role": "assistant", "content": "量子计算是..."}
  ]
}
```

**训练配置**：
- 学习率：~5e-6
- Epochs：2-3
- 批量大小：32-64
- 序列长度：4096

### 3. 强化学习对齐 (RLHF)

**阶段 1：奖励模型 (Reward Model) 训练**

**数据**：人类偏好排序数据
- **Promptiverse**：各类指令
- **Helpfulness 数据集**：人类标注的偏好对

**训练方法**：
- 使用 SFT 模型初始化
- 输入 prompt + response，输出标量奖励
- loss: pairwise ranking loss (Bradley-Terry 模型)

```python
# 奖励模型损失函数
loss = -log(sigmoid(ranking_loss(preferred_response) - ranking_loss(rejected_response)))
```

**阶段 2：PPO 优化**

**算法**：Proximal Policy Optimization (PPO) + KL 散度约束

**目标函数**：
```
maximize E[π(a|s) * A(s,a)] - β * KL(π || π_SFT)
```

**关键组件**：
- **PPO-clip**：限制策略更新幅度
- **价值函数**：估计期望回报
- **KL 惩罚**：防止偏离 SFT 太远
- **优势估计**：GAE (Generalized Advantage Estimation)

**超参数**：
- λ (GAE): 0.95
- γ (discount): 0.99
- β (KL penalty): 0.01-0.1

### 4. DPO (Direct Preference Optimization) - 可选替代

**原理**：直接用偏好数据优化策略，无需显式奖励模型

**损失函数**：
```
L = -log(σ(r(x,y+) - r(x,y-)))
```

**优点**：更简单，训练更稳定

## 架构特点 (GPT-4 为例)

### 模型架构
- **类型**：Transformer Decoder-only
- **参数规模**：1.7T (MoE 架构)
- **层数**：120层
- **隐藏维度**：~16,000
- **注意力头数**：128

### Attention 机制
- **标准**：Multi-Head Self-Attention
- **优化**：Sparse Attention (局部 + 全局)
- **位置编码**： Learned Positional Embeddings (GPT-4)

### 归一化
- **Pre-Norm**：每个子层前进行 LayerNorm
- **GPT-3**：使用 Post-LayerNorm

### 激活函数
-3**：GELU- **GPT (Gaussian Error Linear Unit)
- **词表**：50,257 tokens

### MoE (Mixture of Experts) - GPT-4
- 8个专家路由
- Top-2 激活机制
- 每 token 激活约 300B 参数

## 训练资源

| 模型 | GPU | GPU Hours | 训练 Token 数 |
|------|-----|-----------|--------------|
| GPT-3 175B | V100 x 1024 | ~3640K | 300B |
| GPT-4 | A100 x 25K | ~100M+ | 13T+ |

## 数据格式

### 预训练格式
```json
{"text": "The quick brown fox jumps over the lazy dog", "source": "common_crawl"}
```

### SFT 格式
```json
{
  "messages": [
    {"role": "user", "content": "用户输入"},
    {"role": "assistant", "content": "模型输出"}
  ]
}
```

### RLHF 格式
```json
{
  "prompt": "用户指令",
  "chosen": "偏好响应",
  "rejected": "非偏好响应"
}
```

## 训练技巧

### 优化策略
- **Learning Rate Warmup**：前 0.1% 步数线性增加
- **Gradient Clipping**：max_norm=1.0
- **AdamW**：权重衰减 0.1
- **混合精度**：BF16 + FP32 Master Weights
- **ZeRO**：分片优化，减少显存占用

### 稳定性保障
- Loss Spike 检测与恢复
- NaN/Inf 检测
- Checkpoint 保存
- Early Stopping 监控验证集

## 参考文献

1. **GPT-4 Technical Report**: https://arxiv.org/abs/2303.08774
2. **InstructGPT Paper**: https://arxiv.org/abs/2203.02155
3. **Language Models are Few-Shot Learners (GPT-3)**: https://arxiv.org/abs/2005.14165
4. **DeepSpeed**: https://www.microsoft.com/en-us/research/blog/deepspeed-ai-systems-enable-fast-training-of-large-scale-models/
5. **LLM-RLHF (PPO)**: https://arxiv.org/abs/1909.08593
6. **DPO Paper**: https://arxiv.org/abs/2305.18290

---

*最后更新：2024*
