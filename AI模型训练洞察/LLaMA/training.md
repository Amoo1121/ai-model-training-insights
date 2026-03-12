# LLaMA 系列模型训练洞察

## 模型系列

- LLaMA 1 (7B, 13B, 65B) - 2023年2月
- LLaMA 2 (7B, 13B, 70B) - 2023年7月
- LLaMA 3 (8B, 70B, 400B) - 2024年4月
- LLaMA 3.1 (8B, 70B, 405B) - 2024年7月
- CodeLLaMA - 代码专用
- LLaMA Guard - 安全审查

## 训练流程

### 1. 预训练（Pre-training）

#### 数据来源与配比

| 版本 | 总 Token 数 | 数据来源 | 配比 |
|------|------------|----------|------|
| LLaMA 1 | 1.4T | Common Crawl, GitHub, Wikipedia, Books | 见下文 |
| LLaMA 2 | 2T | 同上，数据质量提升 | 见下文 |
| LLaMA 3 | 15T | 大量网页、代码 | 50% 通用文本，25% 代码，25% 数学/科学 |

**LLaMA 3 详细数据配比**：
- **网页文本**：50% - 包括 Common Crawl 和高质量网页
- **代码**：25% - GitHub、代码数据集
- **数学/科学**：25% - 学术论文、数学网页

#### 数据清洗方法

1. **质量过滤**：
   - **CCNet**：语言识别 → 质量评分 → 过滤
   - **质量分类器**：使用 LightGBM 训练的分类器
   - **Perplexity 过滤**：使用小模型计算困惑度

2. **去重**：
   - **精确去重**：Document-level SHA256 哈希
   - **近似去重**：MinHash + LSH (Locality Sensitive Hashing)
   - **句子级去重**：移除重复句子

3. **启发式规则**：
   - 移除过短文档 (< 200 tokens)
   - 移除异常长度文档
   - 过滤非自然语言内容

4. **分词器**：
   - **Tiktoken**：BPE 算法
   - **词表大小**：128k (LLaMA 3)
   - **SentencePiece**：可选

#### 训练配置

| 模型 | 隐藏层 | 注意力头 | 头维度 | FFN 隐藏 | 序列长度 |
|------|--------|----------|--------|----------|----------|
| LLaMA 7B | 32 | 32 | 128 | 11008 | 4096 |
| LLaMA 13B | 40 | 40 | 128 | 13824 | 4096 |
| LLaMA 65B | 80 | 64 | 128 | 21152 | 4096 |
| LLaMA 3 8B | 32 | 32 | 128 | 14336 | 8192 |
| LLaMA 3 70B | 80 | 64 | 128 | 28672 | 8192 |

**训练超参数**：
- **优化器**：AdamW (β1=0.9, β2=0.95)
- **学习率**：峰值 3e-4 (LLaMA 3), 1e-3 (LLaMA 1/2)
- **学习率调度**：
  - Warmup: 2000 steps
  - Cosine Annealing 到 10% 峰值
- **批量大小**：动态，渐增策略
- **权重衰减**：0.1
- **梯度裁剪**：1.0

### 2. 有监督微调（SFT）

#### 数据来源

| 数据集 | 数量 | 描述 |
|--------|------|------|
| InstructChat | 10k | 人工标注的指令对话 |
| ShareGPT | 70k | 用户分享的 ChatGPT 对话 |
| Alpaca | 52k | Self-Instruct 生成 |
| OpenAssistant | 161k | 开源指令对齐数据 |

#### 训练配置
- **学习率**：2e-5 (LLaMA 3), 1e-4 (LLaMA 1/2)
- **Epochs**：1-3
- **批量大小**：64
- **序列长度**：4096 / 8192
- **梯度累积**：4-8 步

### 3. 对齐训练（RLHF / DPO）

#### RLHF (LLaMA 1/2)

**奖励模型**：
- 数据：人类偏好排序数据 (~1M 对)
- 架构：与 SFT 相同 + 线性输出层
- 训练：pairwise ranking loss

**PPO 优化**：
- 策略模型初始化：SFT 模型
- KL 惩罚系数：0.1
- PPO clip：0.2
- GAE λ：0.95

#### DPO (LLaMA 3 - 推荐方法)

**优势**：
- 无需显式奖励模型
- 训练更稳定
- 计算效率高

**数据偏好**：
- 收集 10M+ 人类偏好对
- 覆盖 30+ 语言

**损失函数**：
```
L = -log(σ(r(x,y^+) - r(x,y^-)))
其中 r 是策略模型输出的 reward
```

**超参数**：
- β (温度参数)：0.1
- 学习率：1e-6
- Batch Size：512

## 架构特点

本节聚焦 LLaMA 系列在归一化、位置编码、注意力机制和激活函数上的典型设计。

### 核心组件

#### 1. 归一化 (RMSNorm)

**特点**：
- 不使用 softmax 归一化的偏置
- 计算更高效
- 提升训练稳定性

```python
# RMSNorm
def rms_norm(x):
    norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x * norm * weight
```

#### 2. 位置编码 (RoPE - Rotary Position Embedding)

**特点**：
- 旋转式位置编码，无需额外参数
- 更好的外推能力
- 支持更长上下文

**实现**：
```python
# RoPE 核心
def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat([-x2, x1], dim=-1)

def apply_rotary_emb(x, cos, sin):
    return (x * cos) + (rotate_half(x) * sin)
```

#### 3. 注意力机制

##### Grouped Query Attention (GQA)

**特点**（LLaMA 3 应用）：
- 多查询注意力的变体
- 平衡质量与效率
- K/V 头数 < Q 头数

| 模型 | Q 头数 | KV 头数 | 比例 |
|------|--------|---------|------|
| LLaMA 2 | 32/40/64 | 32/40/64 | 1:1 |
| LLaMA 3 | 32/64 | 8 | 4:1 |

##### Sliding Window Attention

**特点**（可选）：
- 局部注意力，降低计算
- 窗口大小：4096
- 兼顾全局信息

#### 4. 激活函数 (SwiGLU)

**特点**：
- Swish 激活 + Gated Linear Unit
- 比 ReLU/GELU 更好的性能

```python
# SwiGLU
def swiglu(x, gate):
    return F.silu(x) * gate
```

### 完整架构图

```
Input → Embedding → [RMSNorm → QKV Projection → RoPE → Attention → Dropout] × N
                                           ↓
                                   RMSNorm → SwiGLU FFN → Dropout
                                           ↓
                        → LM Head → Output
```

## 训练资源

| 模型 | GPU | GPU Hours | 训练 Token 数 |
|------|-----|-----------|--------------|
| LLaMA 7B | 8x A100 | 82K | 1.0T |
| LLaMA 13B | 8x A100 | 135K | 1.0T |
| LLaMA 65B | 8x A100 | 1,021K | 1.4T |
| LLaMA 3 8B | H100 x 16K | ~50K | 15T |
| LLaMA 3 70B | H100 x 128K | ~700K | 15T |

## 数据格式

### 预训练格式
```json
{"text": "The quick brown fox...", "source": "common_crawl", "quality_score": 0.95}
```

### SFT 格式
```json
{
  "messages": [
    {"role": "user", "content": "请解释量子纠缠"},
    {"role": "assistant", "content": "量子纠缠是..."}
  ]
}
```

### DPO 格式
```json
{
  "prompt": "解释相对论",
  "chosen": "详细且准确的解释...",
  "rejected": "简短且不准确的解释..."
}
```

## 训练技巧

### 1. 优化器配置
- **AdamW**：权重衰减 0.1
- **学习率调度**：Cosine decay + 线性 warmup
- **梯度裁剪**：max_norm=1.0

### 2. 显存优化
- **Gradient Checkpointing**：时间换显存
- **Mixed Precision**：BF16 + FP32
- **Flash Attention 2**：更高效的注意力计算

### 3. 分布式训练
- **FSDP (Fully Sharded Data Parallel)**
- **ZeRO Stage 3**
- **Tensor Parallelism**：大模型跨设备分片

### 4. 训练稳定性
- **Loss spike 处理**：学习率回退
- **NaN 检测**：提前终止
- **Checkpoint**：定期保存

## LLaMA 3 特殊优化

### 工具调用 (Function Calling)
- 专用微调数据
- JSON 输出格式优化

### 系统指令
- Meta-ChatTemplate
- 角色扮演能力

### 安全性
- LLaMA Guard 3
- 红队测试数据

## 参考文献

1. **LLaMA: Open and Efficient Foundation Language Models**: https://arxiv.org/abs/2302.13971
2. **LLaMA 2: Open Foundation and Chat Models**: https://ai.meta.com/llama/
3. **LLaMA 3 论文**: https://arxiv.org/abs/2407.21783
4. **RoPE 论文**: https://arxiv.org/abs/2104.09864
5. **SwiGLU 论文**: https://arxiv.org/abs/2002.05202
6. **GQA 论文**: https://arxiv.org/abs/2305.13245
7. **Flash Attention**: https://github.com/Dao-AILab/flash-attention
8. **DPO 论文**: https://arxiv.org/abs/2305.18290

---

*最后更新：2024*
