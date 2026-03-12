# 训练技巧通用指南

## 优化器详解

### 1. AdamW

**最常用的优化器**

```python
# PyTorch 实现
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    betas=(0.9, 0.95),
    weight_decay=0.1,
    eps=1e-8
)
```

**核心特点**：
- 动量自适应学习率
- 权重衰减与梯度解耦
- 适合 Transformer 架构

### 2. AdamW + 8-bit

**显存优化版本**

```python
from bitsandbytes import AdamW8bit

optimizer = AdamW8bit(model.parameters(), lr=1e-4)
```

**优势**：
- 减少 50% 显存
- 保持相近性能

### 3. LAMB

**大 batch size 专用**

```python
# LAMB (Layer-wise Adaptive Moments)
optimizer = torch.optim.LAMB(
    model.parameters(),
    lr=1e-4,
    betas=(0.9, 0.999),
    weight_decay=0.01
)
```

**优势**：
- 支持超大 batch size (64K+)
- 保持收敛性

### 4. Sophia

**新方法，理论上更快**

```python
# Sophia (Stochastic Optimization with Hessian Approximation)
optimizer = SophiaG(
    model.parameters(),
    lr=1e-4,
    betas=(0.965, 0.99),
    rho=0.04
)
```

**特点**：
- 使用梯度二阶统计
- 比 AdamW 更快收敛

## 学习率调度

### 1. Warmup + Cosine Decay

**推荐配置**

```python
# Cosine Annealing with Warmup
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=2000,
    num_training_steps=100000,
    num_cycles=0.5
)
```

**曲线**：
```
LR
 ↑
 |    /\        /\        /\
 |   /  \      /  \      /  \
 |  /    \    /    \    /    \
 | /      \  /      \  /      \
 |/        \/        \/        \
 +----------------------------→ Steps
   Warmup  Cosine Decay
```

### 2. Linear Decay

```python
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=1000,
    num_training_steps=50000
)
```

### 3. Polynomial Decay

```python
lr_scheduler = get_polynomial_decay_schedule_with_warmup(
    optimizer,
    num_warmup_steps=1000,
    num_training_steps=50000,
    lr_end=1e-6,
    power=2.0
)
```

### 4. 多阶段调度

```python
# 自定义调度
def lr_lambda(step):
    if step < 10000:
        return step / 10000
    elif step < 50000:
        return 1.0
    elif step < 80000:
        return 0.1
    else:
        return 0.01

lr_scheduler = LambdaLR(optimizer, lr_lambda)
```

## 分布式训练

### 1. 数据并行

#### DDP (Distributed Data Parallel)

```python
# 启动
python -m torch.distributed.launch --nproc_per_node=8 train.py

# 代码
from torch.nn.parallel import DistributedDataParallel as DDP
model = DDP(model, device_ids=[local_rank])
```

#### ZeRO (Zero Redundancy Optimizer)

**Stage 1**：优化器状态分片
**Stage 2**：梯度分片
**Stage 3**：参数分片

```python
# DeepSpeed ZeRO Stage 3
ds_config = {
    "train_batch_size": 32,
    "zero_optimization": {
        "stage": 3,
        "offload_param": {"device": "cpu"},
        "offload_optimizer": {"device": "cpu"}
    }
}
```

#### FSDP (Fully Sharded Data Parallel)

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision

model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    mixed_precision=MixedPrecision(param_dtype=torch.bfloat16)
)
```

### 2. 模型并行

#### Pipeline Parallelism

```python
# GPipe 风格
from torch.distributed.pipeline.sync import Pipe

model = nn.Sequential(
    layer1, layer2, layer3, layer4
).to_grpc()
```

#### Tensor Parallelism

```python
# Megatron-LM 风格
# 矩阵分片：Y = X @ A @ B
# A 按列分片，B 按行分片
```

### 3. 3D 并行

```
Data Parallel (副本)
    ↓
Pipeline Parallel (层分阶段)
    ↓
Tensor Parallel (层内分片)
```

## 混合精度训练

### FP16 / BF16

```python
# 自动混合精度
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
model = model.cuda()

for batch in dataloader:
    with autocast(dtype=torch.float16):
        outputs = model(batch['input'].cuda())
        loss = loss_fn(outputs, batch['label'].cuda())
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**对比**：
| 类型 | 范围 | 精度 | 稳定性 |
|------|------|------|--------|
| FP32 | ±3.4e38 | 高 | 最稳定 |
| FP16 | ±65504 | 中 | 可能溢出 |
| BF16 | ±3.4e38 | 中 | 不易溢出 |

**推荐**：BF16 用于训练，FP16 用于推理

## 梯度技巧

### 1. Gradient Checkpointing

```python
# 时间换显存
from torch.utils.checkpoint import checkpoint_sequential

model = nn.Sequential(*layers)
model = nn.Sequential(*checkpoint_sequential(model.split(4), 4))
```

### 2. Gradient Accumulation

```python
# 虚拟大 batch
accumulation_steps = 4
batch_size = 32

for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 3. Gradient Clipping

```python
# 防止梯度爆炸
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

## 超参数配置

### 1. 推荐超参数

| 参数 | 常见值 | 调整建议 |
|------|--------|----------|
| **学习率** | 1e-4 ~ 1e-5 | 增 batch 则增 LR |
| **Batch Size** | 32-512 | 渐增策略 |
| **Warmup Steps** | 500-2000 | 100K 步训练 |
| **Weight Decay** | 0.01-0.1 | 只用于权重 |
| **Dropout** | 0.1 | 微调时降低 |
| **Context Length** | 4K-32K | 按需求 |

### 2. Batch Size 缩放

**线性缩放规则**：
- Batch ×2 → LR ×2
- 参考：https://arxiv.org/abs/1706.02677

### 3. 学习率估算

| 模型规模 | 建议学习率 |
|----------|-----------|
| 1B | 3e-4 |
| 10B | 1e-4 |
| 100B | 5e-5 |

## 训练稳定性

### 1. Loss Spike 处理

```python
# 检测并处理
if loss > last_loss * 3:
    # 学习率回退
    for param_group in optimizer.param_groups:
        param_group['lr'] *= 0.5
```

### 2. NaN 检测

```python
# 早停
if torch.isnan(loss):
    logger.warning("NaN detected, loading last checkpoint")
    load_checkpoint("last.pt")
    reduce_lr()
```

### 3. Early Stopping

```python
best_loss = float('inf')
patience = 3

for epoch in range(num_epochs):
    val_loss = evaluate(model, val_loader)
    
    if val_loss < best_loss:
        best_loss = val_loss
        save_checkpoint("best.pt")
        patience = 3
    else:
        patience -= 1
        if patience == 0:
            break
```

### 4. Checkpoint 管理

```python
# 定期保存
if global_step % 1000 == 0:
    save_checkpoint(f"checkpoint_{global_step}.pt")

# 只保存权重
torch.save(model.state_dict(), "weights.pt")
# 保存完整
torch.save({
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'step': global_step,
    'loss': loss
}, "checkpoint.pt")
```

## 推理优化

### 1. KV Cache

```python
# 推理时启用
outputs = model.generate(
    input_ids,
    use_cache=True,
    max_new_tokens=100
)
```

### 2. Flash Attention

```python
# 安装后自动使用
from flash_attn import FlashAttention

model = FlashAttention(model)
```

### 3. 量化推理

```python
# GPTQ / AWQ
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "model",
    quantization_config=BitsAndBytesConfig(load_in_4bit=True)
)
```

## 常见问题解决

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| Loss = NaN | 学习率太高 | 降低 LR，使用 BF16 |
| Loss 不下降 | 数据质量问题 | 检查数据，清洗 |
| OOM | 显存不足 | 减小 batch，使用 ZeRO |
| Loss spike | 异常 batch | 跳过或降低 LR |
| 震荡 | LR 太高 | 降低 LR，加 warmup |

## 参考文献

1. **AdamW 论文**: https://arxiv.org/abs/1711.05101
2. **LAMB 论文**: https://arxiv.org/abs/1904.00962
3. **DeepSpeed**: https://www.microsoft.com/en-us/research/blog/deepspeed/
4. **Megatron-LM**: https://github.com/NVIDIA/Megatron-LM
5. **Flash Attention**: https://arxiv.org/abs/2205.14135
6. **Mixed Precision Training**: https://arxiv.org/abs/1710.03740

---

*最后更新：2024*
