# AI模型训练洞察

深入研究主流大模型的训练流程、数据准备、硬件基础设施与架构特点。

## 📁 项目结构

| 目录 | 模型 | 文档 |
|------|------|------|
| `GPT/` | GPT-4, o1, o3 | training.md, GPT_hardware.md |
| `LLaMA/` | LLaMA 1-3, 3.1, 3.2 | training.md, LLaMA_hardware.md |
| `Claude/` | Claude 3-4 | training.md, Claude_hardware.md |
| `GLM/` | GLM-4, 4.5, 4.7, 5 | training.md, overview.md, GLM_hardware.md |
| `DeepSeek/` | DeepSeek V2, V3, Coder | training.md, DeepSeek_hardware.md |
| `QWEN/` | Qwen 2, 2.5, 3, 3.5 | training.md, QWEN_hardware.md |
| `MiniMax/` | abab, M2 | training.md, MiniMax_hardware.md |

## 📚 文档类型

- **training.md** - 训练流程与数据准备
- **overview.md** - 整体演进与汇报（部分模型）
- ***_hardware.md** - 训练硬件与平台基础设施
- `common/` - 通用资料与名词解释

## 🚀 快速开始

```bash
# 克隆项目
git clone https://github.com/Amoo1121/ai-model-training-insights.git

# 查看特定模型
cat AI模型训练洞察/GLM/overview.md
```

## 📊 内容覆盖

### 训练流程
- 预训练 (Pre-training)
- 有监督微调 (SFT)
- 对齐训练 (RLHF/DPO)

### 硬件基础设施
- GPU 型号与数量
- 训练平台 (AWS/GCP/阿里云/自研)
- 分布式训练方案
- 训练成本估算

### 模型架构
- Dense vs MoE
- 注意力机制
- 上下文长度演进

## 🤝 贡献

欢迎提交 PR 和 Issue！

---
*AI 生产线平台洞察项目*
