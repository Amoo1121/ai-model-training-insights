# 数据准备通用指南

## 数据类型详解

### 1. 文本数据

| 来源 | 特点 | 获取方式 |
|------|------|----------|
| **Common Crawl** | 规模最大(~50TB/月)，质量不一 | CCNet 管道下载 |
| **WebText** | 高质量，Reddit 3K+ 链接文章 | OpenAI 发布 |
| **Wikipedia** | 结构化，高质量，多语言 | 官方 dumps |
| **Books** | 连续性强，知识密集 | BookCorpus, LibriSpeech |
| **GitHub** | 代码数据，开源为主 | GH Torrent |

### 2. 代码数据

| 数据集 | 描述 | Token 数 |
|--------|------|----------|
| **StarCoder** | GitHub 代码，80+ 语言 | 250B+ |
| **The Stack** | 30+ 语言代码 | 3TB+ |
| **CodeSearchNet** | 搜索优化代码 | 6M |
| **BigCode** | 开源许可代码 | 1.2TB |

### 3. 对话数据

| 数据集 | 描述 | 规模 |
|--------|------|------|
| **ShareGPT** | 用户与 ChatGPT 对话 | 70K |
| **Anthropic HH-RLHF** | 有帮助/无害偏好 | 170K |
| **OpenAssistant** | 开源指令数据 | 161K |
| **FLAN** | 多种任务指令集合 | 1.8M |

### 4. 数学与科学数据

| 数据集 | 描述 |
|--------|------|
| **ArXiv** | 学术论文 |
| **MathStackExchange** | 数学问答 |
| **MMLU** | 多任务语言理解 |
| **GSM8K** | 数学推理 |

## 数据清洗流程

### 阶段 1：原始数据获取

```
原始数据 → 下载 → 解压 → 初步筛选
```

### 阶段 2：语言过滤

```python
# 语言检测
from langdetect import detect

def filter_language(text, target_lang='en'):
    try:
        return detect(text) == target_lang
    except:
        return False
```

### 阶段 3：质量评分

#### 方法 1：Perplexity 过滤
```python
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def compute_perplexity(text, model, tokenizer):
    encodings = tokenizer(text, return_tensors='pt')
    loss = model(encodings['input_ids'], labels=encodings['input_ids']).loss
    return torch.exp(loss).item()

# 过滤规则：PPL > 1000 → 低质量
```

#### 方法 2：分类器过滤
```python
# 使用预训练分类器
from sklearn.linear_model import LogisticRegression

# 特征：文档长度、特殊字符比例、重复词比例等
# 标签：人工标注质量分数
```

### 阶段 4：去重

#### 精确去重
```python
import hashlib

def exact_deduplicate(documents):
    seen = set()
    unique_docs = []
    
    for doc in documents:
        doc_hash = hashlib.sha256(doc['text'].encode()).hexdigest()
        if doc_hash not in seen:
            seen.add(doc_hash)
            unique_docs.append(doc)
    
    return unique_docs
```

#### 近似去重 (MinHash)
```python
from datasketch import MinHash, MinHashLSH

def approximate_deduplicate(documents, threshold=0.8):
    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    unique_docs = []
    
    for doc in documents:
        minhash = MinHash(num_perm=128)
        for word in doc['text'].split():
            minhash.update(word.encode())
        
        if not lsh.contains(minhash):
            lsh.insert(doc['id'], minhash)
            unique_docs.append(doc)
    
    return unique_docs
```

### 阶段 5：敏感内容过滤

```python
# 敏感词过滤
SENSITIVE_WORDS = ['violence', 'hate', 'illegal', ...]

def filter_sensitive(text):
    text_lower = text.lower()
    for word in SENSITIVE_WORDS:
        if word in text_lower:
            return False
    return True
```

## 数据格式标准

### 1. 预训练格式 (JSONL)

```json
{"text": "The quick brown fox jumps over the lazy dog.", "source": "common_crawl", "quality_score": 0.95}
{"text": "Another document...", "source": "wikipedia", "quality_score": 0.98}
```

### 2. SFT 格式

```json
{"messages": [{"role": "user", "content": "What is AI?"}, {"role": "assistant", "content": "AI is..."}]}
```

### 3. RLHF 偏好格式

```json
{"prompt": "Explain quantum computing", "chosen": "Good response", "rejected": "Poor response"}
```

### 4. HuggingFace Dataset 格式

```python
from datasets import Dataset

dataset = Dataset.from_dict({
    "text": ["doc1", "doc2", ...],
    "source": ["common_crawl", "wikipedia", ...],
    "quality": [0.95, 0.98, ...]
})

# 或从磁盘加载
dataset = Dataset.load_from_disk("path/to/dataset")
```

## 数据配比策略

### 通用大模型配比

| 数据类型 | 配比 | 说明 |
|----------|------|------|
| 网页文本 | 60-70% | 知识广度 |
| 代码 | 15-20% | 推理与结构化 |
| 书籍 | 10-15% | 深度知识 |
| 对话 | 5% | 对话能力 |

### 代码大模型配比

| 数据类型 | 配比 |
|----------|------|
| 代码 | 70-80% |
| 代码文档 | 10-15% |
| 自然语言 | 10-15% |

### 数学大模型配比

| 数据类型 | 配比 |
|----------|------|
| 数学文本 | 40% |
| 代码 | 30% |
| 科学论文 | 20% |
| 通用文本 | 10% |

## 数据质量评估指标

### 1. 质量指标

| 指标 | 计算方法 | 阈值建议 |
|------|----------|----------|
| PPL | 语言模型困惑度 | < 500 |
| 重复率 | 重复 n-gram 比例 | < 10% |
| 特殊字符比 | 特殊字符/总字符 | < 5% |

### 2. 多样性指标

| 指标 | 计算方法 |
|------|----------|
| 词汇多样性 | Unique tokens / Total tokens |
| 主题多样性 | 聚类分布熵 |
| 来源多样性 | 来源分布 |

### 3. 安全指标

| 检查项 | 方法 |
|--------|------|
| 成人内容 | 分类器 |
| 恶意软件 | 签名匹配 |
| 隐私信息 | NER + 规则 |
| 仇恨言论 | 分类器 |

## 数据流水线实现

### 完整管道示例

```python
from datasets import load_dataset
from tqdm import tqdm

class DataPipeline:
    def __init__(self, config):
        self.config = config
        
    def process_common_crawl(self):
        # 1. 加载原始数据
        raw = load_dataset("c4", "en", split="train")
        
        # 2. 语言过滤
        raw = raw.filter(lambda x: x['language'] == 'en')
        
        # 3. 质量过滤
        raw = raw.filter(lambda x: x['perplexity'] < 500)
        
        # 4. 去重
        raw = self.deduplicate(raw)
        
        # 5. 过滤敏感内容
        raw = raw.filter(self.filter_sensitive)
        
        return raw
    
    def create_sft_dataset(self):
        # 指令数据构建
        return load_dataset("tatsu-lab/alpaca", split="train")
    
    def create_preference_dataset(self):
        # 偏好数据构建
        return load_dataset("Anthropic/hh-rlhf", split="train")
```

## 数据集发布资源

### 开源数据集

1. **C4 (Colossal Clean Crawled Corpus)**：https://huggingface.co/datasets/c4
2. **RedPajama**：https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T
3. **Falcon**：https://huggingface.co/datasets/tiiuae/falcon-refinedweb
4. **OpenWebText**：https://huggingface.co/datasets/Skylion007/openwebtext

### 代码数据集

1. **StarCoder**：https://huggingface.co/datasets/bigcode/starcoderdata
2. **The Stack**：https://huggingface.co/datasets/bigcode/the-stack

### 对话数据集

1. **ShareGPT**：https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered
2. **OpenAssistant**：https://huggingface.co/datasets/OpenAssistant/oasst2

## 参考文献

1. **CCNet**: https://arxiv.org/abs/1911.00359
2. **RedPajama**: https://www.togethercomputer.com/redpajama
3. **C4 Dataset**: https://github.com/google-research/text-to-text-transfer-transformer
4. **清洗最佳实践**: https://arxiv.org/abs/2107.06499
5. **数据质量评估**: https://arxiv.org/abs/2305.17364

---

*最后更新：2024*
