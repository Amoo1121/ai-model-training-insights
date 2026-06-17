# 公开资料视角：业界模型在 SFT 与 RL 阶段用了多少数据和模型资源

> 目标：只基于**论文、技术报告、模型卡、官方文档**中能核实的信息，整理主流模型在 SFT（监督微调）与 RL / 偏好优化阶段的数据规模、参与的模型资源，以及公开透明度。
>
> 原则：**有数字写数字，没公开就写“未披露”**，不拿传闻硬凑账单。

---

## 一、先说结论

1. **SFT 数据规模跨度很大**：从 InstructGPT 的约 `13k` prompts，到 Llama 2-Chat 的 `27,540` 条高质量 SFT 标注，再到 DeepSeek-R1 的第二轮约 `800k` 样本、Kimi k1.5 的 `100万 text + 100万 text-vision`。
2. **RL / 偏好优化阶段的数据量，很多时候并不比 SFT 小**。例如 Llama 2 公开了 `1,418,091` 对自建偏好比较；Anthropic Constitutional AI 公开了 `135,296` 条 helpfulness comparisons 和 `182,831` 条 harmlessness comparisons。
3. **公开资料最不透明的通常不是数据量，而是 post-training 算力账单**。大多数团队会公开模型尺寸、阶段设计、部分数据量，但**不会公开 SFT / RL 烧了多少 GPU-hours / TPU-hours**。
4. **后训练路线大体分成两类**：
   - **高质量人工数据路线**：典型是 InstructGPT、Llama 2-Chat；
   - **多阶段“冷启动 SFT → RL → 拒绝采样 / 再 SFT”路线**：典型是 DeepSeek-R1、Qwen3、Kimi k1.5。
5. **大模型负责探索，小模型负责承接 / 蒸馏，已经是公开资料里的常见工业模式**。

---

## 二、口径说明

为了避免不同论文里的口径互相打架，本文统一按下面方式统计：

- **SFT**：instruction tuning / supervised fine-tuning / cold-start finetuning / demonstration data。
- **RL / 偏好优化**：RM training、preference pairs、PPO、RLHF、RLAIF、GRPO、DPO、rejection sampling 等后训练阶段。
- **模型资源**优先记录三类信息：
  1. policy / chat / instruct 模型尺寸；
  2. reward model / judge model / teacher model 的角色与尺寸；
  3. 是否公开了 post-training 的 GPU / TPU / FLOPs / 集群资源。
- **没有公开数字就明确写“未披露”**，不做拍脑袋估算。

---

## 三、总览表

| 模型 | SFT 阶段公开数据 | RL / 偏好优化公开数据 | 参与的模型资源 | post-training 算力公开度 | 主要来源 |
|---|---:|---:|---|---|---|
| **InstructGPT** | 约 `13k training prompts` | `33k` RM prompts + `31k` PPO prompts | policy：1.3B / 6B / 175B；RM：6B | 未披露 | [Ouyang et al., 2022](https://arxiv.org/abs/2203.02155) |
| **Llama 2-Chat** | `27,540` 条 SFT annotations | Meta 自建 `1,418,091` 对 comparisons；RM 总数据 `2,919,326` 对（含开源混合） | 7B / 13B / 70B；70B 用于 rejection sampling；双 reward models | 未披露 | [Touvron et al., 2023](https://arxiv.org/abs/2307.09288) |
| **Constitutional AI / RL-CAI** | `182,831` harmful prompts 驱动 SL-CAI；`135,296` helpfulness prompts | `135,296` helpfulness comparisons + `182,831` harmlessness comparisons | 52B 级 SL-CAI / RL-CAI | 未披露 | [Bai et al., 2022](https://arxiv.org/abs/2212.08073) |
| **Llama 3** | 多轮 SFT，**未披露总 SFT 样本数** | 六轮 post-training；公开了人类偏好数据统计结构，但**未披露总 comparisons 数** | 8B / 70B / 405B；post-training 采用 SFT + RS + DPO | 未披露（仅披露 pretraining FLOPs） | [Llama Team, 2024](https://arxiv.org/abs/2407.21783) |
| **DeepSeek-R1** | 数千条 cold-start data + 第二轮约 `800k` 样本（`600k` reasoning + `200k` non-reasoning） | 两轮 RL；公开了阶段设计，但**未披露总 rollout / comparison 数** | DeepSeek-R1 / R1-Zero：671B total / 37B activated | 未披露 | [DeepSeek-AI, 2025](https://arxiv.org/abs/2501.12948) |
| **Qwen3** | cold-start finetuning，**未披露总样本数** | reasoning RL 使用 `3,995 query-verifier pairs` | 旗舰模型 235B total / 22B activated | 未披露 | [Qwen Team, 2025](https://arxiv.org/abs/2505.09388) |
| **Kimi k1.5** | `100万 text examples` + `100万 text-vision examples` | 使用 RL prompt set + `128k` 上下文 RL，**未披露总 prompt 数** | 论文未公开总参数；公开了长上下文 RL 基础设施设计 | 未披露 | [Kimi Team, 2025](https://arxiv.org/abs/2501.12599) |
| **GPT-4** | 未披露 | 使用 RLHF，但未披露数据量 | 未披露模型尺寸；只说“Transformer-style” | 明确不披露 | [OpenAI, 2023](https://arxiv.org/abs/2303.08774) |
| **Gemini 1.0** | 使用 demonstration data，未披露数据量 | 使用 feedback data + RLHF，未披露数据量 | Gemini Ultra / Pro / Nano；Nano 为 1.8B / 3.25B | 未披露 | [Gemini Team, 2023](https://arxiv.org/abs/2312.11805) |

> 结论很直白：**今天公开得最完整的，仍然是 InstructGPT、Llama 2、Constitutional AI、DeepSeek-R1 这类“论文味儿比较重”的项目。**

---

## 四、逐模型拆解

### 1）InstructGPT：经典 RLHF 三段式基线

**公开数据**

- SFT dataset：约 `13k training prompts`。
- RM dataset：`33k training prompts`。
- PPO dataset：`31k training prompts`。

**公开模型资源**

- policy model：`1.3B`、`6B`、`175B`。
- reward model：只用 `6B RM`，论文明确说明这样更稳定、成本更低。

**要点**

- InstructGPT 不是靠超大 SFT 数据量取胜，而是靠 **SFT → RM → PPO** 这条清晰管线把 base GPT-3 调成“像助手”的模型。
- 论文没有公开后训练 GPU-hours / 集群资源，因此只能比较方法和数据量级，不能精确算成本。

**来源**

- 论文：<https://arxiv.org/abs/2203.02155>
- 重点位置：paper p.5（13k / 33k / 31k）；paper p.2（1.3B / 6B / 175B）

---

### 2）Llama 2-Chat：高质量 SFT + 大规模偏好数据的典型路线

**公开数据**

- SFT：`27,540 annotations`。
- Meta 自建 preference data：`1,418,091` 对 binary comparisons。
- reward model 训练总量（含开源偏好数据）：`2,919,326 comparisons`。

**公开模型资源**

- chat 模型：`7B / 13B / 70B`。
- 采用两个 reward models（helpfulness / safety）。
- `70B` 模型负责 rejection sampling，再把结果蒸馏到小模型。

**要点**

- Meta 在论文里明确说：他们没有继续迷信数百万低质量 SFT 样本，而是收敛到 `27,540` 条高质量人工标注。
- 真正堆大的，是 preference / reward 数据，而不是 SFT 本身。
- post-training 的具体 GPU-hours 没公开；只公开了 pretraining 的 A100 GPU-hours。

**来源**

- 论文：<https://arxiv.org/abs/2307.09288>
- 重点位置：paper p.9（27,540 annotations）；paper p.11（1,418,091 / 2,919,326 comparisons）

---

### 3）Anthropic Constitutional AI / RL-CAI：RLAIF 的工业代表

这个案例的关键点是：**harmlessness 偏好标签大量来自 AI feedback，不是全部靠人工**。

**公开数据（SL / SFT 阶段）**

- red-team harmful prompts：
  - 人写 `42,496` 条；
  - 模型生成 `140,335` 条；
  - 合计 `182,831` 条 harmful prompts。
- helpfulness prompts：`135,296` 条。

**公开数据（RL / preference 阶段）**

- helpfulness comparisons：`135,296`。
- harmlessness comparisons：`182,831`。
- 另有大批 RL prompts 用于 helpfulness / harmlessness 训练。

**公开模型资源**

- 多处实验基于 `52B` 级模型。
- 除 policy model 外，还需要 feedback / preference model 参与评估和打分。

**要点**

- 它证明了另一条路线：**人不一定直接逐条打偏好标签，也可以先给原则（constitution），再用模型扩写 critique / revision / preference。**
- 这是“人定规则，AI 放大监督信号”的代表。
- 同样没有公开 RL 的具体 GPU / TPU 消耗。

**来源**

- 论文：<https://arxiv.org/abs/2212.08073>
- 重点位置：Section 3.2 / 4.2；Figure 2、Figure 3（52B RL runs）

---

### 4）Llama 3：SFT + RS + DPO，公开了方法，没公开总量

**公开数据**

- 论文说明进行了多轮 post-training，**每轮包括 SFT 与 DPO / preference data 更新**。
- 公开了人类偏好数据的结构统计（不同领域占比、平均 turns / tokens 等），并说明**总共做了 six rounds**。
- 但**没有公开总 SFT 样本数和总 comparisons 数**。

**公开模型资源**

- 模型尺寸：`8B / 70B / 405B`。
- post-training 流程是 **SFT + rejection sampling + DPO**，而不是经典的 PPO 为主。
- pretraining 公开了 `3.8 × 10^25 FLOPs`（405B），但这不是 post-training 算力。

**要点**

- Llama 3 的价值在于把 Meta 的后训练路线从“Llama 2 的 RLHF 风格”推进到“更偏 SFT + RS + DPO 的工业稳定路线”。
- 公开信息足够判断方法，但不够精算数据总量和后训练算力。

**来源**

- 论文：<https://arxiv.org/abs/2407.21783>
- 重点位置：paper p.15-16（reward modeling / SFT / DPO / six rounds）；Table 6（preference data statistics）

---

### 5）DeepSeek-R1：RL 比重极高的多阶段后训练路线

**公开数据**

- DeepSeek-R1-Zero：**纯 RL 起步**，不做前置 SFT。
- DeepSeek-R1：
  1. 先做数千条 cold-start data；
  2. 第一轮 reasoning-oriented RL；
  3. 再做一轮约 `800k` SFT：
     - `600k reasoning-related samples`
     - `200k non-reasoning samples`
  4. 最后做第二轮全场景 RL。

**公开模型资源**

- DeepSeek-R1 / R1-Zero：`671B total params`，`37B activated params`。
- 使用 DeepSeek-V3-Base 作为基座。
- 又把推理能力蒸馏到 `1.5B / 7B / 8B / 14B / 32B / 70B` 小模型。

**要点**

- 这是目前公开资料里最典型的“**少量 cold start → 大规模 RL → 整理成更大 SFT 数据 → 再 RL**”路线。
- DeepSeek 公布了阶段设计和数据量级，但**没有公开 RL rollout 总量、GPU-hours、训练时长**。

**来源**

- 论文：<https://arxiv.org/abs/2501.12948>
- 重点位置：paper p.26（800K supervised data）；paper p.41（671B / 37B）

---

### 6）Qwen3：小规模可验证 query-verifier pairs 驱动 reasoning RL

**公开数据**

- cold-start finetuning 存在，但论文强调应尽量减少 preparatory phase 的样本数和步数。
- reasoning RL 阶段，明确公开：`3,995 query-verifier pairs`。
- 通用 SFT / general-domain RL 使用了多阶段方案，但未公布总样本量。

**公开模型资源**

- 旗舰模型：`235B total params`，`22B activated params`。
- reasoning RL 采用 GRPO 风格的可验证任务路线。

**要点**

- Qwen3 的公开信息告诉我们：在 reasoning RL 里，**不一定要海量人工偏好对**，也可以用几千条高质量、可验证的 query-verifier pairs 启动一阶段强化学习。
- 但总的 post-training 数据量和算力账单仍然没有公开。

**来源**

- 技术报告：<https://arxiv.org/abs/2505.09388>
- 重点位置：paper p.10（3,995 query-verifier pairs）；paper p.1-2（235B / 22B）

---

### 7）Kimi k1.5：长上下文 RL 路线代表

**公开数据**

- vanilla SFT dataset：约 `100万 text examples`。
  - `500k` general QA
  - `200k` coding
  - `200k` math / science
  - `5k` creative writing
  - `20k` long-context tasks
- 另有 `100万 text-vision examples`。
- RL 阶段公开了 prompt set 设计与基础设施，但**未公开总 prompt 数**。

**公开模型资源**

- 论文重点公开的是 RL 基础设施和 `128k` context RL，而不是总参数规模。
- 强调 partial rollout、长上下文 RL、共享训练 / 推理 GPU 资源的系统设计。

**要点**

- Kimi k1.5 是“**后训练不只看算法，还看系统工程**”的典型案例。
- 它公开了 SFT 的百万级数据量和 RL 系统优化方向，但依旧没有公开完整后训练算力账单。

**来源**

- 技术报告：<https://arxiv.org/abs/2501.12599>
- 重点位置：paper p.2（128k RL）；paper p.8（100万 text + 100万 text-vision）

---

### 8）GPT-4 与 Gemini：方法公开，数据与算力不公开

#### GPT-4

- 技术报告明确写到：模型经过 RLHF 对齐。
- 同时又明确写到：**不披露 architecture（含 model size）、hardware、training compute、dataset construction、training method 等更细节的信息。**

**来源**

- 技术报告：<https://arxiv.org/abs/2303.08774>
- 重点位置：paper p.2

#### Gemini 1.0

- 明确公开了 post-training 的流程：
  1. prompt data collection
  2. SFT on demonstration data
  3. RM training on feedback data
  4. RLHF
- 但**没有公开 demonstration data / feedback data 的具体条数**。
- Nano 模型尺寸公开为 `1.8B` 和 `3.25B`，但这不是 post-training 数据量披露。

**来源**

- 技术报告：<https://arxiv.org/abs/2312.11805>
- 重点位置：paper p.20-21

---

## 五、横向观察

### 1）SFT 不是越大越好，位置和质量更关键

- InstructGPT：约 `13k` prompts 就能把模型往“助手”方向拧过去。
- Llama 2-Chat：明确说比起海量低质数据，更看重 `27,540` 条高质量人工标注。
- DeepSeek-R1 / Kimi k1.5：则在更成熟的时代，把 SFT 放进多阶段大管线里，规模可到几十万甚至百万。

### 2）RL / preference data 在很多项目里是“真正的大头”

- Llama 2：`1,418,091` 对自建 comparisons。
- Anthropic：`135,296` + `182,831` comparisons。
- DeepSeek-R1：虽然没公开总 comparison / rollout 数，但从两轮 RL + 800k 再整理数据的设计看，RL 已经不是“最后修修边角”，而是核心能力形成机制之一。

### 3）真正最不透明的，是 post-training 算力

公开资料通常只够你看出：

- 大致用了多少数据；
- 上了多大的 policy / RM / teacher；
- 走的是 PPO、DPO、RLAIF、GRPO 还是 rejection sampling；

但通常**不够你精确算出后训练到底花了多少 GPU / TPU 钱**。

### 4）大模型探索、小模型承接，已经是共识

- Llama 2：70B 负责 rejection sampling，小模型吃蒸馏结果。
- DeepSeek-R1：大模型先通过 RL 形成推理模式，再蒸馏到 1.5B–70B 小模型。

这背后其实就是一个很现实的资源分工：**最贵的探索留给最大模型，便宜的部署留给小模型。**

---

## 六、最终判断

如果只看公开资料，比较稳妥的结论是：

1. **SFT 数据量：部分能精确比。**
2. **偏好 / RL 数据量：有些项目公开得很细，有些只给方法不给总量。**
3. **模型资源：通常能知道用了多大的 policy / RM / teacher。**
4. **真实后训练算力账单：大多数时候没法精确比。**

所以，今天公开资料最适合做的是：

- **路线分析**；
- **量级比较**；
- **透明度比较**；

而不是做看上去很像真的、其实缺乏依据的 post-training 成本审计。

---

## 七、参考来源

1. Ouyang et al. (2022), *Training language models to follow instructions with human feedback*  
   <https://arxiv.org/abs/2203.02155>
2. Touvron et al. (2023), *Llama 2: Open Foundation and Fine-Tuned Chat Models*  
   <https://arxiv.org/abs/2307.09288>
3. Bai et al. (2022), *Constitutional AI: Harmlessness from AI Feedback*  
   <https://arxiv.org/abs/2212.08073>
4. Llama Team (2024), *The Llama 3 Herd of Models*  
   <https://arxiv.org/abs/2407.21783>
5. DeepSeek-AI (2025), *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*  
   <https://arxiv.org/abs/2501.12948>
6. Kimi Team (2025), *Kimi k1.5: Scaling Reinforcement Learning with LLMs*  
   <https://arxiv.org/abs/2501.12599>
7. Qwen Team (2025), *Qwen3 Technical Report*  
   <https://arxiv.org/abs/2505.09388>
8. OpenAI (2023), *GPT-4 Technical Report*  
   <https://arxiv.org/abs/2303.08774>
9. Gemini Team (2023), *Gemini: A Family of Highly Capable Multimodal Models*  
   <https://arxiv.org/abs/2312.11805>

---

*最后更新：2026-03-27*