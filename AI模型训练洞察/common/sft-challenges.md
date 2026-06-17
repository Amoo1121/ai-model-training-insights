# 业界训练 SFT 的主要难点：从数据、目标错配到后续 RL 耦合

> 本文聚焦大模型 **post-training** 阶段的 SFT（Supervised Fine-Tuning，监督微调），讨论当前业界在训练 SFT 时最常见、也最棘手的问题。重点不在“怎么把 loss 训下去”，而在：**什么数据值得训、训出来是否能泛化、以及会不会把后续 RL / RLVR 的探索空间训坏。**
>
> 说明：本文优先采用论文、技术报告、模型卡中的公开信息。部分结论来自 2025–2026 的最新预印本，可视为前沿信号，而非所有团队都已达成的最终共识。

---

## 一、先说结论

如果把现在业界训练 SFT 的难点压成一句话，就是：

**SFT 最大的问题已经不是“能不能训”，而是“训什么、训到什么程度、以及怎么别把后面的 RL 训废”。**

更具体地说，当前的核心难点主要集中在 6 个方面：

1. **高质量 SFT 数据稀缺，尤其是 long-CoT / agent / 复杂指令数据**
2. **训练数据是否适配目标模型，比“老师模型够不够强”更重要**
3. **SFT 阶段分数更高，不代表后续 RL 结果更好**
4. **SFT 容易过拟合静态 expert data，压缩后续探索空间**
5. **SFT 与 RL 的优化目标天然存在错配，二者难以彻底解耦**
6. **现代模型要同时兼顾 reasoning / non-reasoning / multilingual / long-context / tool-use，数据配方越来越难调**

---

## 二、总览表：现在 SFT 最难的点到底是什么

| 难点 | 具体表现 | 为什么现在更严重 | 公开证据 / 代表来源 | 业界解法趋势 |
|---|---|---|---|---|
| **高质量数据稀缺** | 好的 SFT 样本难收集、难验证、成本高 | 现在需要长 CoT、复杂 agent、长上下文、多语言，不再是普通问答 | [Qwen3](https://arxiv.org/abs/2505.09388), [Data Repetition Beats Data Scaling in Long-CoT SFT](https://arxiv.org/abs/2602.11149) | 小规模高质量冷启动、蒸馏、验证器过滤、少样本多轮训练 |
| **数据与目标模型不匹配** | 最强 teacher 的回答不一定最适合当前 base model | synthetic data 越来越多，分布偏移更容易积累 | [The Best Instruction-Tuning Data are Those That Fit](https://arxiv.org/abs/2502.04194) | 模型特定数据选择、target-model scoring、分布匹配 |
| **SFT 指标误导** | SFT loss / benchmark 变好，但后续 RL 未必更好，甚至更差 | 训练链路从单阶段变成 SFT → RLVR，多阶段目标不一致 | [Quagmires in SFT-RL Post-Training](https://arxiv.org/abs/2510.01624), [Good SFT Optimizes for SFT, Better SFT Prepares for RL](https://arxiv.org/abs/2602.01058) | 改用“预判 RL 效果”的指标，如 held-out generalization、Pass@large-k |
| **过拟合 expert data / 压缩探索** | 模型学会模仿，但不再愿意探索新推理轨迹 | reasoning 模型更依赖 RLVR，需要保留探索空间 | [On-Policy RL Meets Off-Policy Experts](https://arxiv.org/abs/2508.11408), [GIFT](https://arxiv.org/abs/2601.09233) | exploration-aware SFT、动态加权、有限温度初始化 |
| **SFT 与 RL 目标错配** | SFT 想最像专家，RL 想拿最高 reward，彼此会互相伤害 | 现代 post-training 越来越依赖交替 SFT/RL | [On the Non-decoupling of SFT and RL](https://arxiv.org/abs/2601.07389) | 联合视角设计 SFT，不把 SFT 当独立收尾步骤 |
| **多目标数据配方复杂** | reasoning / chat / tool-use / multilingual / long-context 互相打架 | 模型不再只做聊天，而是“全家桶” | [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388) | 多阶段 post-training、mixture scheduling、分阶段融合 |

---

## 三、难点一：高质量 SFT 数据本身就稀缺

### 3.1 难点本质

SFT 的第一大瓶颈，仍然是 **高质量数据不够，尤其是不够“贵得值得”**。

原因很简单：

- 普通 instruction-following 数据还能批量造；
- 但 **long-CoT、数学、代码、agent、多轮工具调用、长上下文** 数据，采集和校验成本要高得多；
- 这些数据不只是要“答案对”，还要：
  - 推理链可读
  - 格式稳定
  - 中间步骤靠谱
  - 长度合适
  - 能服务于后续 RL / verifier

### 3.2 为什么这个问题越来越严重

因为现在的 reasoning 模型，不再满足于“会答题”，而是要求：

- 能长链推理
- 能多轮决策
- 能控制思考预算
- 能兼容 think / no-think 两种模式

这意味着 SFT 数据必须更精细，而不是更多垃圾样本硬堆。

### 3.3 公开证据

**Qwen3** 明确写到，它的 post-training 采用多阶段流程，前两阶段围绕 long-CoT cold-start finetuning 与数学/代码强化学习展开，后两阶段再把 reasoning 与 non-reasoning 数据统一起来继续训练。这实际上说明：

> 高质量 cold-start SFT 数据依然是关键，但样本和训练步数都要精打细算。

来源：
- Qwen3 Technical Report: <https://arxiv.org/abs/2505.09388>

另一篇更激进的研究甚至发现：

> 在 long-CoT SFT 中，固定 update budget 下，**小数据集多轮重复训练** 可能优于 **大数据集单轮训练**。

这说明问题不只是“数据少”，而是：

> **真正能让模型学会推理结构的数据，其利用方式比数据总量更重要。**

来源：
- Data Repetition Beats Data Scaling in Long-CoT Supervised Fine-Tuning: <https://arxiv.org/abs/2602.11149>

### 3.4 业界解法趋势

- 少量高质量 **cold-start** 数据先定型
- 用蒸馏和验证器降低人工成本
- 用 token accuracy / 训练饱和度，而不是盲目扩数据，决定何时停
- 把 SFT 数据从“越多越好”改成“越对越好”

---

## 四、难点二：数据是否适配目标模型，比 teacher 是否最强更重要

### 4.1 难点本质

业界以前一个很常见的默认思路是：

> 拿更强的 teacher 模型生成答案，然后用这些答案去训较弱的 base model。

但问题在于：

- teacher 的回答风格、推理分布、长度、token 偏好，可能和目标模型严重不匹配；
- 数据量一大，这种 mismatch 会被放大；
- 结果可能不是“蒸馏出能力”，而是把目标模型原本的分布打乱。

### 4.2 公开证据

**The Best Instruction-Tuning Data are Those That Fit** 这篇论文的核心结论很直白：

> 最好的 instruction-tuning 数据，不一定来自最强模型，而是来自**最适配当前目标模型分布**的回答。

它指出，来自其他 LLM 的 response 往往对目标模型来说是 out-of-distribution；当这种数据大量进入 SFT 时，会带来：

- diminishing returns（收益递减）
- robustness 下降
- 数据分布扭曲

论文提出的 GRAPE 方案，会为每条 instruction 从多个候选回答中，选出 **目标模型本身概率最高、最“fit” 的那个回答**。结果显示，这种方法能明显优于“直接用最强 teacher 的答案”。

来源：
- The Best Instruction-Tuning Data are Those That Fit: <https://arxiv.org/abs/2502.04194>

### 4.3 业界解法趋势

- 不再迷信“最强 teacher = 最佳 SFT label”
- 用 target model 自身打分做 response selection
- 优先选择 **distribution-compatible** 的回答
- 把数据选择从“挑最强答案”变成“挑最合适答案”

---

## 五、难点三：SFT 阶段分数更高，不代表后续 RL 更好

### 5.1 难点本质

这是现在 reasoning 模型训练里最容易把团队带沟里的坑：

> **SFT 阶段评测分数高，不等于这个 checkpoint 是最好的 RL 初始化点。**

如果团队只盯：
- training loss
- validation loss
- post-SFT benchmark

很可能会选出一个“看起来最好、实际上最不适合进 RL”的 checkpoint。

### 5.2 公开证据 1：高 SFT 分数可能误导

**Quagmires in SFT-RL Post-Training** 直接挑战了这个常见假设。

论文发现：

- 高 SFT 分数经常偏向 **更简单、更同质化** 的数据；
- 这种 checkpoint 未必在后续 RL 中收益更高；
- 有些情况下，SFT 做得更“好”，反而会让 RL 最终结果更差。

它还给出一些很有意思的现象：

- 同样的 SFT budget 下，只训短样本，SFT 分数可能更好；
- 但后续 RL 效果，往往不如训长度更丰富的数据；
- 训练更多 epoch 也可能抬高 SFT 表现，却损伤后续 RL 的 headroom。

来源：
- Quagmires in SFT-RL Post-Training: <https://arxiv.org/abs/2510.01624>

### 5.3 公开证据 2：更强 SFT checkpoint 经过同样 RL，可能输给更弱的

**Good SFT Optimizes for SFT, Better SFT Prepares for Reinforcement Learning** 进一步指出：

> 离线 SFT 阶段更强的 checkpoint，在经过相同 RL 后，可能显著输给离线阶段更弱的 checkpoint。

论文认为根源在于：

- offline SFT 学的是 behavior policy 的数据分布；
- online RL 学的是模型自己 rollout 出来的分布；
- 两者之间有明显 distribution mismatch。

因此，一个“很会做 SFT 题”的模型，并不一定是“很会被 RL 继续推高”的模型。

来源：
- Good SFT Optimizes for SFT, Better SFT Prepares for Reinforcement Learning: <https://arxiv.org/abs/2602.01058>

### 5.4 业界解法趋势

- 不再只看 post-SFT 分数选 checkpoint
- 开始关注：
  - held-out generalization loss
  - Pass@large-k
  - RL 前后的 headroom
  - SFT 对 rollout 分布的适配程度
- 用“更适合 RL 的 SFT”替代“单纯更强的 SFT”

---

## 六、难点四：SFT 很容易过拟合 expert data，并压缩后续探索空间

### 6.1 难点本质

SFT 本质是离线模仿：

- 数据是静态的
- response 是给定的
- 模型优化目标是“尽量像专家”

这会天然带来两个问题：

1. **过拟合 expert style**：模型越来越像标注数据，而不一定更会解决新问题；
2. **探索空间被压缩**：模型在后续 RL 中更不愿意尝试非目标 token，导致探索困难。

### 6.2 公开证据 1：SFT 可能带来“shift-readapt-overfit”

**On-Policy RL Meets Off-Policy Experts** 提出一个很有代表性的观察：

把 off-policy 的 expert data 生硬塞给已有 policy 的模型，常出现一个三阶段现象：

- **shift**：模型原有能力先被打乱
- **readapt**：慢慢适应 expert 数据
- **overfit**：最后对 expert data 过拟合

这篇论文把问题说得很透：

> expert data 能带来新能力，但也可能破坏已有 response pattern，并诱发 overfitting。

来源：
- On-Policy RL Meets Off-Policy Experts: <https://arxiv.org/abs/2508.11408>

### 6.3 公开证据 2：rigid SFT 会诱发 distributional collapse

**GIFT** 这篇论文更进一步，从理论角度说：

> 标准 SFT 的刚性监督会诱发 distributional collapse，从而耗尽后续 RL 所需的探索空间。

它把标准 SFT 看成一种“零温度极限”，会把概率质量硬压到 expert token 上；而 RL 恰恰需要一个还有足够熵、还能探索高回报轨迹的分布。

来源：
- GIFT: Reconciling Post-Training Objectives via Finite-Temperature Gibbs Initialization: <https://arxiv.org/abs/2601.09233>

### 6.4 业界解法趋势

- exploration-aware SFT
- token / block / sequence 级别 loss reweighting
- 有限温度初始化，保留 base model prior
- 把 SFT 看成 RL 的 initialization，而不是独立终点

---

## 七、难点五：SFT 与 RL 的优化目标天然错配，二者很难解耦

### 7.1 难点本质

SFT 和 RL 的目标函数本来就不一样：

- **SFT**：最小化 expert token 的交叉熵
- **RL**：最大化 reward / verifier score

直白点说：

> SFT 想让模型“像老师”，RL 想让模型“拿高分”。

很多时候，这俩不是一回事。

### 7.2 公开证据

**On the Non-decoupling of Supervised Fine-tuning and Reinforcement Learning in Post-training** 的结论非常硬：

- 在 **SFT → RL** 中，RL 会提高 SFT loss；
- 在 **RL → SFT** 中，SFT 会降低 RL reward；
- 因而 SFT 和 RL 不能被看成两个互不影响、可以随便拼接的积木块。

换句话说：

> 现代 post-training 里，SFT 和 RL 不是简单串联，而是相互耦合的联合优化问题。

来源：
- On the Non-decoupling of Supervised Fine-tuning and Reinforcement Learning in Post-training: <https://arxiv.org/abs/2601.07389>

### 7.3 业界解法趋势

- 联合设计 SFT 和 RL，而不是各训各的
- 用 hybrid / unified 视角看待 post-training
- 让 SFT 目标显式为后续 RL 服务

补充综述：
- Supervised Fine-Tuning versus Reinforcement Learning: A Study of Post-Training Methods for Large Language Models: <https://arxiv.org/abs/2603.13985>

---

## 八、难点六：现代 SFT 要同时兼顾太多目标，数据配方越来越难调

### 8.1 难点本质

早期 SFT 更多是：

- instruction-following
- chat style
- 格式控制

现在不是了。一个主流模型往往同时要兼顾：

- reasoning / non-reasoning
- think / no-think
- coding / math / general chat
- multilingual
- long-context
- tool use / agent behavior

这使得 SFT 不只是“喂点样本”，而是一个 **复杂的数据混配与阶段调度问题**。

### 8.2 公开证据

**Qwen3** 技术报告非常有代表性：

- 前两阶段重点做 reasoning 能力：long-CoT cold-start + 数学/代码 RL
- 后两阶段把 reasoning 和 non-reasoning 数据统一起来，再做进一步训练
- 同时还要服务于 think mode / non-thinking mode 的统一框架

这说明当代 SFT 的难点，已经从“有没有数据”升级成：

> **多种能力的数据该怎么混、先训什么后训什么、哪些能力要隔离、哪些能力可以融合。**

来源：
- Qwen3 Technical Report: <https://arxiv.org/abs/2505.09388>

### 8.3 业界解法趋势

- multi-stage post-training
- 不同能力拆阶段处理
- reasoning 与非 reasoning 数据分阶段融合
- distillation + RL + SFT 混合配方

---

## 九、一个更实用的判断：今天业界真正卡在哪

如果只看当前公开资料，我会把业界 SFT 的核心难点归纳成下面三条：

### 9.1 不是“数据不够多”，而是“真正有用的数据太贵、太难挑”

SFT 已经进入“数据工程”阶段：

- 你要挑哪类题
- 你要不要保留长链路
- 你选哪个 teacher
- 你要不要按目标模型重新筛 response

现在决定上限的，往往不是 optimizer，而是数据 recipe。

### 9.2 不是“把 SFT 做到最好”，而是“把 SFT 做成更好的 RL 起点”

这可能是今天最重要的观念转变：

> **好 SFT != 好 post-training**

真正重要的是：

- SFT 后模型还能不能探索
- 还能不能从 RL 中再涨
- 会不会因为过拟合而把 headroom 吃光

### 9.3 不是“单一能力对齐”，而是“多目标统一配方”

现代模型不只是聊天机器人，还要：

- 会思考
- 会快答
- 会写代码
- 会调工具
- 会多语言
- 会长上下文

所以 SFT 难点已经从“训练技巧问题”，变成“能力编排问题”。

---

## 十、当前最值得关注的业界解法方向

结合公开资料，当前比较值得盯的 SFT 方向有：

1. **model-specific data selection**  
   代表：GRAPE 这类让 SFT 数据更贴近目标模型分布的方法。

2. **exploration-aware SFT**  
   代表：PEAR、GIFT 这类明确为了后续 RL 初始化而改造 SFT 的方法。

3. **joint / hybrid post-training**  
   代表：SFT 和 RL 不再硬分阶段，而是通过动态加权、交替优化、统一目标来协调。

4. **small high-quality cold-start + more RL later**  
   代表：Qwen3、DeepSeek-R1 一类 reasoning 路线。

5. **better checkpoint selection metrics**  
   不再只看 post-SFT 分数，而要看能否预测 post-RL 效果。

---

## 参考来源

### 官方技术报告 / 模型报告

1. **Qwen Team (2025)**, *Qwen3 Technical Report*  
   <https://arxiv.org/abs/2505.09388>

### 研究论文 / 预印本

2. **Kang et al. (2025)**, *Quagmires in SFT-RL Post-Training: When High SFT Scores Mislead and What to Use Instead*  
   <https://arxiv.org/abs/2510.01624>

3. **Zhang et al. (2025/2026)**, *On-Policy RL Meets Off-Policy Experts: Harmonizing Supervised Fine-Tuning and Reinforcement Learning via Dynamic Weighting*  
   <https://arxiv.org/abs/2508.11408>

4. **Niu et al. (2026)**, *On the Non-decoupling of Supervised Fine-tuning and Reinforcement Learning in Post-training*  
   <https://arxiv.org/abs/2601.07389>

5. **Kopiczko et al. (2026)**, *Data Repetition Beats Data Scaling in Long-CoT Supervised Fine-Tuning*  
   <https://arxiv.org/abs/2602.11149>

6. **Zhang et al. (2026)**, *Good SFT Optimizes for SFT, Better SFT Prepares for Reinforcement Learning*  
   <https://arxiv.org/abs/2602.01058>

7. **Zhao et al. (2026)**, *GIFT: Reconciling Post-Training Objectives via Finite-Temperature Gibbs Initialization*  
   <https://arxiv.org/abs/2601.09233>

8. **Zhang, Dai, Peng (2025)**, *The Best Instruction-Tuning Data are Those That Fit*  
   <https://arxiv.org/abs/2502.04194>

9. **Jiang et al. (2026)**, *Supervised Fine-Tuning versus Reinforcement Learning: A Study of Post-Training Methods for Large Language Models*  
   <https://arxiv.org/abs/2603.13985>

---

*最后更新：2026-03-27*