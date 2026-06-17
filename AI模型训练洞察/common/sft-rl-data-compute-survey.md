# 业界模型 SFT 与 RL 阶段的数据与资源规模调研

## 概述

本报告系统梳理了主流大语言模型在监督微调（SFT）和强化学习（RL）阶段使用的数据规模与计算资源。通过分析公开论文与技术报告，我们发现：

1. **SFT 阶段**：数据规模从数千到数百万不等，质量比数量更关键
2. **RL 阶段**：提示（prompt）数量通常远少于 SFT，但需要大量采样与迭代
3. **透明度差异**：开源模型披露较详细，闭源模型（GPT-4、Claude、Gemini）仅公开方法论
4. **长 CoT 趋势**：推理模型（DeepSeek-R1、Kimi k1.5、QwQ）倾向于用少量高质量 CoT 数据冷启动，再通过 RL 扩展

---

## 一、代表性模型数据规模对比

### 表 1：公开数据规模的模型

| 模型 | SFT 阶段公开数据 | RL/偏好数据阶段 | 模型规模 | 计算资源披露 | 备注 | 来源 |
|------|-----------------|----------------|---------|------------|------|------|
| **InstructGPT** | ~13,000 prompts | RM: ~33,000 comparisons<br>PPO: ~31,000 prompts | 1.3B / 6B / 175B | 未披露 | 开创性 RLHF 工作 | [Ouyang et al., 2022](https://arxiv.org/abs/2203.02155) p.5 |
| **Llama 2-Chat** | 27,540 annotations | 1.4M+ preference pairs<br>(含开源数据) | 7B / 13B / 70B | 未披露具体 GPU 时 | 迭代式 RLHF | [Touvron et al., 2023](https://arxiv.org/abs/2307.09288) p.9, p.11 |
| **Llama 3** | 多轮迭代（未公开具体数量） | 人类偏好数据（Table 6）<br>6 轮迭代 | 8B / 70B / 405B | 405B 用 3.8×10²⁵ FLOPs 预训练 | SFT+RS+DPO 流程 | [Llama Team, 2024](https://arxiv.org/abs/2407.21783) p.16 |
| **Constitutional AI** | 数千条（few thousand） | AI 生成偏好对<br>（无人类标注） | 52B | 未披露 | 纯 AI 反馈 | [Bai et al., 2022](https://arxiv.org/abs/2212.08073) |
| **DeepSeek-R1** | 冷启动：数千条 long-CoT<br>第二阶段 SFT：~800K | RL 阶段：规则奖励<br>（数学/代码验证器） | 671B 总参数<br>37B 激活 | 未披露 | 两阶段 RL + SFT | [DeepSeek-AI, 2025](https://arxiv.org/abs/2501.12948) p.26 |
| **Kimi k1.5** | 1M text + 1M text-vision | RL prompt set（未公开数量）<br>128K 上下文 RL | 未公开总参数 | 未披露 GPU 数 | 长上下文 RL | [Kimi Team, 2025](https://arxiv.org/abs/2501.12599) p.8 |
| **Qwen3** | 冷启动：最小化样本<br>通用 SFT：未公开 | 推理 RL：3,995 query-verifier pairs | 235B 总参数<br>22B 激活（旗舰 MoE） | 未披露 | 思维模式 + 非思维模式 | [Qwen Team, 2025](https://arxiv.org/abs/2505.09388) p.10 |

### 表 2：未公开数据规模的闭源模型

| 模型 | 公开信息 | 来源 |
|------|---------|------|
| **GPT-4** | 使用 RLHF，但"不披露架构、硬件、训练计算、数据集构建"等细节 | [OpenAI, 2023](https://arxiv.org/abs/2303.08774) p.2 |
| **Gemini** | 使用 SFT + RLHF，人类标注者提供偏好反馈，但未公开数据量 | [Gemini Team, 2023](https://arxiv.org/abs/2312.11805) p.21 |
| **Claude 3** | 基于 Constitutional AI 方法，但 Anthropic 未公开 Claude 3 具体训练数据 | Anthropic 官方博客 |

---

## 二、关键发现

### 2.1 SFT 阶段特点

1. **数据量级差异大**：
   - 早期模型（InstructGPT）：~13K prompts
   - 中期模型（Llama 2）：~27K annotations
   - 多模态/长上下文模型（Kimi k1.5）：1M+ examples
   - 推理模型冷启动：数千条高质量 long-CoT

2. **质量 > 数量**：
   - Llama 2 报告指出"数万条高质量标注即可达到良好效果"
   - DeepSeek-R1 用数千条冷启动数据即可让 RL 收敛更快

3. **合成数据崛起**：
   - Qwen3、Kimi k1.5 使用模型生成 + 拒绝采样扩充 SFT 数据
   - DeepSeek-R1 第二阶段用 RL checkpoint 做拒绝采样生成 600K 推理样本

### 2.2 RL 阶段特点

1. **Prompt 数量远少于 SFT**：
   - InstructGPT PPO：31K prompts（vs. 33K RM training pairs）
   - Qwen3 推理 RL：3,995 query-verifier pairs
   - 原因：RL 通过多次采样同一 prompt 生成大量轨迹

2. **偏好数据规模**：
   - Llama 2：1.4M+ binary comparisons（含开源数据集）
   - Llama 3：多轮迭代，每轮收集新偏好标注（Table 6 显示统计信息）
   - Constitutional AI：完全用 AI 生成偏好对，无人类标注

3. **奖励建模方式**：
   - 人类偏好：Llama 系列、InstructGPT
   - 规则验证器：DeepSeek-R1（数学/代码正确性）
   - AI 反馈：Constitutional AI、部分 Qwen3 数据

### 2.3 计算资源披露情况

**几乎所有模型都未公开 SFT/RL 阶段的具体计算资源**，仅有：

- Llama 3 405B：预训练用 3.8×10²⁵ FLOPs（但后训练未披露）
- GPT-4：明确声明"不披露硬件、训练计算"
- 推测原因：竞争考量 + 后训练成本相对预训练较小

---

## 三、趋势观察

### 3.1 长 CoT 推理模型的新范式

以 DeepSeek-R1、Kimi k1.5、QwQ-32B 为代表的推理模型采用：

1. **冷启动 SFT**：数千条高质量 long-CoT 示例
2. **大规模 RL**：在数学/代码等可验证任务上用规则奖励训练
3. **拒绝采样 SFT**：用 RL checkpoint 生成新数据再做监督学习
4. **通用 RL**：最后阶段加入非推理任务的人类偏好数据

这种"少量 SFT → 大规模 RL → 再 SFT"的迭代模式成为新趋势。

### 3.2 开源 vs. 闭源透明度

| 维度 | 开源模型 | 闭源模型 |
|------|---------|---------|
| 数据规模 | 通常公开（如 Llama、DeepSeek） | 不公开（GPT-4、Claude、Gemini） |
| 方法论 | 详细描述 | 高层描述 |
| 模型权重 | 公开 | 不公开 |
| 计算资源 | 部分公开（预训练 FLOPs） | 不公开 |

### 3.3 合成数据与蒸馏

- **自我改进**：模型生成数据 → 拒绝采样 → 再训练自己
- **蒸馏**：DeepSeek-R1 蒸馏到 Qwen/Llama 小模型，用 800K 样本
- **多模态扩展**：Kimi k1.5 用 1M text-vision 样本支持视觉推理

---

## 四、数据来源与引用

所有数据均来自公开的学术论文或官方技术报告：

1. **InstructGPT**: Ouyang et al. (2022). *Training language models to follow instructions with human feedback*. NeurIPS. [arXiv:2203.02155](https://arxiv.org/abs/2203.02155)

2. **Llama 2**: Touvron et al. (2023). *Llama 2: Open Foundation and Fine-Tuned Chat Models*. [arXiv:2307.09288](https://arxiv.org/abs/2307.09288)

3. **Llama 3**: Llama Team (2024). *The Llama 3 Herd of Models*. [arXiv:2407.21783](https://arxiv.org/abs/2407.21783)

4. **Constitutional AI**: Bai et al. (2022). *Constitutional AI: Harmlessness from AI Feedback*. [arXiv:2212.08073](https://arxiv.org/abs/2212.08073)

5. **DeepSeek-R1**: DeepSeek-AI (2025). *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*. [arXiv:2501.12948](https://arxiv.org/abs/2501.12948)

6. **Kimi k1.5**: Kimi Team (2025). *Kimi k1.5: Scaling Reinforcement Learning with LLMs*. [arXiv:2501.12599](https://arxiv.org/abs/2501.12599)

7. **Qwen3**: Qwen Team (2025). *Qwen3 Technical Report*. [arXiv:2505.09388](https://arxiv.org/abs/2505.09388)

8. **GPT-4**: OpenAI (2023). *GPT-4 Technical Report*. [arXiv:2303.08774](https://arxiv.org/abs/2303.08774)

9. **Gemini**: Gemini Team (2023). *Gemini: A Family of Highly Capable Multimodal Models*. [arXiv:2312.11805](https://arxiv.org/abs/2312.11805)

---

## 五、总结

1. **SFT 数据规模**：从数千到百万级，推理模型倾向于少量高质量冷启动
2. **RL 数据规模**：Prompt 数量通常 < 10 万，但通过多次采样生成大量轨迹
3. **计算资源**：几乎无模型公开后训练的具体 GPU 时或 FLOPs
4. **透明度**：开源模型（Llama、DeepSeek、Qwen）披露较详细，闭源模型仅公开方法论
5. **新趋势**：长 CoT + RL + 合成数据 + 蒸馏成为推理模型的标准流程

---

*最后更新：2026-03-27*  
*数据来源：公开学术论文与技术报告*
