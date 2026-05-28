---
layout: post
title: "LLM 训练参数调节指南（CPT / SFT / DPO）"
date: 2026-05-14 00:00:00 +0800
categories: [llm, machine-learning]
---

> 整理自技术讨论，涵盖通用超参数、各阶段专有参数、高效训练参数，以及 Batch Size 对梯度质量与训练稳定性的深度分析。


## 一、通用超参数（都需要关注，但推荐值因阶段而异）

### 学习率（Learning Rate）

最关键的超参数。LLM 微调一般用 `1e-5` 到 `5e-5`，CPT 可以稍大（`1e-4`）。通常配合 warmup + cosine decay 调度器使用。

### Batch Size

影响梯度质量和训练稳定性。实际训练中通过 gradient accumulation 来模拟大 batch，等效 batch 一般在 128~512 之间。太小容易梯度噪声大、训练不稳定；太大则噪声温度过低，倾向于收敛到尖锐极小值，导致泛化性能下降（详见第三节）。

### 训练轮数（Epochs）

- SFT：通常 1~3 轮，多了容易过拟合
- CPT：数据量大可以只过一遍
- DPO：一般 1~3 轮

### Warmup Steps / Ratio

一般对前 1%~5% 的步数做 warmup，防止初期学习率过大破坏预训练权重。

### Weight Decay

没有统一常数。预训练常见到 `0.1`；全参 SFT 常见 `0.01` 左右；LoRA / DPO 往往更小，很多配置会设成 `0` 到 `0.01`。

### 梯度裁剪（Gradient Clipping）

`max_norm` 一般设 `1.0`，防止梯度爆炸。

### 序列长度（Max Sequence Length）

越长越贵，按任务需求设，不要无脑拉满。


## 二、各阶段专有参数

### SFT 专有参数

**数据配比（Data Mixture Ratio）**
如果有多个数据源，各类数据的比例影响极大，往往比调学习率更重要。

**Loss Masking**
通常只对 assistant 回复部分计算 loss，而不对 prompt 部分计算。这个开关要确认已打开。

**Packing vs Padding**
把多条短样本打包进一个序列（packing）可以提升 GPU 利用率，但要注意 attention mask 隔离，防止跨样本信息泄漏。


### DPO 专有参数

**β（Beta）**
DPO 最核心的参数，控制偏离参考模型的惩罚力度。

$$\mathcal{L}_{\text{DPO}} = -\mathbb{E} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_\text{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_\text{ref}(y_l|x)} \right) \right]$$

- 通常取值范围：`0.01` 到 `0.5`
- β 越大 → 越保守（离 ref model 越近）
- β 越小 → 越激进，可能出现 reward hacking
- 推荐从 `0.1` 开始调，观察 chosen/rejected reward margin 是否在扩大

**Reference Model**
一般用 SFT 后的模型作为 ref，直接用 base model 效果通常较差。

**Chosen/Rejected 数据质量**
严格来说不是超参数，但数据对的质量差异（margin）比数量更重要。对的差异太小，模型学不到东西。

**SimPO / IPO 变体参数**
- SimPO：有 γ（margin）参数
- IPO：有自己的正则系数


## 三、高效训练参数（LoRA / QLoRA）

| 参数 | 说明 | 常用值 |
|------|------|--------|
| `lora_r` | LoRA 秩，越大容量越强但越贵 | 8 / 16 / 64 |
| `lora_alpha` | 缩放系数，常设为 r 的 2 倍 | 16 / 32 |
| `lora_dropout` | 防过拟合 | 0.05 ~ 0.1 |
| `target_modules` | 应用 LoRA 的层（q/k/v/o/mlp） | 按任务选 |
| 量化精度 | QLoRA 用 4-bit/8-bit | nf4 推荐 |


## 四、调参优先级与策略

**优先级排序：**

$$\text{数据质量} > \text{数据量} > \text{学习率} > \beta_{\text{DPO}} > \text{batch size} > \text{其他}$$

**学习率怎么找：**
先跑 loss curve。loss 不下降说明太小；loss 震荡/发散说明太大。对于 7B 模型全参微调，`1e-5` 到 `3e-5` 是安全区。

**过拟合信号：**
train loss 持续下降但 eval loss 开始回升，或者模型开始说话方式变奇怪（"灾难性遗忘"）。

**DPO β 调节：**
β 太小 → 模型过于激进，可能出现 reward hacking；β 太大 → 模型几乎不变。看 chosen/rejected reward margin 是否在扩大来判断 β 是否有效。

**学习率调度：**
Warmup + cosine 是常见且稳妥的默认项；`linear + warmup`、`constant + warmup` 在不少微调场景也仍然有效。

**验证策略：**
先用小模型（1.5B / 3B）跑消融，找到合适的超参范围，再放到大模型上验证。


## 五、Batch Size 对梯度质量与训练稳定性的深度分析

### 5.1 梯度本质上是个估计量

设数据集有 $N$ 个样本，真实梯度（全量）：

$$g = \frac{1}{N} \sum_{i=1}^{N} \nabla_\theta \ell_i(\theta)$$

Mini-batch 用 $B$ 个样本估计它：

$$\hat{g}_B = \frac{1}{B} \sum_{i \in \mathcal{B}} \nabla_\theta \ell_i(\theta)$$

$\hat{g}_B$ 是 $g$ 的**无偏估计**（期望等于真值），但有方差：

$$\text{Var}(\hat{g}_B) = \frac{\sigma^2}{B}$$

其中 $\sigma^2 = \text{Var}(\nabla_\theta \ell_i)$ 是单样本梯度的方差。**这个公式是一切的出发点。**


### 5.2 为什么 Batch Size 太小会不稳定

定义梯度信噪比（SNR）：

$$\text{SNR} = \frac{\|g\|^2}{\text{Var}(\hat{g}_B)} = \frac{B \cdot \|g\|^2}{\sigma^2}$$

$B$ 小 → SNR 小 → 每一步更新方向充满噪声，参数在乱跑。

**参数更新的实际分解：**

$$\theta_{t+1} = \theta_t - \eta \hat{g}_B = \underbrace{\theta_t - \eta g}_{\text{有效信号}} + \underbrace{\eta(g - \hat{g}_B)}_{\text{噪声项}}$$

噪声项的方差为 $\eta^2 \cdot \sigma^2 / B$，$B$ 越小噪声越大，loss 曲线会震荡，甚至发散。


### 5.3 临界 Batch Size（Critical Batch Size）

McCandlish et al. (2018) 定义了**临界批大小**：

$$B_{\text{crit}} = \frac{\sigma^2}{\|g\|^2}$$

| 区间 | 状态 | 含义 |
|------|------|------|
| $B \ll B_{\text{crit}}$ | 噪声主导 | 增大 $B$ 能显著提升效果 |
| $B \approx B_{\text{crit}}$ | 平衡点 | 最佳效率区间 |
| $B \gg B_{\text{crit}}$ | 信号主导 | 再加样本边际收益接近零，算力浪费 |

> LLM 训练中 $B_{\text{crit}}$ 在训练初期较小（梯度大），后期增大（接近收敛），这也是为什么有人会动态调节 batch size。


### 5.4 为什么 Batch Size 太大会导致泛化变差

**概念辨析：**

| | 训练 loss | 测试 loss | 本质 |
|---|---|---|---|
| 欠拟合 | 高 | 高 | 模型容量不足或训练不够 |
| 正常 | 低 | 低 | — |
| 大 batch 的问题 | 低 | **可能偏高** | 可能出现 generalization gap |
| 过拟合 | 低 | 高 | 记住了训练集的噪声 |

大 batch 的表现（train 好 / test 差）与过拟合类似，但机制不一定完全相同：
- **过拟合**：模型记住了训练集噪声
- **大 batch 泛化差**：优化轨迹噪声更小，可能更容易出现 generalization gap；“尖锐极小值”更适合看作一种常见解释框架，而不是唯一机制


### 5.5 尖锐极小值 vs 平坦极小值（Keskar et al., 2017）

定义极小值的尖锐程度：

$$\phi(\theta, \varepsilon) = \max_{\|\delta\| \leq \varepsilon} \left[ \mathcal{L}(\theta + \delta) - \mathcal{L}(\theta) \right]$$

$\phi$ 大 → 尖锐极小值；$\phi$ 小 → 平坦极小值。

**为什么平坦极小值常被认为泛化更好？**

从 Hessian 角度看，尖锐极小值意味着 $\nabla^2_\theta \mathcal{L}$ 的特征值很大——权重的微小扰动（比如测试数据分布的轻微偏移）会导致 loss 急剧上升。平坦极小值对扰动不敏感，泛化能力强。

```
Loss
 |         /\          ← 尖锐极小值（大 batch 倾向）
 |        /  \             测试时一偏移 loss 飙升
 |       /    \
 |______/      \______
 |
 |    ~~~~~~~~~       ← 平坦极小值（小 batch 倾向）
 |   /         \         测试时偏移影响小
 |__/           \____
                        θ
```

**机制：噪声的隐式正则化作用**

小 batch 的梯度噪声起到了**隐式正则化**的作用：

$$\text{噪声温度} \propto \frac{\eta}{B}$$

噪声让优化器"跳出"尖锐的窄谷，最终落在更宽阔的平坦区域。大 batch 噪声小，直接滑入最近的（可能是尖锐的）极小值。


### 5.6 学习率与 Batch Size 的联动：线性缩放规则

既然增大 $B$ 减少了噪声，等效的补救方法是**同比放大学习率**（线性缩放规则，Goyal et al., 2017）：

$$\eta_{\text{new}} = \eta_{\text{base}} \cdot \frac{B_{\text{new}}}{B_{\text{base}}}$$

**直觉：** $B$ 翻倍 → 每步梯度估计更准 → 可以迈更大的步子。

> ⚠️ 该规则只在中等 $B$ 范围成立。$B$ 过大时，即使放大 $\eta$ 也救不回泛化问题，因为噪声温度 $\eta/B$ 整体仍然更低。


### 5.7 Batch Size 影响总结

```
泛化性能 / 训练稳定性

好  |          ___________
    |        /             \
    |       /               \
    |      /                 \
坏  |_____/                   \_________

         小 B              大 B
    (高方差/不稳定)    (尖锐极小值/泛化变差)

              ↑ 最优区间 ↑
```

**实践策略：** 从小 batch 开始 warmup，逐步增大到目标 batch size，同时按线性规则放大学习率，在训练效率和泛化性能之间取得平衡。


## 六、学习率对模型训练的深度分析

### 6.1 学习率的基本作用：梯度下降的步长

参数更新公式（SGD）：

$$\theta_{t+1} = \theta_t - \eta \cdot \nabla_\theta \mathcal{L}(\theta_t)$$

$\eta$ 控制每步沿梯度方向走多远。但这个"步长"的后果远不止快慢：

**η 太大**：参数跨过极小值，在谷底两侧反复横跳，loss 震荡甚至发散：

$$\theta_{t+1} \approx \theta_t - \eta \cdot H \cdot (\theta_t - \theta^*) \quad \Rightarrow \quad \text{收敛条件：} \eta < \frac{2}{\lambda_{\max}(H)}$$

其中 $H = \nabla^2 \mathcal{L}$ 是 Hessian，$\lambda_{\max}$ 是最大特征值。η 超过这个上界，更新会沿曲率最大的方向发散。

**η 太小**：收敛极慢，且容易陷在鞍点附近出不来（梯度本来就近零，步子再小就完全不动了）。

**η 适中**：在 loss landscape 中稳步下降，同时保留足够的"动能"穿越鞍点。


### 6.2 Adam 下学习率的实际含义

实际 LLM 训练几乎都用 Adam，更新公式变为：

$$\theta_{t+1} = \theta_t - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

其中 $\hat{m}_t$ 是一阶矩（梯度方向），$\hat{v}_t$ 是二阶矩（梯度幅度的滑动平均）。

关键点：Adam 会按历史一阶/二阶矩对不同参数做自适应缩放，因此 **$\eta$ 更接近一个全局步长系数**，而不再是原始梯度幅度的直接倍数。把它理解成一种“信任半径”是有帮助的直觉，但不能简单等同于所有参数更新都会被严格压到某个固定区间。这也是为什么 Adam 的 $\eta$ 往往比 SGD 小得多。


### 6.3 Warmup + Cosine Decay 调度器

整个训练过程分三段：

```
学习率
  |
  |          /‾‾‾‾‾‾‾‾‾‾‾\
η_max |       /               \
  |      /                 \
  |     /                   \___________
η_min |____/                              \_____
  |
  |-- warmup --|----- cosine decay -----|- 尾部 -|
              T_w                       T_total    步数
```

**Warmup 阶段**（线性升温，步数 $t \in [0, T_w]$）：

$$\eta_t = \eta_{\max} \cdot \frac{t}{T_w}$$

**Cosine Decay 阶段**（步数 $t \in [T_w, T_{\text{total}}]$）：

$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min}) \left(1 + \cos\left(\pi \cdot \frac{t - T_w}{T_{\text{total}} - T_w}\right)\right)$$

**为什么需要 Warmup？**

训练初期 Adam 的二阶矩估计 $\hat{v}_t$ 还没有积累，方差极大——此时用大学习率，等效于在一个对曲率估计极差的情况下迈大步，极易破坏预训练权重中已经编码好的知识结构。Warmup 让优化器先"热身"，等 $\hat{v}_t$ 稳定后再加速。

**为什么很多人喜欢用 Cosine，而不是直接用线性衰减？**

Cosine 衰减的导数在两端接近零——训练中期学习率下降较缓，尾部也更平滑。很多实践里它是稳妥默认项，但这不意味着线性衰减一定更差；在不少短程微调里，`linear + warmup` 仍然有效。

**$\eta_{\min}$ 一般设为多少？** 常见做法包括取 0、接近 0，或取 $\eta_{\max}$ 的一个较小比例；具体取值依赖训练总步数和后期是否还希望保留学习率。

**Warmup 步数怎么设？** 一般取总步数的 1%~5%。数据量越大、模型越大，warmup 可以适当长一些（保护预训练权重的时间更长）。


### 6.4 不同情况下的学习率适用值

#### 按模型规模

| 模型规模 | 全参微调 η | LoRA 微调 η |
|---|---|---|
| 1B 以下 | 5e-5 ~ 1e-4 | 1e-4 ~ 3e-4 |
| 1B ~ 7B | 1e-5 ~ 5e-5 | 5e-5 ~ 2e-4 |
| 7B ~ 30B | 5e-6 ~ 2e-5 | 2e-5 ~ 1e-4 |
| 70B+ | 1e-6 ~ 1e-5 | 1e-5 ~ 5e-5 |

**背后逻辑：** 模型越大，每层参数对整体输出的影响越"杠杆化"，同样幅度的参数变动引起的输出变化越大，所以要用更小的步子。LoRA 只更新低秩矩阵，不直接修改原始权重，相当于加了一层缓冲，可以承受更大的学习率。

#### 按训练阶段

**CPT（继续预训练）**：通常比 SFT 大，1e-4 ~ 5e-5。数据量大、训练久，需要足够的学习率驱动参数大幅移动；但也不能太大，否则遗忘预训练知识。

**SFT（监督微调）**：1e-5 ~ 5e-5。目标是在预训练基础上精细调整行为，不需要大幅改变参数，学习率适中偏小。

**DPO（偏好优化）**：通常低于 SFT，但具体范围差异很大。全参 DPO 常比 SFT 小一截；LoRA DPO 往往能用更大的学习率。原因是它既依赖偏好数据质量，也依赖是否全参、模型规模、参考模型约束和 β 设置。

**各阶段关系示意：**

```
η
1e-4 |  CPT ████████████████
     |
5e-5 |  CPT ░░░░  SFT ████████████
     |
1e-5 |             SFT ░░░░
     |
1e-6 |                        DPO ████████
     |
1e-7 |                        DPO ░░░░░░░░
     +----------------------------------------→ 越到后期越小
```

#### 多阶段退火（Annealing）

训练后期对少量高质量领域数据做额外退火，有时能提升特定能力，尤其是较小模型在代码、数学和部分推理基准上的表现。以 Llama 3 公开报告为例，这种做法对 `8B` 模型在 `GSM8k` 和 `MATH` 上有明显提升，但对 `405B` 的收益很小，因此不宜泛化为对所有模型都能显著提升指令遵循和推理能力。


### 6.5 诊断学习率是否合适

| 现象 | 诊断 | 方向 |
|---|---|---|
| Loss 一直不下降 | η 太小，或 warmup 太长 | 放大 η，缩短 warmup |
| Loss 前期下降后震荡 | η 偏大 | 缩小 η，或加梯度裁剪 |
| Loss 迅速降到某值后停滞 | 可能陷入鞍点，η 过小 | 适当放大，或换有 momentum 的调度 |
| Loss 前期正常，后期下降极慢 | cosine 衰减太快 | 拉长 decay 阶段，或提高 η_min |
| 训练 loss 低但 eval 快速变差 | η 偏大导致过拟合 | 缩小 η，加 weight decay |

实操中最直接的方式是跑一个 **learning rate range test**：固定其他参数，将 η 从极小（1e-7）线性升到较大（1e-2），观察 loss 曲线，找到 loss 开始快速下降的点作为下界、loss 开始震荡的点作为上界，取中间偏下的值作为起始 $\eta_{\max}$。


## 七、训练轮数（Epochs）的深度分析

### 7.1 为什么 SFT 多 epoch 容易过拟合

SFT 的数据量通常很小——几千到几十万条对话。模型参数量却是几十亿。这个比例下，**模型的容量远远超过数据的信息量**。

第一轮过完数据，模型从每条样本里学到了"行为模式"。第二轮开始重复见到同样的数据，模型没有新的信息可以学，梯度开始推动参数去**记住具体样本的细节**——包括数据里的措辞习惯、格式噪声、甚至标注错误。这就是过拟合的本质：不是学规律，而是在背答案。

表现出来就是：train loss 持续下降，但 eval loss 开始回升，模型的回答越来越像训练集的语气，泛化到新问题的能力变差。

一个更直接的角度：SFT 的目标是**行为对齐**，不是最优化 loss。loss 降到某个点之后继续压低，带来的不是更好的对齐，而是过度拟合标注者的风格偏好。


### 7.2 为什么数据量大只需要过一遍

**第一个原因：信息已经够了。** 假设有 1T token 的 CPT 数据，模型参数量是 7B，参数量远小于数据量。一遍下来，每个参数平均"见过"了海量不同的上下文，梯度信号已经充分覆盖了数据分布。再过一遍，新增的信息量边际接近零——见到的都是重复的，学不到什么新东西。

**第二个原因：记忆和泛化的权衡。** 数据量大时，每条样本对参数的影响被"稀释"了（因为同一批次里有大量不同样本在竞争梯度方向），模型不容易记住单条样本，而是被迫学习跨样本的共性规律，自然泛化更好。过多 epoch 之后，同样的样本反复出现，这个"稀释效应"消失，高频样本开始被过度强化。

**统一视角**：可以用有效信息量来理解：

$$\text{有效信息} = \text{数据多样性} \times \text{epoch 数}$$

- SFT 场景：数据少，epoch 多 → 有效信息量快速饱和后继续"注水"，开始记忆噪声。
- CPT 场景：数据极多，epoch 少（甚至 < 1）→ 有效信息量仍在增长，还远没到饱和点，不需要也不应该重复。

需要注意的是，**训练时长本身不是过拟合的原因**，而是数据被重复看的次数多才是直接原因。你可以训练很久但一直喂新数据，就不会过拟合；也可以只训练很短时间但反复看同一批数据，同样过拟合。这也解释了为什么 CPT 有时甚至连一遍都跑不完——数据本身就是瓶颈，而不是训练时间。


## 八、Learning Rate Scheduler 深度分析

### 8.1 Scheduler 对训练的影响

Scheduler 本质上在回答一个问题：**在训练的不同阶段，应该用多大的步子走路？**

固定学习率的问题在于，训练初期和收敛期对 η 的需求是矛盾的——初期需要大步快速靠近极小值区域，收敛期需要小步精细定位。固定值只能折中，两头都不理想。Scheduler 的作用是让 η 随训练进程动态调整，让每个阶段都用上"恰好合适"的步长。

从 loss landscape 的视角理解：训练初期模型在高原上，梯度信号强，需要大 η 快速下山；中期进入山谷边坡，需要稳定 η 横向探索；后期到达谷底附近，需要小 η 精细收敛，避免在谷底来回震荡。


### 8.2 主要 Scheduler 类型

#### Constant（固定学习率）

$$\eta_t = \eta$$

最简单的基线，没有调度。适合快速实验验证，或者已经通过其他方式（如 Adam 的自适应性）隐式处理了步长问题的场景。实际 LLM 训练几乎不单独使用。


#### Linear Decay

$$\eta_t = \eta_{\max} \cdot \left(1 - \frac{t}{T}\right)$$

从 $\eta_{\max}$ 线性降到 0。导数是常数，全程匀速下降。在很多长程训练里，cosine 往往更常作为默认项；但 `linear + warmup` 在不少微调任务里依然常见，并没有被完全取代。


#### Cosine Decay（最主流）

$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\pi \cdot \frac{t}{T}\right)\right)$$

导数在两端接近零、中间最大，形成"慢-快-慢"的下降节奏。它是很多 LLM 训练 recipe 的稳妥默认项，通常搭配 warmup 使用，但不宜把它表述成所有主流模型都统一采用的唯一方案。


#### Cosine with Restarts（SGDR）

在 cosine decay 的基础上，每隔 $T_i$ 步将学习率重置回 $\eta_{\max}$，重新开始一次 cosine 衰减，每个周期可以递增：

$$T_{i+1} = T_i \cdot T_{\text{mult}}$$

重置的意义：把模型从当前极小值"踢出去"，强迫它探索 loss landscape 的其他区域，有助于跳出局部极小值、找到更平坦的解。代价是训练曲线会周期性反弹，不适合对收敛曲线有强监控需求的场景，且需要额外调 $T_i$ 和 $T_{\text{mult}}$。


#### Warmup + Cosine Decay（标准 LLM 方案）

详见第六节。这里补充一点：warmup 的本质是给 Adam 的二阶矩 $\hat{v}_t$ 足够的积累时间，而不只是"保护预训练权重"。$\hat{v}_t$ 在初期极不稳定，此时即使梯度方向正确，Adam 对步长的估计也是错的。warmup 让 $\hat{v}_t$ 收敛后再加速，等于是在等优化器本身"热身"完毕。


#### Polynomial Decay

$$\eta_t = (\eta_{\max} - \eta_{\min}) \cdot \left(1 - \frac{t}{T}\right)^p + \eta_{\min}$$

线性衰减（$p=1$）和更陡的衰减（$p>1$）的统一形式。$p=2$ 接近 cosine 的效果但尾部更陡，$p<1$ 则前期衰减更快。灵活性高但多了一个需要调的超参数 $p$，实际使用不多。


#### Step Decay / Multi-Step Decay

每隔固定步数将学习率乘以一个衰减因子 $\gamma$：

$$\eta_t = \eta_0 \cdot \gamma^{\lfloor t / s \rfloor}$$

曲线是阶梯状，每次"跳降"之后 loss 通常会出现一个明显的下降台阶。来自图像分类领域（ResNet 训练），在 LLM 训练中几乎不用，因为台阶式衰减会引入不必要的训练不稳定。


#### WSD（Warmup-Stable-Decay）

近期被 MiniCPM、Llama 3 等工作关注的方案，将训练分三段：

```
η_max |      /‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\
      |     /                     \
η_min |____/                       \___
      |
      |- warmup -|--- stable ---|-- decay --|
```

Stable 阶段保持固定 $\eta_{\max}$，Decay 阶段快速（cosine 或线性）降到 $\eta_{\min}$。

核心优势：stable 阶段可以**随时截断插入退火**——当临时增加高质量数据或需要在特定 checkpoint 上做退火时，不需要重新规划整个调度曲线。MiniCPM 论文证明这让训练更灵活，尤其适合数据课程（curriculum）动态调整的场景。


#### Inverse Square Root（Transformer 原始方案）

$$\eta_t = d_{\text{model}}^{-0.5} \cdot \min\left(t^{-0.5},\ t \cdot T_w^{-1.5}\right)$$

"Attention is All You Need" 原始论文中的方案，warmup 后按 $1/\sqrt{t}$ 衰减。理论上对 Adam 的收敛性有保证，但实践中在大规模 LLM 上表现不如 cosine，基本只在小模型或学术复现场景中见到。


### 8.3 如何选择 Scheduler

**默认选 Warmup + Cosine Decay**，这是一个稳妥且常见的默认项；但在不少短程微调里，`linear + warmup`、`constant + warmup` 也依然值得考虑。

**选 WSD 的场景**：训练过程中需要动态加入数据、做多阶段退火、或者希望保留"随时插入高质量退火"的灵活性。

**选 Cosine with Restarts 的场景**：有充足实验时间，且怀疑当前 loss landscape 有大量局部极小值（任务特别复杂、数据分布多样），可以尝试，但需要额外调周期参数。

**Step Decay** 在 LLM 训练里相对少见；**Linear** 则并非不能用，而是在很多长程训练中不如 cosine 那么常作为默认项。

选择时的关键判断问题：训练数据是固定的还是动态变化的（动态 → WSD）；是否需要多阶段退火（需要 → WSD）；计算资源是否充足做超参搜索（不充足 → Cosine，成熟可靠）；是预训练还是微调（微调步数少，cosine 的探索平台价值有限，有时 constant + warmup 反而更稳）。


## 九、训练步数、Warmup Ratio 与 Gradient Accumulation

### 9.1 Warmup Ratio 的含义

Warmup ratio 就是预热阶段占总训练步数的比例：

$$\text{warmup ratio} = \frac{T_w}{T_{\text{total}}}$$

前 $T_w$ 步学习率从 0 线性升到 $\eta_{\max}$，这段占总训练步数的比例就是 warmup ratio。1%~5% 是常见的经验范围，但背后有几个影响因素值得注意。

**数据量 / 训练步数**：总步数越多，绝对的 warmup 步数也应该更多，否则热身太短，Adam 的二阶矩还没稳定就直接全速跑了。实践中也有人直接指定绝对步数而不是比例；例如《The Llama 3 Herd of Models》公开过一个使用 8000 步 linear warmup 的训练配方。

**模型规模**：模型越大，参数初始状态距离稳定的梯度估计越远，warmup 可以适当拉长，给 $\hat{v}_t$ 更多时间收敛。

**训练阶段**：SFT / DPO 总步数通常很少（几百到几千步），这时 1%~5% 对应的绝对步数可能只有几十步，实际效果微乎其微，有时直接用 50~100 步的固定值更合理。

**从头预训练 vs 继续训练**：继续训练（CPT / SFT）时模型权重已经有意义，warmup 主要是保护这些权重不被破坏，不需要太长；从头预训练时初始权重本来就随机，反而让模型快点动起来更好，warmup 同样不需要太长。

真正的决策依据是**绝对步数是否足够让 Adam 的二阶矩收敛**，而不是比例本身。很多微调任务里，几百到几千步 warmup 已经够用；但在大规模预训练或特定 recipe 中，更长 warmup 也完全可能合理。


### 9.2 步数与数据量的关系

一步 = 处理一个 batch，参数更新一次，所以：

$$T_{\text{total}} = \frac{\text{总 token 数}}{\text{batch size} \times \text{sequence length}}$$

或者从样本角度：

$$T_{\text{total}} = \frac{\text{总样本数} \times \text{epoch 数}}{\text{batch size}}$$


### 9.3 Gradient Accumulation 下的步数含义

实际训练中显存往往放不下目标 batch size，所以会把一个大 batch 拆成若干小 batch 分批前向、累积梯度，最后统一做一次参数更新：

$$\text{等效 batch size} = \text{micro batch size} \times \text{gradient accumulation steps} \times \text{GPU 数}$$

这时"步"有两种含义需要区分：

**optimizer step**（参数真正更新一次）= accumulation 结束时，对应调度器里说的"一步"。

**forward step**（一次前向传播）= 每个 micro batch，不更新参数。

Scheduler、warmup ratio、学习率衰减，都是按 **optimizer step** 计数的，不是 forward step。所以"总步数"指的是参数更新的次数，不是前向传播的次数。


### 9.4 训练日志中 step 的含义

不同框架的定义不完全统一：

**Hugging Face Transformers / Accelerate**：`step` 和 `global_step` 通常是同义词，都指 optimizer step（参数更新次数）。micro-batch 的前向传播次数一般不直接暴露在日志里。

**DeepSpeed**：`global_step` 指 optimizer step，`micro_step` 或 `step` 有时指 forward step（每个 micro batch 一次）。

**Megatron-LM**：`iteration` 对应 optimizer step，日志里的 `consumed_samples` 可以反推实际处理了多少数据。

实际使用中更常见的混淆是：设置了 `gradient_accumulation_steps=4`，日志显示走了 1000 步，但实际前向传播了 4000 次。warmup_steps=100 是指 100 次参数更新，对应的"热身"实际消耗了 400 个 micro batch 的数据。

看日志时最保险的方式是直接观察 `learning_rate` 字段的变化，确认 warmup 阶段是否在按预期爬升，而不是依赖步数字段的含义猜测。


## 十、Weight Decay 与 AdamW 深度分析

### 10.1 Weight Decay 的基本机制

标准梯度下降只优化 loss：

$$\theta_{t+1} = \theta_t - \eta \cdot \nabla_\theta \mathcal{L}(\theta_t)$$

加入 weight decay 之后，每步更新之前先把参数缩小一点：

$$\theta_{t+1} = (1 - \eta \lambda) \cdot \theta_t - \eta \cdot \nabla_\theta \mathcal{L}(\theta_t)$$

其中 $\lambda$ 就是 weight decay 系数。$(1 - \eta\lambda)$ 这个因子每步都把参数往零方向拉，所以 weight decay 也叫**参数衰减**。


### 10.2 与 L2 正则化的等价关系

Weight decay 等价于在 loss 上加一个 L2 惩罚项（在 SGD 下严格等价，Adam 下有差异）：

$$\mathcal{L}_{\text{reg}}(\theta) = \mathcal{L}(\theta) + \frac{\lambda}{2} \|\theta\|^2$$

对这个带正则项的 loss 求梯度：

$$\nabla_\theta \mathcal{L}_{\text{reg}} = \nabla_\theta \mathcal{L} + \lambda \theta$$

代入参数更新：

$$\theta_{t+1} = \theta_t - \eta(\nabla_\theta \mathcal{L} + \lambda \theta_t) = (1 - \eta\lambda)\theta_t - \eta \nabla_\theta \mathcal{L}$$

和上面的形式完全一致。**L2 正则化在优化目标上加惩罚，weight decay 在参数更新上做缩减，SGD 下两者等价。**


### 10.3 为什么能防止过拟合

L2 惩罚项 $\frac{\lambda}{2}\|\theta\|^2$ 的效果是惩罚参数的绝对值过大。过拟合时模型倾向于让某些权重变得很大来"精确拟合"训练集的每个细节（包括噪声）；weight decay 给大权重施加额外的梯度压力，让优化器在"拟合数据"和"保持参数小"之间做权衡，从而避免对训练数据过度记忆。

几何直觉：L2 正则化相当于在参数空间中心放了一个"弹力球"，参数离原点越远，被拉回的力越大，最优解被约束在原点附近的一个椭球内。


### 10.4 什么情况下会过拟合

过拟合本质是模型的**有效容量超过了数据能提供的约束**，以下情况都会触发：

**数据量相对模型容量太小**：SFT 场景最典型。几千条数据、几十亿参数，模型有足够的"自由度"去记住每条样本，而不是学规律。

**训练轮数过多**：反复见同样数据，高频样本的损失被持续压低，模型开始记忆而非泛化（见第七节）。

**Learning rate 过大**：大步更新让参数在短时间内大幅偏移，跑过了泛化区域，陷入训练集特有的尖锐极小值。

**Weight decay 过小或为零**：没有正则化约束，参数可以无限增大来拟合训练集，模型自由度过高。

**数据噪声高、标注质量差**：噪声本身没有规律，模型只能靠记忆来降低 loss，必然导致过拟合。

**批次过大**：大 batch 的噪声温度低，容易陷入尖锐极小值，泛化变差（generalization gap，表现类似过拟合但机制不同，见第五节）。


### 10.5 Weight Decay 的取值

| 场景 | 常用值 |
|---|---|
| 从头预训练 | 0.1 |
| SFT 全参微调 | 0.01 ~ 0.1 |
| LoRA 微调 | 0 ~ 0.01（只作用于 LoRA 参数） |
| DPO | 0 ~ 0.01 |

λ 过大：参数被过度压缩，模型容量受限，出现欠拟合，loss 下不去。λ 过小：正则化效果不足，SFT 场景容易过拟合。通常 embedding 层和 LayerNorm 的参数不加 weight decay，只对线性层的权重矩阵施加（具体见 10.7 节）。


### 10.6 AdamW：解耦 Weight Decay 的优化器

Adam 的更新公式里有自适应缩放 $\frac{\hat{m}_t}{\sqrt{\hat{v}_t}}$，如果直接把 L2 梯度加进去（即 L2 正则化），这个额外的梯度也会被自适应缩放，导致**不同参数受到不同强度的正则化**，效果不稳定。

AdamW（Loshchilov & Hutter, 2019）把衰减项从梯度里剥离出来，直接作用在参数上：

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$

$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t} \qquad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

$$\theta_{t+1} = \theta_t - \eta \left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon} + \lambda\theta_t\right)$$

其中 $m_t$ 是梯度的指数移动平均（一阶矩，估计梯度**方向**），$v_t$ 是梯度平方的指数移动平均（二阶矩，估计梯度**幅度**），$\hat{m}_t$ 和 $\hat{v}_t$ 是 bias correction 之后的版本。

**Bias correction 的意义**：初始时 $m_0 = v_0 = 0$，前几步的 EMA 会系统性低估真实梯度（例如第一步 $m_1 = (1-\beta_1)g_1$，比真实梯度小了 $(1-\beta_1)$ 倍）。除以 $(1-\beta_t^t)$ 可以校正这个偏差——训练初期 correction 因子显著，随着 $t$ 增大 $\beta^t \to 0$，影响逐渐消失。

**Adam 对单步异常梯度的自保护**：当 $g_t$ 异常大（设为 $M \gg$ 历史梯度）时，$m_t \approx (1-\beta_1)M$，$\sqrt{v_t} \approx \sqrt{1-\beta_2} \cdot M$，两者都正比于 $M$，所以：

$$\frac{\hat{m}_t}{\sqrt{\hat{v}_t}} \approx \frac{(1-\beta_1)M}{\sqrt{1-\beta_2} \cdot M} = \frac{1-\beta_1}{\sqrt{1-\beta_2}} \approx 0.447$$

$M$ 被约掉，当步更新量与异常梯度的大小无关。这是 Adam 区别于 SGD 的重要性质——SGD 直接用 $\eta \cdot g_t$ 更新，没有任何归一化，gradient clipping 对 SGD 是刚需，对 Adam 更多是工程安全边界（尤其用于防止 fp16 数值溢出）。

L2 梯度项走自适应缩放，weight decay 项不走。这就是 Adam 和 AdamW 的核心区别——**LLM 训练应该用 AdamW，而不是带 L2 正则的 Adam**。

一些 LLM 预训练 recipe 常用的超参数是：$\beta_1 = 0.9$，$\beta_2 = 0.95$，$\epsilon = 1\text{e-8}$，并常配合较大的 weight decay；但这不是 AdamW 的通用默认值。以 PyTorch 为例，AdamW 默认 `betas` 仍是 `(0.9, 0.999)`。

$\beta_2$ 从 0.999 改成 0.95 这一点值得留意——0.999 会让历史梯度的影响衰减极慢，在长序列、大 batch 的 LLM 训练中容易造成自适应步长过于保守，0.95 让优化器对梯度变化更敏感，收敛更快。


### 10.7 对不同层施加不同 Weight Decay

#### 为什么 Embedding 和 LayerNorm 通常不加

**Embedding**：每一行是一个 token 的向量表示，其模长可能与词频、训练动力学和表示几何相关。是否对 embedding 做 weight decay 更像工程取舍，而不是一个有统一理论定论的问题；过强的 weight decay 可能压缩本就较弱的低频 token 表示。

**LayerNorm**：参数只有缩放系数 $\gamma$ 和偏移 $\beta$，直接决定每层输出的数值范围。压缩这两个参数会破坏各层之间精心校准的激活分布，导致训练不稳定甚至梯度消失。Bias 项同理，其绝对值代表输出的基准偏移，不应强制压向零。

#### PyTorch 实现（Parameter Group）

```python
def get_param_groups(model, weight_decay):
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if (
            param.ndim <= 1          # bias 和 LayerNorm 参数都是 1D
            or "bias" in name
            or "layernorm" in name.lower()
            or "layer_norm" in name.lower()
            or "norm" in name.lower()
            or "embed" in name.lower()
        ):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return [
        {"params": decay_params,    "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

param_groups = get_param_groups(model, weight_decay=0.1)
optimizer = torch.optim.AdamW(param_groups, lr=1e-4, betas=(0.9, 0.95))
```

#### HuggingFace Trainer 的实现

HF Trainer 内部自动处理，逻辑是把所有 LayerNorm 层的参数和所有 bias 排除在 weight decay 之外，**实际上对 embedding 是加了 weight decay 的**（embedding 权重是 2D，不含 `bias`，不是 LayerNorm 层，会落入 decay 组）：

```python
decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
decay_parameters = [n for n in decay_parameters if "bias" not in n]

optimizer_grouped_parameters = [
    {"params": [p for n, p in model.named_parameters()
                if n in decay_parameters],
     "weight_decay": args.weight_decay},
    {"params": [p for n, p in model.named_parameters()
                if n not in decay_parameters],
     "weight_decay": 0.0},
]
```

#### DeepSpeed 的实现

DeepSpeed 本身不负责参数分组，由用户在初始化前按 PyTorch 方式完成，再传入 `deepspeed.initialize()`：

```python
param_groups = get_param_groups(model, weight_decay=0.1)
optimizer = torch.optim.AdamW(param_groups, lr=1e-4)

model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    optimizer=optimizer,
    config=ds_config,
)
```

ZeRO-1/2/3 会把 optimizer state、梯度、参数分片到多个 GPU，但 param_groups 的结构不变——`DeepSpeedZeroOptimizer` 内部记录每个参数分片属于哪个 group，更新时按 group 取对应的 weight_decay，分组逻辑对上层透明。

若通过 config 文件指定优化器，weight_decay 只能全局设置，无法分组，实际 LLM 训练中通常不用这种方式。

#### Megatron-LM 的实现

Megatron 有内置函数处理参数分组，用 `isinstance` 检查模块类型而非名字匹配，更严格：

```python
def _get_params_for_weight_decay_optimization(modules):
    weight_decay_params = {'params': []}
    no_weight_decay_params = {'params': [], 'weight_decay': 0.0}

    for module in modules:
        for module_ in module.modules():
            if isinstance(module_, (LayerNorm, RMSNorm)):
                no_weight_decay_params['params'].extend(
                    [p for p in module_._parameters.values() if p is not None]
                )
            else:
                weight_decay_params['params'].extend(
                    [p for p in module_._parameters.values()
                     if p is not None and p.ndim >= 2]
                )
                no_weight_decay_params['params'].extend(
                    [p for p in module_._parameters.values()
                     if p is not None and p.ndim < 2]
                )
    return weight_decay_params, no_weight_decay_params
```

`ndim >= 2` 的逻辑同样会把 embedding（2D）归入 weight decay 组，和 HF 一样是刻意设计。张量并行切分后参数 `ndim` 不变，分组逻辑仍然正确。


### 10.8 Embedding 加不加 Weight Decay 的实际取舍

学术上建议 embedding 不加 weight decay，但 **HuggingFace 和 Megatron 的默认实现都对 embedding 加了 weight decay**，这不是疏漏，而是不同场景下的权衡：

**预训练场景**：数据量极大，embedding 见过足够多的上下文，不容易过拟合；weight decay 反而能防止高频 token 的向量模长随训练步数无限增大。

**SFT 场景**：数据量小，低频 token 的 embedding 梯度信号本来就弱，再加 weight decay 可能让这些 token 的表示越压越小，学不到东西。理论上 SFT 时应把 embedding 排除，但实践中影响通常不大——真正影响 SFT 效果的是学习率和数据质量，而不是 embedding 的 weight decay 设置。

| 框架 | Embedding 是否加 weight decay | 依据 |
|---|---|---|
| HuggingFace Trainer | **是**（默认） | 名字不含 `bias`，不是 LayerNorm |
| Megatron-LM | **是**（默认） | `ndim >= 2` |
| 手写 PyTorch | 用户决定 | 可通过名字过滤排除 |


## 十一、Max Sequence Length 深度分析

### 11.1 训练框架里序列长度是固定的吗

三个框架默认都要求同一个 batch 内所有序列等长，因为 GPU 矩阵运算要求张量形状一致。但"等长"的实现方式有两种：

**Padding**：把短序列用特殊 token（`<pad>`）补到 `max_seq_len`，attention mask 标记哪些位置是真实内容、哪些是 padding，loss 计算时 mask 掉 padding 位置。

**Packing**：把多条短序列拼接成一条长序列，直到填满 `max_seq_len`，用 attention mask 或 position id 隔离不同样本之间的注意力。

常见做法是：Hugging Face 生态里往往依赖 data collator 做 padding；packing 通常需要自定义数据处理。Megatron-LM 更常见 packed/contiguous token 训练，但具体行为仍取决于数据管线和配置；DeepSpeed 本身不负责上层样本拼接策略。


### 11.2 截断与填充的代价

**截断**：超过 `max_seq_len` 的部分直接丢掉，这条样本的后半段信息完全丢失。对长文档任务（摘要、长对话）是实质性的信息损失。

**Padding 的浪费**：短序列补全到 `max_seq_len` 后，padding 位置前向传播照样算，GPU 在算这些 padding token 时没有产生任何有效梯度，是纯粹的算力浪费。一条平均长度 200 token 的样本在 `max_seq_len=4096` 下，有效计算比例只有 ~5%，95% 的算力打水漂。

**Packing 的效率**：把多条短样本拼到一个序列里，有效 token 密度接近 100%，几乎没有浪费。大规模预训练几乎都用 packing 而不是 padding。


### 11.3 Packing 的注意事项：跨样本 Attention 泄漏

朴素的 packing 只是把序列拼在一起，如果不处理 attention mask，样本 B 的 token 可以 attend 到样本 A 的内容，相当于把两条无关的对话强行关联，引入噪声。

正确做法是用 **document mask**（也叫 sample mask）：每条子序列只能 attend 自己内部的 token，跨样本的注意力全部 mask 掉。HuggingFace 的 `DataCollatorForSeq2Seq` 不自动处理这个，需要手动传入正确的 attention mask 或 position ids。对于 Megatron 一类 packed sample 场景，仅重置 `position ids` 并不足以保证不发生跨样本 attention，还需要配套的 attention mask / boundary mask 逻辑。


### 11.4 超长上下文与 Max Sequence Length 的关系

超长上下文训练的核心矛盾就是 `max_seq_len` 的扩展问题。标准 Transformer 的注意力计算复杂度是 $O(n^2)$，显存占用随序列长度平方增长，直接扩展 `max_seq_len` 在工程上面临三个障碍：显存、位置编码泛化、计算效率。

**RoPE 外推（Position Interpolation）**：原始 RoPE 位置编码在训练长度之外会退化（模型没见过那么大的位置 id）。Position Interpolation（PI）把超出训练范围的位置线性插值映射回训练范围，让模型能泛化到更长序列，不需要从头训练。YaRN 是更精细的变体，对不同频率的旋转维度做非线性缩放。

**Flash Attention**：重新设计注意力的计算顺序，利用 GPU SRAM 做分块计算，把显存占用从 $O(n^2)$ 降到 $O(n)$，让长序列在显存上变得可行。没有 Flash Attention，128K 上下文在单卡上根本放不下。

**长上下文继续训练（Long Context CPT）**：把 `max_seq_len` 从较短上下文扩展到更长上下文，通常需要额外的长上下文训练或适配。以 Llama 3 系列公开资料为例，最大模型支持到 128K，但不宜把这一结论泛化成“Llama 3 都是从 4096 直接扩到 128K，且方式完全一致”。

**序列并行（Sequence Parallelism）**：当序列长到一张卡放不下时，把序列切分到多张卡上并行计算注意力，这是 Megatron 和 DeepSpeed-Ulysses 提供的功能。


### 11.5 总结

```
短序列 + Padding  →  算力浪费严重
短序列 + Packing  →  高效，但需处理跨样本 attention 泄漏
长序列超出上限   →  截断，信息损失
超长上下文训练   →  RoPE 外推 + Flash Attention +
                    长文档数据 + 可能需要序列并行
```

`max_seq_len` 的选择本质是在**训练效率、信息完整性、显存成本**之间做权衡，不是越大越好，也不是越小越省——要根据任务的实际序列长度分布来定。


## 十二、Gradient Clipping 深度分析

### 12.1 是什么

Gradient Clipping 是在参数更新之前对梯度幅度做约束的操作。最常用的是 **Global Norm Clipping**：

$$\|g\|_2 = \sqrt{\sum_{i} g_i^2}$$

$$g \leftarrow g \cdot \min\left(1,\ \frac{\text{max\_norm}}{\|g\|_2}\right)$$

即：计算所有参数梯度拼成的向量的 L2 范数，如果超过阈值 `max_norm`，就等比例缩小整个梯度向量使其范数恰好等于 `max_norm`；没超过则不做任何处理。

关键点：**等比例缩小，方向不变，只是缩短了长度**。这和直接截断每个参数的梯度（per-element clipping）有本质区别。


### 12.2 为什么引入：梯度爆炸问题

深层网络的反向传播是链式法则连乘：

$$\frac{\partial \mathcal{L}}{\partial \theta_0} = \frac{\partial \mathcal{L}}{\partial h_L} \cdot \prod_{k=1}^{L} \frac{\partial h_k}{\partial h_{k-1}}$$

如果每层 Jacobian 矩阵的谱范数大于 1，连乘之后梯度会指数级增大——**梯度爆炸**。对于 $L$ 层网络，梯度范数可达 $\rho^L$ 量级（$\rho > 1$ 是每层的放大因子）。

LLM 训练中，梯度爆炸通常由以下几种情况触发：训练初期权重未稳定，某些层激活方差很大，反向传播信号被大幅放大；遇到异常长的序列或罕见语言模式，loss 对该样本特别敏感；学习率设置偏大时，梯度方向没问题但步长过大，等效于"梯度爆炸"的效果。

梯度爆炸的表现：loss 突然从正常值跳到 NaN 或异常大的数，训练直接崩掉。


### 12.3 对训练的影响

**稳定性**：把梯度范数限制在合理范围内，防止一次异常更新破坏已学好的权重。对 LLM 尤其重要——几百亿参数训练几天，一次爆炸就要从 checkpoint 回滚。

**隐式的学习率上界**：参数更新量满足：

$$\|\Delta\theta\| = \eta \cdot \|g_{\text{clipped}}\| \leq \eta \cdot \text{max\_norm}$$

Gradient clipping 给每步参数更新的幅度设了硬上界 $\eta \cdot \text{max\_norm}$，是在动态保护学习率不会"有效过大"。

**只干预异常情况**：如果梯度本来就在 `max_norm` 以内，clipping 完全不介入。这比直接调小学习率更优雅——小学习率让所有步都走得更小，clipping 只处理异常。

**grad_norm 是重要监控指标**：训练日志里的 `grad_norm` 曲线能反映训练健康状况。训练初期 grad_norm 通常较大，随着收敛逐渐降低。偶尔出现尖峰后恢复正常是健康的；如果持续贴着 `max_norm` 走，说明 `max_norm` 设得太小或学习率太大。


### 12.4 max_norm 的取值

标准默认值是 `1.0`，几乎所有主流 LLM 训练都用这个值。

`max_norm=1.0` 意味着参数更新量最大为 $\eta$（学习率），对于 `1e-4` 的学习率，每步最多移动 `1e-4` 的距离，在大多数场景下合理。

| 场景 | max_norm |
|---|---|
| 预训练 | 1.0 |
| SFT | 1.0 |
| DPO | 1.0（有时用 0.5，因为 DPO 梯度信号弱，不希望偶尔的大梯度主导更新） |
| LoRA 微调 | 1.0，有时可放宽到 5.0（只更新少量参数，爆炸风险低） |

诊断方法：观察 `grad_norm` 的分布。如果 95% 的步数里 grad_norm 都远低于 `max_norm`，说明 clipping 几乎没有生效，设置合理。


### 12.5 与 Adam 的交互：一个常见误解的纠正

Adam 有自适应步长，那 gradient clipping 在 Adam 下还有必要吗？理解这个问题需要先推导 Adam 对异常梯度的实际响应。

当 $g_t$ 异常大（设为 $M \gg$ 历史梯度）时：

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) M \approx (1-\beta_1) M$$

$$v_t = \beta_2 v_{t-1} + (1-\beta_2) M^2 \approx (1-\beta_2) M^2$$

$m_t$ 和 $\sqrt{v_t}$ 都正比于 $M$，因此当步更新量：

$$\frac{\hat{m}_t}{\sqrt{\hat{v}_t}} \approx \frac{(1-\beta_1)M}{\sqrt{(1-\beta_2)} \cdot M} = \frac{1-\beta_1}{\sqrt{1-\beta_2}} \approx 0.447$$

**$M$ 被约掉，更新量与异常梯度的大小无关**。Adam 的一阶矩和二阶矩都随 $M$ 等比例增长，两者互相抵消，自适应机制对单步异常梯度有内生的保护能力。

因此，gradient clipping 对 Adam **不像对 SGD 那样是刚需**（SGD 直接用 $\eta \cdot g_t$ 更新，没有任何归一化），更多是工程安全边界，主要价值在于：

**fp16 数值溢出**：混合精度训练下梯度以 fp16 存储，上限约 65504，超过直接变 inf/NaN，Adam 的自适应机制根本来不及介入。Gradient clipping 在这一步之前截住异常梯度，是 fp16 训练的必要保护。

**连续多步异常梯度**：如果不是单个尖峰而是持续多步的大梯度，参数每步都在移动 $\eta \times 0.447$，累积位移不小。Gradient clipping 限制了每步的移动量。


## 十三、SFT 数据配比（Data Mixture Ratio）深度分析

### 13.1 什么是数据配比

SFT 阶段通常有多个数据源：通用对话、代码、数学、指令跟随、特定领域……数据配比指的是训练时各类数据所占的比例。

**它往往比调学习率更重要。** 学习率调不好，训练曲线会抖或者收敛慢；配比调不好，模型能力会出现结构性偏差——在某些任务上表现很好，在其他任务上明显退化。

### 13.2 两种混合策略

**静态混合（Static Mixing）**：在开始训练前把各数据源按比例混好，打乱后当作一个大数据集来训练。实现简单，但灵活性低——如果中途发现某类数据比例不对，要重新处理数据集。

**动态混合（Dynamic Mixing）**：训练时每个 batch 按概率从各数据源采样。可以随时调整比例，也方便实现课程学习（Curriculum Learning）——训练前期多采质量高的数据，后期逐步引入难例。

### 13.3 配比为什么影响大

**遗忘问题**：某类数据占比太低，模型会在训练过程中"遗忘"对应能力。比如代码数据只占 2%，训练结束后代码能力可能明显下滑。

**任务过拟合**：某类数据占比太高，模型会过度偏向那个分布。比如全部用单轮问答训练，多轮对话能力会变差。

**质量放大效应**：高质量数据的权重会被放大。如果某类噪声大的数据占比过高，会污染整体训练信号，影响不只是那一类任务，而是全局能力。

### 13.4 核心调节方法：幂律采样

设第 $i$ 类数据有 $n_i$ 个样本，采样概率为：

$$p_i \propto n_i^{\alpha}$$

- $\alpha = 1$：按原始数量采样，数据量大的完全主导
- $\alpha = 0$：均匀采样，每类各占 $1/K$
- $\alpha \in (0, 1)$：折中，$\alpha \approx 0.7$ 是 LLM 训练中常用的经验值（兼顾覆盖率和数量优势）

**直觉**：数据量大的来源已经见了很多，继续按线性比例追加边际收益递减；数据量小的来源需要被额外上采样，否则模型几乎看不见这类分布。

**上采样的实际操作**：
- 直接重复数据（Repeat）：把数据量少的来源复制多份
- 合成扩充（Augmentation）：如果真实数据太少，用更强的模型合成更多同类数据

### 13.5 调节方法论

**以 eval 为指导，而非 train loss。**

Train loss 是所有类型数据的混合，看 train loss 下降不能判断各类任务的质量。正确做法是：

1. 为每类任务准备专属的 eval 集（代码、数学、对话……各一套）
2. 训练一个 baseline，记录各 eval 指标
3. 调整配比，重新训练
4. 对比各 eval 指标的变化，而不是 train loss

**常见的调节流程**：

```
初始配比（经验猜测或参考已有工作）
    ↓
小规模实验（1/10 数据，1/10 步数）
    ↓
跑完 → 评估各任务 eval 指标
    ↓
发现某任务掉点 → 上调对应数据比例
发现某任务过强但其他任务掉点 → 下调对应比例
    ↓
确认配比后上大规模训练
```

### 13.6 常见坑

**用 train loss 当配比调优信号**：train loss 整体下降但专项任务可能已经退化，必须用专项 eval。

**忽略跨任务迁移**：有时候减少某类数据不会导致对应能力明显下降，因为其他类型数据有迁移效果（比如高质量推理数据会提升代码表现）。配比实验要看全局指标，不要只看直接相关任务。

**把上采样当作有新数据**：重复数据（Repeat）本质上是让模型多次看到同样的样本，多轮 epoch 的效果，过度重复会导致过拟合。一般对稀少类数据上采样不超过 3~5 倍，超过了考虑合成新数据。

**数据质量不一致**：不同来源的数据质量差异很大。低质量数据哪怕占比不高也会污染训练信号，清洗和质量过滤先于配比调整。

### 13.7 通用对话模型的经验参考配比

以下是社区公开经验和论文（如 Llama 3）中对通用能力 SFT 的大致配比参考：

| 数据类型 | 大致比例 |
|---------|---------|
| 通用对话 / 指令跟随 | 40%~60% |
| 代码 | 15%~25% |
| 数学 / 推理 | 10%~20% |
| 长文档 / RAG 相关 | 5%~10% |
| 特定领域（可选） | 5%~15% |

这些比例不是金律，实际以 eval 指标为准。模型的预训练数据分布也会影响 SFT 配比的最优值——预训练阶段代码数据比例高的模型，SFT 阶段代码数据的边际贡献更小。


## 十四、DPO 数据质量：Chosen/Rejected 差异的两个维度

### 14.1 "差异太小学不到东西"指的是哪种差异

这句话混用了两个不同维度，需要分开看。

**维度一：数据质量差异（标注质量）**

这是"差异太小学不到东西"原本指的意思。

Chosen 和 Rejected 的内容质量几乎一样，比如两个回答都正确、只是措辞略有不同，偏好标注就变成了主观噪声。此时梯度方向是随机的——模型不是"学不到"，而是"学到了错的东西"，在噪声上优化。

**维度二：从模型当前状态看的难易度（梯度大小）**

DPO 的损失函数：

$$\mathcal{L} = -\log \sigma\!\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)$$

设隐式奖励差 $\Delta r = \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}$，则梯度权重正比于：

$$\sigma(-\Delta r) = \frac{1}{1+e^{\Delta r}}$$

| 模型状态 | $\Delta r$ | $\sigma(-\Delta r)$ | 梯度大小 |
|---------|-----------|-------------------|---------|
| 已经强烈偏好 chosen | $\gg 0$ | $\approx 0$ | 接近零，easy example |
| 对两者没有偏好 | $\approx 0$ | $\approx 0.5$ | 最大，hard example |
| 错误偏好 rejected | $\ll 0$ | $\approx 1$ | 最大，very hard example |

从模型的角度：**差异大且模型已经学会 → easy example → 梯度趋近于零 → 同样学不到东西**。

### 14.2 两个维度组合起来

最理想的训练数据：

- **数据质量差异大**（标注可靠，方向正确）
- **模型当前还没区分出来**（$\Delta r$ 接近零，梯度大）

最差的两种情况：

- 数据质量差异小 → 标注噪声 → 梯度方向随机，模型学坏
- 数据质量差异大但模型已经完全学会 → 梯度为零 → 浪费算力

这也是为什么 DPO 进阶版本（如 **online DPO** 和 **iterative DPO**）要动态生成 hard examples：用当前模型采样 chosen/rejected 对，专门找 $\Delta r \approx 0$ 的样本，而不是用静态数据集一遍遍重复训练已经会的 easy pairs。


## 十五、GRPO vs DPO：现代 RLHF 流程的分化

### 15.1 不是替代，而是分化

DPO 仍然广泛用于通用对齐（helpfulness、harmlessness、风格偏好）；GRPO 以及更广泛的 online RL 方法在**推理能力提升**场景（数学、代码）里已经成为主流，DeepSeek-R1 之后尤其如此。这两类方法针对的问题不完全一样，不是简单的替代关系。

### 15.2 GRPO 的核心机制

GRPO 对同一个 prompt，采样 $G$ 个输出，用 reward 的组内相对值做 advantage：

$$A_i = \frac{r_i - \text{mean}(r_1, \ldots, r_G)}{\text{std}(r_1, \ldots, r_G)}$$

然后用类似 PPO 的 clip 目标优化，但省掉了 critic 网络（用组内均值代替 baseline）。

关键优势是天然适配**可验证奖励（verifiable reward）**——数学题对不对、代码能不能跑——这类奖励信号客观、无偏，不需要训一个奖励模型。DPO 依赖人工标注的偏好对，在推理类任务上收集这类数据既贵又慢，且标注者本身可能判断不了复杂推理的对错。

### 15.3 DPO 的优势还在的地方

DPO 是 offline 方法，不需要在线采样，训练成本低得多。对于**风格、安全、指令跟随**这类没有客观答案的对齐任务，DPO（或其变体 SimPO、IPO 等）更实用——这里本来就没有可验证的奖励函数，只有人类偏好，正是 DPO 的主场。

### 15.4 目前的主流分工

| 目标 | 常用方法 |
|------|---------|
| 通用对话风格 / 安全对齐 | DPO / SimPO / IPO |
| 数学 / 代码推理能力 | GRPO / online PPO |
| 混合目标（通用 + 推理） | SFT → DPO → GRPO 多阶段，或 online DPO |

DeepSeek-R1 之后，"SFT → GRPO"的路径在推理模型领域已经成为事实标准，但如果目标不是推理能力而是通用对话质量，DPO 还是更轻量可行的选择。


## 十六、DPO 专有参数 β 深度分析

### 16.1 β 的来源：带 KL 约束的 RLHF 目标

$$\max_{\pi_\theta} \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi_\theta} \left[ r(x, y) \right] - \beta \, D_{\text{KL}}\!\left[\pi_\theta(y|x) \,\|\, \pi_{\text{ref}}(y|x)\right]$$

这是"奖励最大化 + 不要偏离参考模型太远"的约束优化。β 是 KL 惩罚项的系数，控制两者的权衡。该目标有闭式最优解：

$$\pi^*(y|x) = \frac{1}{Z(x)}\, \pi_{\text{ref}}(y|x) \exp\!\left(\frac{r(x,y)}{\beta}\right)$$

其中 $Z(x)$ 是归一化常数。

### 16.2 β 如何被吸收进 DPO loss

对上式取对数，反解出奖励：

$$r(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)$$

代入 Bradley-Terry 偏好模型 $p(y_w \succ y_l) = \sigma(r(x,y_w) - r(x,y_l))$，$\beta \log Z(x)$ 项相消，得：

$$p(y_w \succ y_l \mid x) = \sigma\!\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)$$

DPO loss 是这个概率的负对数似然：

$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}\!\left[\log \sigma\!\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]$$

### 16.3 β 的实际作用

β 直接控制模型偏离参考模型的幅度：

**β 小**：KL 惩罚弱，模型可以大幅偏离 $\pi_{\text{ref}}$，对偏好信号响应激进。极端情况下会退化——chosen 的概率被拉高的同时，rejected 的概率可能崩塌到接近零，模型丧失多样性。

**β 大**：KL 惩罚强，模型更新保守，始终贴近参考模型，偏好信号的影响被压制。

实践中 β 通常取 **0.01～0.5**，常用值 **0.1**。

### 16.4 为什么说是 DPO 专有参数

PPO-based RLHF 也有 KL 惩罚，但它是作为独立的 `kl_coef` 项显式加在 shaped reward 上，与策略梯度分离：

$$r_{\text{shaped}}(x,y) = r_\phi(x,y) - \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$$

DPO 没有奖励模型，β 被直接推导进 loss 函数本身——它不是一个外加的正则项，而是从 RLHF 目标的数学推导中自然出现的、决定整个 loss 形状的核心参数。调 β 就是在调"这次 alignment 允许模型改变多大"，这个语义是 DPO 独有的。


## 十七、LoRA / QLoRA 参数深度分析

### 17.1 LoRA 基础公式

LoRA 冻结原始权重 $W_0 \in \mathbb{R}^{d \times k}$，在旁路加入低秩分解：

$$W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} B A$$

其中 $B \in \mathbb{R}^{d \times r}$，$A \in \mathbb{R}^{r \times k}$，$r \ll \min(d, k)$。前向传播：

$$h = W_0 x + \frac{\alpha}{r} B A x$$

初始化时 $A$ 用高斯随机，$B$ 全零，因此训练开始时 $\Delta W = 0$，不破坏预训练权重。

### 17.2 lora_r（秩）

$r$ 决定低秩分解的秩，直接控制可训练参数量。单个权重矩阵的参数量：

$$\text{LoRA 参数} = r \cdot d + r \cdot k = r(d + k)$$

原始参数量为 $d \times k$，压缩比约：

$$\frac{r(d+k)}{dk} \approx \frac{2r}{\min(d,k)}$$

对于 $d = k = 4096$ 的矩阵，$r=8$ 时压缩比约 $0.4\%$。

$r$ 小 → 参数少，欠拟合风险，训练快，内存省；$r$ 大 → 表达力强，趋近全量微调，但边际收益递减。$r$ 超过一定阈值（通常 64～128）后效果提升不明显。

### 17.3 lora_alpha（缩放因子）

$\alpha$ 控制 LoRA 更新的实际幅度，通过 $\alpha/r$ 这个比值作用：

$$\Delta W_{\text{effective}} = \frac{\alpha}{r} B A$$

LoRA 参数的有效学习率：

$$\eta_{\text{LoRA}} = \eta_{\text{base}} \times \frac{\alpha}{r}$$

常见用法：$\alpha = r$（缩放因子为 1），$\alpha = 2r$（放大一倍，常用于 SFT），或固定 $\alpha$（如 16 或 32）调节 $r$（此时增大 $r$ 相当于降低有效学习率）。

**直接理解**：$\alpha/r$ 越大，LoRA 更新对模型的影响越激进。

### 17.4 lora_dropout

应用在 LoRA 旁路的输入上：

$$h = W_0 x + \frac{\alpha}{r} B \cdot \text{Dropout}(Ax,\; p)$$

作为正则化手段，防止 LoRA 权重在小数据集上过拟合。大数据集（CPT）通常设 0；小数据集 SFT / DPO 设 0.05～0.1。

### 17.5 QLoRA 与 LoRA 的区别

QLoRA 在 LoRA 基础上加入三项工程：

**① 4-bit NormalFloat（NF4）量化基础模型**：$W_0$ 以 4-bit 存储，利用正态分布分位数设计量化格点，比线性量化对 LLM 权重更准确。前向传播时实时反量化到 BF16 参与计算，LoRA 的 $A$、$B$ 始终保持 BF16。

**② 双重量化（Double Quantization）**：对量化常数本身再做一次量化，额外节省约 0.37 bits/param。

**③ 分页优化器（Paged Optimizer）**：利用 NVIDIA 统一内存，将优化器状态在 GPU 内存紧张时换页到 CPU，避免峰值 OOM。

**内存对比（7B 模型）：**

| 方法 | 基础模型显存 | 总显存估算 |
|------|------------|----------|
| 全量 BF16 | ~14 GB | ~60+ GB（含优化器） |
| LoRA（BF16 基础） | ~14 GB | ~18 GB |
| QLoRA（4-bit 基础） | ~3.5 GB | ~6 GB |

质量代价：NF4 量化引入精度损失。对 SFT 小数据集影响可忽略；对 CPT 长时间训练会有可观测退化。资源允许时优先 LoRA。

### 17.6 不同场景的推荐值

**按模型大小：**

| 模型规模 | lora_r | lora_alpha | lora_dropout | 方法 |
|---------|--------|-----------|-------------|------|
| 1B～3B | 4～16 | r 或 2r | 0.05 | LoRA |
| 7B～13B | 8～32 | 16～64 | 0.05 | LoRA / QLoRA |
| 30B～70B | 16～64 | 32～128 | 0.05 | LoRA（多卡）/ QLoRA（单卡） |
| 100B+ | 32～64 | 64～128 | 0 | QLoRA 几乎必选 |

**按训练阶段：**

- **CPT**：需要注入大量新知识，$r = 32 \sim 64$，$\alpha = r$，dropout = 0，覆盖 attention + MLP 全部线性层。数据量够大时考虑全量微调，LoRA 的低秩压缩会成为瓶颈。
- **SFT**：$r = 8 \sim 32$，$\alpha = 2r$，dropout = 0.05，覆盖 Q/K/V/O + MLP。
- **DPO**：只做偏好对齐，改变很小，$r = 4 \sim 16$，$\alpha = r$，dropout = 0.05，有时只作用在 Q/V 就够了。
- **PPO（actor）**：$r = 8 \sim 16$，$\alpha = 2r$，dropout = 0.05；critic model 通常全量微调或更大的 $r$。

### 17.7 LoRA 作用在哪些层

Transformer 里的可选矩阵：

```
Attention:  Q, K, V, O（output projection）
MLP/FFN:    gate_proj, up_proj, down_proj（LLaMA style）
            fc1, fc2（GPT style）
Embedding:  token embedding
LM Head:    output embedding
```

**HuggingFace PEFT**：通过 `target_modules` 配置，默认只含 Q/V，推荐扩展到全部线性层：

```python
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]
```

Embedding 和 LM Head 默认不含，可通过 `modules_to_save` 单独做全量微调。

**DeepSpeed**：本身不实现 LoRA，层选择由 PEFT 控制。ZeRO-3 下 LoRA 参数被分片到多个 GPU，需确保参数聚合逻辑正确（部分框架如 LLaMA-Factory 有专门处理）；ZeRO-2 下 LoRA 参数不分片，相对简单。

**Megatron-LM**：有自己的 LoRA 实现，默认只作用在 **attention 的 Q/K/V/O**，**不包含 MLP**——这是与 PEFT 最主要的差异。原因是 Megatron 的 MLP 做了列并行 + 行并行切分，把 LoRA 插进去需要额外处理张量并行边界，官方实现默认跳过。

| 框架 | LoRA 实现 | 默认目标层 | MLP 支持 |
|------|----------|-----------|---------|
| HuggingFace PEFT | 独立库 | Q/V（可扩展） | 支持，需手动配置 |
| DeepSpeed | 依赖 PEFT | 同 PEFT | 同 PEFT |
| Megatron-LM | 内置 | Q/K/V/O | 默认不含，需额外处理 |

实践上，覆盖 attention + MLP 全部线性层比只覆盖 Q/V 效果更好，尤其是 CPT 和较大 $r$ 的场景。


## 参考文献

- Keskar, N. S., et al. (2017). *On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima.* ICLR 2017.
- McCandlish, S., et al. (2018). *An Empirical Model of Large-Batch Training.* arXiv:1812.06162.
- Goyal, P., et al. (2017). *Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour.* arXiv:1706.02677.
- Rafailov, R., et al. (2023). *Direct Preference Optimization: Your Language Model is Secretly a Reward Model.* NeurIPS 2023.
- Dubey, A., et al. (2024). *The Llama 3 Herd of Models.* arXiv:2407.21783.
- Loshchilov, I., & Hutter, F. (2017). *SGDR: Stochastic Gradient Descent with Warm Restarts.* ICLR 2017.
- Loshchilov, I., & Hutter, F. (2019). *Decoupled Weight Decay Regularization.* ICLR 2019.
- Vaswani, A., et al. (2017). *Attention Is All You Need.* NeurIPS 2017.
- Hu, S., et al. (2024). *MiniCPM: Unleashing the Potential of Small Language Models with Scalable Training Strategies.* arXiv:2404.06395.
