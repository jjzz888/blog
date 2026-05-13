---
layout: post
title: "A Practical Guide to LLM Training Hyperparameters"
date: 2026-05-14 00:00:00 +0800
categories: [llm, machine-learning]
---

**A Technical Report on CPT, SFT, and Preference Optimization**

*Version 1.0 — May 2026*

---

## Executive Summary

This report consolidates current best practices for tuning hyperparameters in large language model (LLM) training across three primary stages: continued pre-training (CPT), supervised fine-tuning (SFT), and preference optimization (DPO and its variants, with GRPO covered for completeness). It is intended as a working reference for engineers and researchers who already understand the basics of transformer training and need a single document that explains *why* recommended values are what they are, not just *what* they are.

The report has three main goals. First, it provides defensible default ranges for every major hyperparameter, with explicit notes on the conditions under which those defaults break. Second, it derives the mathematical reasoning behind several non-obvious choices — batch size scaling, AdamW's decoupled weight decay, the implicit regularization of small batches, and the origin of the DPO β parameter — so that readers can extrapolate sensibly to new situations. Third, it explicitly flags places where the literature is contested or where common framework defaults disagree with academic recommendations, so practitioners are not surprised when reality diverges from textbook advice.

A pre-training configuration checklist appears in Section 2 and is the recommended starting point for any new training run.

---

## 1. Scope and Conventions

This report covers the hyperparameters most commonly adjusted during LLM training:

- General hyperparameters relevant to all stages: learning rate, batch size, epochs, warmup, weight decay, gradient clipping, and sequence length.
- Stage-specific parameters: data mixture and loss masking for SFT; β and reference-model choice for DPO.
- Efficiency-oriented parameters: LoRA rank and scaling, QLoRA quantization.
- Optimizer internals: AdamW's β₁, β₂, ε, and weight-decay coupling.
- Distributed training and mixed precision considerations that interact with hyperparameter choice.

Throughout, the report uses η for learning rate, B for batch size, λ for weight decay coefficient, and β for the DPO KL-penalty coefficient (distinct from Adam's β₁, β₂, which are always written with subscripts). Equations use standard notation; π_θ denotes the policy being trained and π_ref the reference model.

The report deliberately does not cover full RLHF with PPO in depth, classical reinforcement learning algorithms, or pre-training from scratch at frontier scale. Where GRPO is discussed, it is in the context of how it relates to DPO in modern alignment pipelines, not as a comprehensive RL guide.

---

## 2. Pre-Training Configuration Checklist

Before launching any non-trivial training run, the following ten questions should have explicit answers. Most failed runs trace back to one of these being implicit or wrong.

1. **What is the equivalent batch size, and how is it constructed?** Document micro-batch size, gradient accumulation steps, and number of GPUs explicitly. Equivalent batch size should be in the 128–512 sample range for most fine-tuning, larger for pre-training.

2. **What is η_max, and where does it fall on the model-size and stage table?** A 7B model SFT run at 1e-4 is almost certainly too aggressive; a 1B model SFT run at 5e-6 is almost certainly too conservative.

3. **Is warmup specified in absolute steps or as a ratio?** If as a ratio, has the resulting absolute step count been verified to be large enough for Adam's second moment to stabilize (typically several hundred optimizer steps minimum)?

4. **Is loss masking correct for SFT?** Confirm that loss is computed only on assistant-response tokens, not on the prompt. This is the single most common silent failure in SFT.

5. **For DPO, what reference model is being used, and what is β?** A non-aligned base model as reference typically performs poorly; β = 0.1 is a reasonable starting point but not a default.

6. **What precision is the run using, and what is the optimizer-state precision?** Bf16 weights with fp32 optimizer state is the modern default; fp16 weights without loss scaling will silently corrupt training.

7. **What is the gradient clipping threshold, and is grad_norm being logged?** max_norm = 1.0 is the conventional default. Without grad_norm logs, gradient pathologies are nearly invisible until loss diverges.

8. **What is the sequence-packing strategy, and is cross-sample attention correctly masked?** Naive packing without document masks introduces silent contamination across unrelated samples.

9. **What evaluation sets exist for this run, and are they per-capability rather than aggregate?** A single train-loss curve cannot diagnose data-mixture problems.

10. **Is there a smaller-scale ablation (smaller model, fewer steps) that confirms the recipe before the full run?** For multi-day training, the cost of a one-hour ablation is essentially zero.

The remaining sections expand on each of these and the surrounding theory.

---

## 3. General Hyperparameters

### 3.1 Learning Rate

The learning rate is the single most consequential hyperparameter. For LLM fine-tuning the typical range is 1e-5 to 5e-5, with continued pre-training often using larger values. Detailed recommendations by model scale and training stage appear in Section 7. The standard schedule is linear warmup followed by cosine decay, discussed in Section 9.

### 3.2 Batch Size

Batch size affects gradient quality and training stability. In practice, the equivalent batch size is constructed via gradient accumulation across multiple micro-batches; typical equivalent values are 128–512 samples for fine-tuning. Too small a batch produces noisy gradients and unstable training; too large a batch reduces gradient noise to a point where optimization may converge to sharp minima with poor generalization. A formal analysis appears in Section 6.

### 3.3 Epochs

SFT typically runs 1–3 epochs; more frequently overfits given the small data volumes involved. CPT often runs less than one full pass over the data when the corpus is large. DPO usually runs 1–3 epochs. The underlying mechanism — why these are the right ranges and what determines them — is in Section 8.

### 3.4 Warmup

Warmup typically covers the first 1–5% of total training steps. The purpose is partly to protect pre-trained weights from large early updates, but the more fundamental reason is to give Adam's second-moment estimate v̂_t time to stabilize. A detailed treatment is in Section 11.

### 3.5 Weight Decay

There is no single universally correct value. Common ranges:

| Stage | Typical λ |
|---|---|
| Pre-training | 0.1 |
| Full-parameter SFT | 0.01–0.1 |
| LoRA fine-tuning | 0–0.01 |
| DPO | 0–0.01 |

Section 12 covers AdamW's decoupling of weight decay from L2 regularization and the framework-specific subtleties of which layers actually receive weight decay.

### 3.6 Gradient Clipping

A max_norm of 1.0 on the global gradient norm is the conventional default. The theoretical role and the interaction with Adam's adaptive scaling are covered in Section 14.

### 3.7 Maximum Sequence Length

Longer sequences are quadratically more expensive in attention. The right length depends on the actual length distribution of the task and on whether sequence packing is used. Section 13 covers padding versus packing, the cost of truncation, and the considerations around long-context training.

---

## 4. Stage-Specific Parameters

### 4.1 SFT-Specific Parameters

**Data mixture ratio.** When multiple data sources are combined, the relative proportion of each source typically has more impact on final quality than learning-rate tuning does. Section 15 covers data-mixture methodology in detail.

**Loss masking.** Loss is conventionally computed only on the assistant-response tokens, not on the prompt or system message. This is implemented as a label mask that sets prompt-token labels to a sentinel value (commonly -100 in PyTorch) so they are excluded from cross-entropy computation. Confirming this mask is correct is the single most important pre-flight check in SFT; runs that compute loss on the full sequence often appear to train normally but produce models that imitate prompt formatting rather than respond to it.

**Packing versus padding.** Packing concatenates multiple short samples into one fixed-length sequence to improve GPU utilization, but requires per-document attention masking to prevent cross-sample information leakage. Implementation details and pitfalls are in Section 13.3.

### 4.2 DPO-Specific Parameters

**β (KL penalty coefficient).** The most important DPO-specific parameter. It controls how far the policy is permitted to drift from the reference model. The DPO loss is:

$$\mathcal{L}_{\text{DPO}} = -\mathbb{E} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_\text{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_\text{ref}(y_l|x)} \right) \right]$$

Typical range: 0.01 to 0.5. Larger β produces more conservative updates that stay closer to the reference; smaller β allows more aggressive updates and risks reward hacking or distribution collapse on rejected outputs. A common starting point is β = 0.1, with diagnostic emphasis on whether the chosen-versus-rejected reward margin is widening over training. The mathematical derivation of β from the KL-constrained RLHF objective is in Section 17.

**Reference model.** A reference is typically the SFT-trained model. Using a base (non-instruction-tuned) model as reference generally underperforms because the reference distribution differs too much from the preference data distribution. Some recent work explores alternative reference strategies; the right choice depends on how preference data was collected.

**Chosen/rejected data quality.** Strictly not a hyperparameter, but the quality gap between chosen and rejected responses is more important than the dataset size. Pairs with too small a quality margin contribute mostly noise to the gradient. This interacts with example difficulty in a non-obvious way; see Section 16.

**Variant-specific parameters.** SimPO introduces a margin parameter γ that imposes a minimum gap (analogous to a hinge loss). IPO introduces its own regularization coefficient. These are not interchangeable with DPO's β; their semantics differ.

---

## 5. Tuning Priority and General Strategy

A reasonable priority ordering for SFT, with the caveat that priorities shift across stages:

> data quality > data volume > learning rate > β (for DPO) > batch size > everything else

For CPT, data volume and coverage often matter more than per-sample quality, since the goal is broad distributional coverage rather than behavioral imitation. For DPO, the quality of the chosen/rejected pairs and the choice of β dominate; learning rate matters less because preference signal is generally weak relative to SFT cross-entropy.

**Finding the right learning rate.** Run a learning-rate range test on a small data sample: hold all other parameters fixed and sweep η from very small (e.g., 1e-7) up to clearly unstable (e.g., 1e-2), observing the loss curve. Identify the η at which loss begins to fall rapidly (lower bound) and the η at which loss begins to oscillate or diverge (upper bound). Pick a value in the lower portion of this stable region as η_max. The range test should be conducted on early-stage data, not after partial convergence, since converged dynamics distort the curve.

**Detecting overfitting.** Train loss continuing to fall while evaluation loss rises is the classical signal. In LLMs, a more practical signal is degradation on held-out evaluation tasks specifically, particularly capabilities that were not heavily represented in the fine-tuning data (a phenomenon often called catastrophic forgetting).

**Tuning DPO β.** Watch the chosen/rejected reward margin. If β is too small, the model updates aggressively and the rejected reward may collapse toward zero (loss of diversity, a precursor to reward hacking). If β is too large, the model barely moves and the margin fails to widen.

**Validation strategy.** Run ablations at smaller scale (1.5B–3B parameters) to identify good hyperparameter ranges before committing to the full training run. Many decisions transfer across model sizes; some, especially around effective learning rate, do not — see Section 7.4.

---

## 6. Batch Size: Theory and Practice

### 6.1 Mini-batch Gradient Estimation

For a dataset of N samples, the true (full-batch) gradient is

$$g = \frac{1}{N} \sum_{i=1}^{N} \nabla_\theta \ell_i(\theta)$$

A mini-batch estimator using B samples is

$$\hat{g}_B = \frac{1}{B} \sum_{i \in \mathcal{B}} \nabla_\theta \ell_i(\theta)$$

This estimator is unbiased (its expectation equals g), and its covariance scales as

$$\text{Cov}(\hat{g}_B) = \frac{\Sigma}{B}$$

where Σ is the per-sample gradient covariance matrix. Taking the trace gives a scalar variance σ²/B with σ² = tr(Σ). All subsequent batch-size analysis derives from this fact.

### 6.2 Why Small Batches Are Unstable

Define a scalar gradient signal-to-noise ratio:

$$\text{SNR} = \frac{\|g\|^2}{\text{Var}(\hat{g}_B)} = \frac{B \cdot \|g\|^2}{\sigma^2}$$

Smaller B yields smaller SNR — each update direction is dominated by sampling noise rather than the true gradient. The parameter update decomposes as

$$\theta_{t+1} = \theta_t - \eta \hat{g}_B = \underbrace{\theta_t - \eta g}_{\text{signal}} + \underbrace{\eta(g - \hat{g}_B)}_{\text{noise}}$$

with noise variance η²σ²/B. As B shrinks, this noise grows and the loss curve oscillates or diverges.

### 6.3 Critical Batch Size

McCandlish et al. (2018) defined a critical batch size:

$$B_{\text{crit}} = \frac{\sigma^2}{\|g\|^2}$$

| Regime | Behavior |
|---|---|
| B ≪ B_crit | Noise-dominated; increasing B yields large gains |
| B ≈ B_crit | Efficient operating point |
| B ≫ B_crit | Signal-dominated; additional samples have diminishing returns |

In LLM training, B_crit tends to be small early (gradient norms are large) and grows toward convergence (gradient norms shrink). This motivates dynamic batch-size scaling schedules in some training pipelines.

### 6.4 Why Large Batches Can Hurt Generalization

Large-batch training can produce a generalization gap: training loss is low but evaluation loss is higher than what a small-batch run achieves. The phenomenon is empirically reproducible (Keskar et al., 2017) but its mechanism remains partly contested.

| Regime | Train loss | Test loss | Likely mechanism |
|---|---|---|---|
| Underfitting | high | high | Insufficient capacity or training |
| Healthy convergence | low | low | — |
| Large-batch generalization gap | low | typically higher | Smoother optimization trajectory; sharp-minimum hypothesis is one explanation among several |
| Classical overfitting | low | high | Memorization of training-set noise |

The presentations of "large-batch generalization gap" and "classical overfitting" look similar (low train, high test) but the underlying mechanisms differ. Classical overfitting is about model capacity exceeding the information content of the data and memorizing noise. Large-batch generalization gap is about optimization noise being insufficient to escape narrow loss-landscape regions. Both can occur in the same training run.

### 6.5 Sharp versus Flat Minima

Keskar et al. (2017) defined a sharpness measure:

$$\phi(\theta, \varepsilon) = \max_{\|\delta\| \leq \varepsilon} \left[ \mathcal{L}(\theta + \delta) - \mathcal{L}(\theta) \right]$$

Large φ indicates a sharp minimum (small parameter perturbations cause large loss increases); small φ indicates a flat minimum. The Hessian view: sharp minima have large Hessian eigenvalues, so any test-time distributional shift maps to a large change in loss.

The sharp-versus-flat framework is widely used as intuition. It is not unconditional — Dinh et al. (2017) showed that sharpness can be made arbitrarily large or small under reparameterization without changing the function the network computes. The practical message is that small-batch noise acts as an implicit regularizer:

$$\text{noise temperature} \propto \frac{\eta}{B}$$

Small batches inject enough noise to escape narrow basins; large batches descend smoothly into the nearest minimum, which may be sharp.

### 6.6 Linear Scaling Rule

Goyal et al. (2017) showed that, within a moderate range, increasing B can be compensated for by linearly increasing η:

$$\eta_{\text{new}} = \eta_{\text{base}} \cdot \frac{B_{\text{new}}}{B_{\text{base}}}$$

The rule holds because doubling B halves the gradient noise, and doubling η restores the per-step displacement statistics. The rule fails for B substantially above B_crit (typically beyond 2–4× B_crit): even with η scaled up, the noise temperature η/B is lower in absolute terms, and the implicit-regularization benefit cannot be recovered.

### 6.7 Practical Strategy

Start from a small batch warmup, increase to the target equivalent batch size, and scale learning rate linearly during the increase. Beyond the linear-scaling regime, expect generalization gap to appear regardless of η compensation; consider whether the additional throughput is worth the quality cost.

---

## 7. Learning Rate: Mechanisms and Tuning

### 7.1 Learning Rate as Step Size

For plain SGD,

$$\theta_{t+1} = \theta_t - \eta \cdot \nabla_\theta \mathcal{L}(\theta_t)$$

Too large η causes the parameter to overshoot minima and oscillate. As an order-of-magnitude bound, in a locally quadratic region with Hessian H, convergence requires η < 2/λ_max(H). LLM loss landscapes are far from quadratic, so this expression is a heuristic upper bound rather than a tight criterion; the practically stable upper bound is typically much smaller.

Too small η causes slow convergence and increases the risk of stalling near saddle points (where gradients are already small and shrinking the step further produces near-zero updates).

### 7.2 Adam's Effective Step Size

In practice almost all LLM training uses Adam or AdamW:

$$\theta_{t+1} = \theta_t - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

Because the update is normalized by √v̂_t, Adam's η is closer to a global trust-region scale than a direct multiplier on raw gradient magnitudes. This is why AdamW η values for LLMs are typically much smaller than equivalent SGD η values would be; the per-parameter normalization already fixes most of the scale issue.

### 7.3 Why Warmup Is Needed

Warmup is often described as protecting pre-trained weights from early disruption. The more fundamental reason is that Adam's second moment v̂_t is poorly estimated in the first several hundred steps. Until v̂_t accumulates enough samples, the per-parameter normalization is unreliable, and large η at this stage can produce wildly miscalibrated update magnitudes for parameters with low historical gradient variance. Linear warmup gives v̂_t time to stabilize before applying full step size.

### 7.4 Recommended Ranges by Model Scale and Stage

For full-parameter fine-tuning:

| Model size | Full-parameter η | LoRA η |
|---|---|---|
| < 1B | 5e-5 to 1e-4 | 1e-4 to 3e-4 |
| 1B–7B | 1e-5 to 5e-5 | 5e-5 to 2e-4 |
| 7B–30B | 5e-6 to 2e-5 | 2e-5 to 1e-4 |
| 70B+ | 1e-6 to 1e-5 | 1e-5 to 5e-5 |

The pattern: larger models are more sensitive to per-step parameter changes because the cumulative effect of layer-wise updates is amplified by depth. LoRA tolerates larger η because it modifies low-rank adapters rather than the original weights, providing a buffer.

By stage:

- **CPT.** Typically 1e-4 to 5e-5. Long training over large data needs enough η to drive substantial parameter movement, while remaining small enough to avoid catastrophic forgetting of pre-trained knowledge.
- **SFT.** Typically 1e-5 to 5e-5. The goal is behavioral refinement on top of pre-training, not large-scale parameter change.
- **DPO.** Typically lower than SFT for full-parameter runs; LoRA DPO can use larger values. Specific ranges vary widely with preference-data quality, model size, β, and reference-model choice.

### 7.5 Late-Stage Annealing

Annealing on a small high-quality dataset late in training has been reported to improve specific capabilities, particularly for smaller models on code and math benchmarks. The Llama 3 technical report describes a meaningful improvement at 8B on GSM8K and MATH from this approach, with much smaller effects at 405B. Annealing is not a universally beneficial technique; it should be evaluated empirically per model and benchmark.

### 7.6 Diagnostic Patterns

| Symptom | Likely cause | Adjustment |
|---|---|---|
| Loss flat throughout | η too small or warmup too long | Increase η, shorten warmup |
| Early descent then oscillation | η too large | Reduce η or add gradient clipping |
| Rapid descent then immediate stall | Saddle point with η too small | Slightly increase η or use momentum-based scheduler |
| Slow late-stage descent | Cosine decay too aggressive | Extend decay phase or raise η_min |
| Train loss low, eval degrading | η too large for the data size | Reduce η, increase weight decay |

The single most informative log field is the learning-rate value itself. Plotting it alongside loss confirms whether warmup is ramping correctly and whether decay is on schedule.

---

## 8. Training Epochs

### 8.1 Why SFT Overfits with Multiple Epochs

SFT datasets typically contain thousands to hundreds of thousands of examples. Model parameter counts are in the billions. The capacity-to-information ratio is heavily skewed: the model has more than enough room to memorize each sample exactly.

The first epoch teaches behavioral patterns. Subsequent epochs introduce no new information; gradients begin pushing parameters toward memorizing surface features of specific samples — phrasing tics, formatting noise, even labeling errors. The signature is train loss continuing to fall while eval loss rises, and the model's outputs starting to mirror the stylistic quirks of the training set.

A useful framing: SFT optimizes for behavioral alignment, not for minimum training loss. Driving loss below the alignment threshold trades alignment quality for over-fitting to annotator style.

### 8.2 Why Large Datasets Need Only One Pass

Two reasons. First, with enough data, every parameter has effectively seen a wide range of contexts in a single pass; additional repetition adds no new information. Second, the gradient-dilution effect: in any given batch, many distinct samples compete for gradient direction, forcing the model to learn cross-sample regularities rather than memorizing individual samples. Repeated epochs reduce dilution because the same samples reappear with similar gradient signal.

Unifying view:

> effective information ≈ data diversity × number of epochs

In SFT, diversity is small and additional epochs quickly saturate effective information, after which gradients begin encoding noise. In CPT, diversity is enormous and fewer than one epoch may already be sufficient.

A subtle but important point: training duration itself does not cause overfitting; the number of times the model has seen the same data does. A long run on continuously fresh data does not overfit, while a short run repeating the same small dataset can.

### 8.3 DPO Epochs

DPO typically runs 1–3 epochs. The reasoning is twofold: preference data is expensive and usually small; and the chosen-versus-rejected reward margin tends to compress as training proceeds, so additional epochs produce diminishing learning signal even before classical overfitting begins.

---

## 9. Learning Rate Schedulers

### 9.1 The Role of a Scheduler

A scheduler answers a single question: what step size should be used at each phase of training? A constant learning rate is suboptimal because the requirements differ across phases. Early training benefits from larger steps to traverse high-gradient regions quickly; mid-training benefits from steady steps that explore the basin; late training benefits from small steps to refine the final solution without oscillating around the minimum.

### 9.2 Common Schedulers

**Constant.** η_t = η. Simplest baseline, used for short experiments and rarely for production LLM training.

**Linear decay.** η_t = η_max · (1 − t/T). Linear from η_max to zero. Cosine is more common in long training, but linear with warmup remains effective for many short fine-tuning runs and should not be considered obsolete.

**Cosine decay.** The most widely used schedule for LLM training:

$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\pi \cdot \frac{t}{T}\right)\right)$$

Derivative is near-zero at both endpoints and largest in the middle, producing a slow-fast-slow rhythm. It is the conventional default but not the only viable choice.

**Cosine with restarts (SGDR).** Periodically resets η_max and restarts the cosine cycle, with cycle length growing geometrically. The restarts are intended to escape current minima and explore alternative basins. Useful when local minima are suspected to be problematic, but the loss curve becomes harder to monitor.

**Warmup + cosine decay.** The standard LLM recipe: linear ramp from 0 to η_max over T_w steps, followed by cosine decay to η_min over the remaining steps.

**Polynomial decay.** η_t = (η_max − η_min)(1 − t/T)^p + η_min. A generalization of linear (p=1); rarely used in LLM training because tuning p adds little over choosing between linear and cosine.

**Step decay / multi-step decay.** Multiplies η by γ at fixed step counts. The step changes introduce instability at each transition, which is why this scheduler is uncommon in LLM training despite being a staple of image classification.

**WSD (Warmup-Stable-Decay).** Used in MiniCPM and similar recipes:

```
η_max |     /‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\
      |    /                     \
η_min |___/                       \___
        warmup    stable          decay
```

The stable phase holds η_max constant; decay collapses η rapidly at the end. Key advantage: the stable phase can be truncated at any point to insert annealing on a different data mixture, without redesigning the entire schedule. Useful when the data curriculum is dynamic.

**Inverse square root.** The original Transformer scheduler: warmup followed by 1/√t decay. Has theoretical convergence guarantees for Adam but underperforms cosine empirically at large LLM scale. Still seen in some MoE training and academic reproductions.

### 9.3 Choosing a Scheduler

A simple decision flow:

1. Is the data curriculum dynamic, or do you anticipate inserting annealing phases? If yes, choose **WSD**.
2. Is this a short fine-tuning run (< 10K optimizer steps)? **Constant + warmup** or **linear + warmup** are reasonable; the cosine plateau provides little benefit at short horizons.
3. Otherwise, default to **warmup + cosine decay**.
4. Only consider **cosine with restarts** if you have evidence of problematic local minima and resources for the additional tuning of cycle length.

---

## 10. Steps, Warmup Ratio, and Gradient Accumulation

### 10.1 Warmup Ratio

The warmup ratio is T_w / T_total. The conventional 1–5% range is a starting point, not a rule. The real decision criterion is whether the absolute number of warmup steps is large enough for Adam's second moment to stabilize — typically several hundred optimizer steps minimum. Training recipes for very long runs (the Llama 3 reports 8000 steps of linear warmup, against a multi-trillion-token total) often specify warmup in absolute steps rather than as a ratio.

Larger models warrant longer warmup because the initial gradient distribution is further from steady state. SFT and DPO with only a few hundred to a few thousand total steps may benefit from a fixed absolute warmup (e.g., 50–100 steps) rather than a percentage that produces too few steps to be useful.

For continued training (CPT, SFT, DPO), warmup primarily protects the existing weights. For training from scratch, the initial weights are random, so the protective role is moot, but warmup is still needed for v̂_t stabilization.

### 10.2 Step Count and Data Volume

$$T_{\text{total}} = \frac{\text{total tokens}}{\text{batch size} \times \text{sequence length}}$$

Or by sample count:

$$T_{\text{total}} = \frac{\text{total samples} \times \text{epochs}}{\text{batch size}}$$

These expressions assume a vanilla data-parallel setup. With sequence parallelism, context parallelism, or expert parallelism, the relationship between forward passes and tokens-per-step changes; the formulas above should be adjusted for the specific parallelism configuration.

### 10.3 Gradient Accumulation and Step Semantics

Gradient accumulation splits a target equivalent batch into multiple micro-batches, accumulating gradients before a single optimizer step:

$$\text{equivalent batch size} = \text{micro batch size} \times \text{accumulation steps} \times \text{number of GPUs}$$

Two distinct notions of "step" arise:

- **Optimizer step:** one parameter update, occurring once per accumulation cycle. Schedulers, warmup ratios, and η decay all count optimizer steps.
- **Forward step:** one micro-batch forward pass, with no parameter update.

A configuration of `gradient_accumulation_steps=4` and `warmup_steps=100` means 100 optimizer steps of warmup, consuming 400 forward passes' worth of data.

### 10.4 Step Semantics Across Frameworks

Framework conventions are not uniform:

| Framework | Step terminology |
|---|---|
| Hugging Face Transformers / Accelerate | `step` and `global_step` typically denote optimizer steps |
| DeepSpeed | `global_step` is optimizer step; `micro_step` may denote forward steps |
| Megatron-LM | `iteration` denotes optimizer step; `consumed_samples` reflects data volume |

When in doubt, plot the `learning_rate` field against the step counter. If warmup is configured correctly, η will ramp linearly from 0 to η_max over the expected step range. This is more reliable than trusting the name of the step counter.

---

## 11. Weight Decay and AdamW

### 11.1 The Mechanism

Standard gradient descent updates only based on the loss gradient:

$$\theta_{t+1} = \theta_t - \eta \cdot \nabla_\theta \mathcal{L}(\theta_t)$$

Weight decay adds a parameter-shrinkage term:

$$\theta_{t+1} = (1 - \eta \lambda) \cdot \theta_t - \eta \cdot \nabla_\theta \mathcal{L}(\theta_t)$$

The factor (1 − ηλ) pulls each parameter slightly toward zero on every step.

### 11.2 Equivalence to L2 Regularization (Under SGD)

Adding an L2 penalty to the loss:

$$\mathcal{L}_{\text{reg}}(\theta) = \mathcal{L}(\theta) + \frac{\lambda}{2} \|\theta\|^2$$

Its gradient:

$$\nabla_\theta \mathcal{L}_{\text{reg}} = \nabla_\theta \mathcal{L} + \lambda \theta$$

Substituted into the update:

$$\theta_{t+1} = \theta_t - \eta(\nabla_\theta \mathcal{L} + \lambda \theta_t) = (1 - \eta\lambda)\theta_t - \eta \nabla_\theta \mathcal{L}$$

This is identical to weight decay under SGD, which is why the two terms are often used interchangeably. Under Adam they are not equivalent, which motivated AdamW (Section 11.5).

### 11.3 Why Weight Decay Helps Generalization

Overfitting tends to produce large-magnitude weights as the model fits training-set noise exactly. The L2 penalty applies upward gradient pressure on large weights, forcing a tradeoff between fitting the data and keeping parameters small. Geometrically, the regularizer constrains the optimum to lie within an ellipsoid centered at the origin.

### 11.4 What Causes Overfitting

Overfitting occurs when effective model capacity exceeds the constraints provided by the data. Common triggers in LLM training:

- Small data relative to model capacity (the SFT setting).
- Excessive epochs over the same data.
- Learning rate too large, pushing parameters far enough from the pre-trained region to enter sharp minima specific to the training set.
- Insufficient or zero weight decay.
- High-noise or low-quality data; the model has nothing to learn but the noise.
- Very large batch size, producing a generalization gap that resembles overfitting in symptom but not in mechanism (Section 6.4).

### 11.5 AdamW: Decoupling Weight Decay

Adam normalizes per-parameter updates by √v̂_t. If L2 regularization is implemented by adding λθ to the gradient, that term also gets normalized, with the consequence that different parameters effectively receive different regularization strength depending on their gradient histories. This is undesirable.

AdamW (Loshchilov & Hutter, 2019) decouples weight decay from the gradient:

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t} \qquad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
$$\theta_{t+1} = \theta_t - \eta \left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon} + \lambda\theta_t\right)$$

The λθ term is applied directly, not through the adaptive normalization. All LLM training should use AdamW, not Adam with L2.

**Bias correction.** With m_0 = v_0 = 0, the first few steps systematically underestimate the true moments. Dividing by (1 − β_1^t) and (1 − β_2^t) corrects this; the correction factors approach 1 as t grows.

**Adam's response to gradient outliers.** When g_t is anomalously large (say, magnitude M ≫ historical), both m_t ≈ (1−β_1)M and √v_t ≈ √(1−β_2) · M scale with M. The update magnitude is

$$\frac{\hat{m}_t}{\sqrt{\hat{v}_t}} \approx \frac{1-\beta_1}{\sqrt{1-\beta_2}} \approx 0.447 \quad \text{(for typical } \beta_1, \beta_2\text{)}$$

The magnitude M cancels: a single large gradient produces a bounded update, regardless of how large M is. This is why Adam is far more tolerant of single-step gradient anomalies than SGD, which would apply η · M directly.

### 11.6 Common β₁, β₂ Choices

PyTorch's AdamW defaults are betas = (0.9, 0.999). Several large LLM pre-training runs (Llama, GPT-NeoX, and others) have reported using betas = (0.9, 0.95). The motivation is that β_2 = 0.999 makes v_t adapt very slowly to changing gradient distributions, which can cause overly conservative steps in long training. β_2 = 0.95 makes the optimizer more responsive but slightly noisier. This is not the right choice for every training setup; it is a known recipe, not a default.

### 11.7 Per-Layer Weight Decay

**Layers conventionally excluded from weight decay:**

- **LayerNorm parameters (γ, β).** These directly determine the activation scale at each layer; shrinking them disrupts the careful activation calibration of pre-trained models and can cause training instability.
- **Bias terms.** Their absolute values represent baseline offsets and should not be forced toward zero.

**Embeddings.** The conventional academic recommendation is to exclude embeddings from weight decay, on the grounds that low-frequency tokens have weak gradient signal and weight decay can over-shrink their representations. However, both Hugging Face Trainer and Megatron-LM apply weight decay to embeddings by default. This is not a bug but a deliberate choice for pre-training, where the data volume is large enough that embedding weight decay prevents unbounded growth of high-frequency token magnitudes. For SFT, the difference is usually small in practice; learning rate and data quality matter more.

| Framework | Default behavior on embeddings | Selection criterion |
|---|---|---|
| Hugging Face Trainer | Includes in weight decay | Name does not contain `bias`; not a LayerNorm |
| Megatron-LM | Includes in weight decay | `ndim ≥ 2` |
| Custom PyTorch | User decides | Typically by name match |

**Reference implementation (PyTorch):**

```python
def get_param_groups(model, weight_decay):
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if (
            param.ndim <= 1
            or "bias" in name
            or "norm" in name.lower()
            or "embed" in name.lower()
        ):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
```

When using DeepSpeed with parameter groups defined this way, ZeRO-1/2/3 sharding does not affect group membership; each shard inherits the weight-decay setting of its parameter group. Specifying optimizer settings entirely through DeepSpeed config is simpler but precludes per-group weight decay; for production training, configure parameter groups in user code.

---

## 12. Maximum Sequence Length

### 12.1 Fixed-Length Constraints

GPU matrix operations require uniform tensor shapes within a batch. There are two common ways to enforce this:

- **Padding.** Short sequences are padded with `<pad>` tokens up to max_seq_len. Attention masks distinguish real tokens from padding, and loss computation excludes padding positions.
- **Packing.** Multiple short sequences are concatenated into a single sequence of length max_seq_len. Attention masking and position-id construction must isolate the constituent samples to prevent cross-sample information leakage.

Hugging Face workflows typically use a data collator for padding; packing is supported in TRL and similar wrappers but requires correct attention-mask construction. Megatron-LM is designed around packed tokens but the actual cross-sample isolation depends on data-pipeline configuration. DeepSpeed itself does not implement sample-level packing; that is the responsibility of the data layer above it.

### 12.2 Truncation and Padding Costs

**Truncation.** Tokens beyond max_seq_len are discarded. For long documents, summaries, or multi-turn conversations, this is a substantive information loss.

**Padding waste.** Short sequences padded to max_seq_len consume FLOPs on padding tokens. A sample whose actual length is 200 tokens, padded to max_seq_len = 4096, achieves only ~5% useful compute density.

**Packing efficiency.** Packing brings useful-token density close to 100% with negligible computational overhead. Large-scale pre-training almost universally uses packing rather than padding.

### 12.3 Cross-Sample Attention Leakage

Naive concatenation allows tokens in sample B to attend to tokens in sample A. The resulting model effectively trains on noise — pairs of unrelated contexts incorrectly treated as a single sequence.

The correct fix is a **document mask** (also called a sample mask): each subsequence's attention is restricted to tokens within its own boundaries. Resetting position IDs at sample boundaries is necessary but not sufficient; the attention mask itself must enforce the isolation. Common pitfalls include using a default `DataCollatorForSeq2Seq` without overriding mask behavior, or relying on position-id resets without confirming that the attention-mask construction respects them.

### 12.4 Long-Context Training

Extending max_seq_len faces three obstacles: memory (attention is O(n²) in the naive implementation), positional encoding generalization, and computational efficiency. Common techniques:

- **Position interpolation (PI).** Original RoPE degrades on positions beyond the training range. PI linearly interpolates out-of-range position IDs back into the trained range. YaRN refines this with frequency-dependent nonlinear scaling.
- **Flash Attention.** Reorganizes attention computation to use blocked SRAM access, reducing the *activation memory* of attention from O(n²) to O(n). Note that the *FLOPs* of attention remain O(n²) — Flash Attention is a memory optimization, not an algorithmic complexity reduction. Without it, very long contexts cannot fit on a single device's memory regardless of FLOPs available.
- **Long-context continued pre-training.** Extending from a short context (e.g., 4K) to a long context (e.g., 128K) typically requires explicit long-context training with appropriate data and possibly multi-stage extension. The Llama 3 series reaches 128K context; the precise progression varies by model size and is not a single recipe.
- **Sequence parallelism.** When a single sequence does not fit on one device's memory, the sequence is partitioned across devices and attention is computed with cross-device communication. Megatron's sequence parallelism and DeepSpeed-Ulysses are two implementations.

### 12.5 Summary

Choosing max_seq_len is a tradeoff among training efficiency, information completeness, and memory cost. The right value is determined by the actual length distribution of the data, not by the model's nominal maximum. Use packing for short-sample data; use long-context techniques for long-document data; do not pad excessively in either case.

---

## 13. Gradient Clipping

### 13.1 Definition

Global-norm gradient clipping computes the L2 norm of the entire flattened gradient vector and rescales the gradient if it exceeds a threshold:

$$\|g\|_2 = \sqrt{\sum_{i} g_i^2}, \qquad g \leftarrow g \cdot \min\left(1, \frac{\text{max\_norm}}{\|g\|_2}\right)$$

The rescaling preserves direction and shortens magnitude. Per-element clipping (truncating each gradient component independently) changes direction and is rarely used in LLM training.

### 13.2 The Gradient-Explosion Problem

In a deep network, backpropagation chains Jacobian matrices:

$$\frac{\partial \mathcal{L}}{\partial \theta_0} = \frac{\partial \mathcal{L}}{\partial h_L} \cdot \prod_{k=1}^{L} \frac{\partial h_k}{\partial h_{k-1}}$$

If the spectral norm of each layer Jacobian exceeds 1, the product grows exponentially with depth. In LLM training, gradient explosion is most commonly triggered by:

- Unstable early-training weights producing large activation variance.
- Anomalously long or rare-distribution sequences whose loss is unusually sensitive.
- Effective step size too large; the gradient direction is correct but the displacement overshoots.

The visible signature is loss jumping to NaN or to unreasonable values, ending the run.

### 13.3 Why Clipping Helps

**Stability.** Bounds the per-step parameter displacement, preventing one anomalous gradient from destroying weeks of training.

**Implicit step-size cap.** Each update satisfies

$$\|\Delta\theta\| = \eta \cdot \|g_{\text{clipped}}\| \leq \eta \cdot \text{max\_norm}$$

Clipping protects against effective-η spikes from gradient anomalies, even when nominal η is fixed.

**Targeted intervention.** Within-threshold gradients are not modified at all. Unlike reducing η (which shrinks all updates), clipping affects only outliers.

**Diagnostic value.** The `grad_norm` log field is one of the most informative training signals. Healthy training shows decreasing grad_norm over time, with occasional spikes that recover. Persistent grad_norm at or above max_norm indicates that either max_norm is set too low or η is too large.

### 13.4 max_norm Values

The conventional default is 1.0, which combined with η ≈ 1e-4 yields per-step displacement bounded by 1e-4. This is appropriate for most LLM training.

| Stage | Typical max_norm |
|---|---|
| Pre-training | 1.0 |
| SFT | 1.0 |
| DPO | 1.0 (sometimes 0.5; the preference signal is weaker, and large gradients shouldn't dominate updates) |
| LoRA | 1.0 (occasionally relaxed to higher values when LoRA parameter count is small and η is conservative) |

Diagnostic: if 95% of training steps have grad_norm well below max_norm, clipping is rarely active and the threshold is reasonable. If clipping is active in most steps, raise max_norm (allowing larger updates) only after confirming that doing so does not destabilize training.

### 13.5 Interaction with Adam

A natural question: given that Adam's adaptive normalization already bounds single-step updates (Section 11.5), is gradient clipping still needed? The empirical answer is yes, for two reasons that do not contradict the cancellation argument.

**fp16 numerical overflow.** In mixed-precision training, gradients are often stored in fp16 with maximum representable magnitude about 65504. A spike that exceeds this becomes inf or NaN before Adam's normalization can act. Gradient clipping, applied before the fp16 cast or with explicit overflow handling, is a necessary safeguard.

**Sustained anomalous gradients.** Adam's protection is per-step; the cancellation of M in m_t/√v_t reasoning assumes a single outlier. A run of consecutive large gradients moves both m_t and v_t together, and the per-step update magnitude of ~0.447 · η is not negligible if it persists for many steps. Gradient clipping caps the magnitude in either case.

So for AdamW-trained LLMs, gradient clipping is best understood as an engineering safety boundary rather than an algorithmic necessity, but it is a safety boundary that should not be omitted.

---

## 14. SFT Data Mixture

### 14.1 What Data Mixture Is

SFT typically combines multiple data sources: general dialogue, code, math, instruction following, domain-specific data. The data mixture is the relative proportion of each source in the training set. In practice, mixture ratio has more impact on final quality than learning-rate tuning does. Wrong learning rate causes unstable training curves; wrong mixture causes structural capability imbalances — strong on some tasks, regressed on others.

### 14.2 Static versus Dynamic Mixing

**Static mixing.** Sources are combined offline in target proportions, shuffled, and treated as one dataset. Simple to implement, but inflexible — adjusting proportions requires regenerating the dataset.

**Dynamic mixing.** Each batch is sampled from sources according to specified probabilities. This supports curriculum learning (high-quality data first, harder data later) and allows mid-training adjustments.

### 14.3 Why Mixture Matters

**Capability forgetting.** A capability whose training data is too thinly represented degrades over the training run. If code is only 2% of the data, a strong code-pretrained model can lose most of its code performance during SFT.

**Task-specific overfitting.** A heavily over-represented capability dominates. Training entirely on single-turn QA data degrades multi-turn dialogue capability.

**Quality amplification.** Noisy data scaled up contaminates the global gradient signal, degrading not only the noisy task's performance but unrelated capabilities through interference.

### 14.4 Power-Law Sampling

For source i with n_i samples, sampling probability is

$$p_i \propto n_i^{\alpha}$$

- α = 1: sample by raw count; large sources dominate.
- α = 0: uniform across sources, regardless of size.
- α ∈ (0, 1): a compromise. α ≈ 0.7 is a common empirical choice in LLM training, balancing coverage of small sources with exploitation of large ones.

The intuition: large sources offer diminishing marginal returns, so growing them linearly wastes capacity; small sources need upsampling to be visible to the model at all.

**Upsampling techniques.**
- **Repetition.** Duplicate small-source data several times. Effective up to roughly 3–5× repetition; beyond that, the model begins memorizing the repeated samples.
- **Synthetic augmentation.** Generate additional same-domain data using a stronger model. Requires careful quality filtering to avoid contaminating the small source with its own model's biases.

### 14.5 Tuning Methodology

**Use evaluation, not training loss.** A single train-loss curve mixes signal from all data sources and cannot distinguish per-task performance. Build per-capability evaluation sets (code, math, dialogue, etc.) and track them independently.

A workable iteration:

1. Set an initial mixture from prior experience or published recipes.
2. Run a small-scale ablation (1/10 data, 1/10 steps).
3. Measure per-capability evaluation metrics.
4. Increase the proportion of capabilities that regressed; decrease the proportion of capabilities that are over-represented.
5. After convergence on small scale, validate at full scale.

### 14.6 Common Mistakes

- **Treating train loss as the mixture-tuning signal.** Train loss may fall while specific capabilities silently regress.
- **Ignoring transfer effects.** High-quality reasoning data sometimes improves code performance. Per-task changes can be misleading without checking the global metric set.
- **Confusing repetition with new data.** Upsampled data is not new information; over-repetition causes overfitting on the upsampled source.
- **Adjusting mixture before cleaning.** Quality filtering should precede mixture tuning; otherwise the mixture compensates for noise it should remove.

### 14.7 Reference Mixtures

Approximate proportions for general-purpose SFT, drawn from public reports (Llama 3, Tülu 3, and similar):

| Data type | Approximate proportion |
|---|---|
| General dialogue / instruction following | 40–60% |
| Code | 15–25% |
| Math / reasoning | 10–20% |
| Long-document / RAG | 5–10% |
| Domain-specific (optional) | 5–15% |

These are starting points, not targets. The optimal mixture depends on the model's pre-training distribution: if the base model has heavy code coverage, less code data is needed at SFT.

For specialized domain models, mixtures can deviate substantially — domain data may be 60–80% of the SFT mix, with the remainder allocated to general capabilities to prevent regression. Such cases call for evaluation sets that explicitly cover both the target domain and the general capabilities being protected.

---

## 15. DPO Data Quality and the β Parameter

### 15.1 Two Dimensions of "Difference"

The advice "if chosen and rejected are too similar, the model can't learn anything" conflates two different notions of similarity that should be kept separate.

**Dimension 1: annotation quality (label-level difference).** Chosen and rejected differ only in surface phrasing while being substantively equivalent. The preference label is then mostly noise. The model is not failing to learn; it is learning the wrong thing — fitting random preferences in the data.

**Dimension 2: per-example difficulty for the current policy (gradient magnitude).** Define the implicit reward gap

$$\Delta r = \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}$$

The gradient of the DPO loss is proportional to σ(−Δr):

| Policy state | Δr | σ(−Δr) | Gradient magnitude |
|---|---|---|---|
| Strongly prefers chosen | ≫ 0 | ≈ 0 | Near zero (easy example, no signal) |
| Indifferent | ≈ 0 | ≈ 0.5 | Maximum (hard example) |
| Prefers rejected | ≪ 0 | ≈ 1 | Maximum (very hard, large update) |

So even with high-quality labels, examples where the policy already strongly prefers the chosen response provide essentially no gradient signal.

### 15.2 The Ideal Pair

The most informative training pair has both:
- Large label-level quality difference (annotation is reliable and meaningful).
- Small implicit reward gap on the current policy (the policy has not yet learned to prefer chosen).

The two failure modes:
- High annotation noise → directionally random gradients → policy learns the noise.
- High annotation quality but already-learned by the policy → near-zero gradients → wasted compute.

This is why **online DPO** and **iterative DPO** were developed. Both dynamically generate hard examples by sampling from the current policy and selecting pairs with small Δr. Online DPO does this continuously during training; iterative DPO does it in stages, regenerating the dataset between epochs. Static DPO datasets often spend much of their compute on already-easy examples.

### 15.3 The Origin of β

DPO's β comes from a KL-constrained reward maximization objective:

$$\max_{\pi_\theta} \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi_\theta} \left[ r(x, y) \right] - \beta \, D_{\text{KL}}\!\left[\pi_\theta(y|x) \,\|\, \pi_{\text{ref}}(y|x)\right]$$

This is "maximize reward but stay close to the reference." It admits a closed-form optimal solution:

$$\pi^*(y|x) = \frac{1}{Z(x)}\, \pi_{\text{ref}}(y|x) \exp\!\left(\frac{r(x,y)}{\beta}\right)$$

Solving for r:

$$r(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)$$

Substituting into the Bradley-Terry preference model p(y_w ≻ y_l) = σ(r(x, y_w) − r(x, y_l)) cancels the β log Z(x) terms, yielding the DPO probability:

$$p(y_w \succ y_l \mid x) = \sigma\!\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)$$

The DPO loss is the negative log-likelihood of this probability over the preference dataset.

### 15.4 The Practical Effect of β

β controls how far the policy can drift from the reference:

- **Small β.** Weak KL penalty; aggressive updates. The model can rapidly increase the chosen probability, but the rejected probability may collapse toward zero, eliminating diversity. Reward hacking — increasing implicit reward without actually improving response quality — becomes more likely.
- **Large β.** Strong KL penalty; conservative updates. The model stays close to the reference; the preference signal is largely suppressed.

Typical values: 0.01–0.5, with 0.1 a common starting point.

### 15.5 Why β Is DPO-Specific

PPO-based RLHF also has a KL penalty, but it is implemented as an explicit `kl_coef` term added to the reward signal:

$$r_{\text{shaped}}(x,y) = r_\phi(x,y) - \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$$

The β coefficients in DPO and PPO refer to the same KL constraint, but in DPO it is folded into the loss function structure rather than added as a separate term. In DPO, β is not an external regularizer — it determines the shape of the loss itself. Tuning β in DPO is tuning how much the alignment step is allowed to change the policy.

---

## 16. GRPO and Modern Preference Optimization

### 16.1 Differentiation, Not Replacement

DPO and GRPO are not interchangeable. DPO remains the standard for general alignment goals — helpfulness, harmlessness, style preference — that lack objectively verifiable answers. GRPO and broader online-RL methods have become dominant for reasoning capabilities — math, code — where correctness can be checked automatically. The DeepSeek-R1 release marked a clear shift in this direction for reasoning-focused models. For general-purpose alignment, DPO and its variants (SimPO, IPO) remain the more practical choice.

### 16.2 GRPO's Mechanism

For each prompt, GRPO samples G outputs and computes within-group standardized advantages:

$$A_i = \frac{r_i - \text{mean}(r_1, \ldots, r_G)}{\text{std}(r_1, \ldots, r_G)}$$

These advantages drive a PPO-style clipped objective. The critical simplification relative to PPO is that the group mean replaces a learned critic network, eliminating the need to train and maintain a separate value model.

The natural fit is **verifiable rewards**: math correctness, code execution success. These reward signals are objective, do not require human labeling, and do not need a learned reward model. DPO, in contrast, depends on labeled preference pairs, which are expensive to collect for complex reasoning tasks where annotators may not be able to judge correctness reliably.

### 16.3 DPO's Continuing Role

DPO is offline: no online sampling, no rollouts, no value model. Training cost is much lower. For style, safety, and instruction-following tasks — where there is no objective reward function and human preference is the only signal available — DPO remains more practical and resource-efficient than GRPO.

### 16.4 Current Division of Labor

| Goal | Common method |
|---|---|
| General dialogue style, safety alignment | DPO, SimPO, IPO |
| Math, code, reasoning | GRPO, online PPO |
| Combined goals | Multi-stage: SFT → DPO → GRPO, or online DPO |

The "SFT → GRPO" pipeline has become the de facto standard for reasoning-focused models since DeepSeek-R1. For general dialogue quality without strong reasoning emphasis, DPO remains the appropriate choice.

---

## 17. LoRA and QLoRA

### 17.1 LoRA Formulation

LoRA freezes the original weight W_0 ∈ ℝ^(d×k) and adds a low-rank adapter:

$$W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} B A$$

with B ∈ ℝ^(d×r), A ∈ ℝ^(r×k), and r ≪ min(d, k). The forward pass is:

$$h = W_0 x + \frac{\alpha}{r} B A x$$

The standard initialization (PEFT and most implementations) uses Gaussian initialization for A and zeros for B, ensuring ΔW = 0 at the start of training. Some implementations swap which matrix is zero-initialized; the result is the same starting point, only the gradient flow direction differs initially.

### 17.2 lora_r (Rank)

The rank r determines trainable parameter count. For one weight matrix:

$$\text{LoRA params} = r(d + k), \qquad \text{compression ratio} \approx \frac{2r}{\min(d, k)}$$

For d = k = 4096 and r = 8, the compression ratio is about 0.4%.

- **Small r:** few parameters; risk of underfitting; fast and memory-efficient.
- **Large r:** more representational capacity, approaching full fine-tuning. Empirically, returns diminish past r ≈ 64–128.

### 17.3 lora_alpha (Scaling Factor)

α scales the LoRA contribution through the ratio α/r:

$$\Delta W_{\text{effective}} = \frac{\alpha}{r} B A$$

The effective learning rate for LoRA parameters is α/r times the nominal η. Common conventions:

- **α = r** (ratio = 1): conservative, often used for CPT where the model is already learning broadly.
- **α = 2r** (ratio = 2): more aggressive, common in SFT.
- **Fixed α** (e.g., 16 or 32) while varying r: increasing r effectively decreases the LoRA learning rate.

### 17.4 lora_dropout

Dropout applied to the LoRA branch input:

$$h = W_0 x + \frac{\alpha}{r} B \cdot \text{Dropout}(Ax,\; p)$$

Functions as regularization on the adapter. Use 0 for large datasets (CPT); 0.05–0.1 for small-data SFT and DPO.

### 17.5 QLoRA Additions

QLoRA extends LoRA with three engineering changes:

1. **4-bit NormalFloat (NF4) quantization** of the base model weights. NF4 uses normal-distribution quantile points rather than linear spacing, which better matches LLM weight distributions. The base weights are stored in 4-bit and dequantized to bf16 on demand during forward passes; LoRA matrices A and B remain in bf16.

2. **Double quantization** — quantizing the per-block quantization constants themselves. Saves approximately 0.37 bits per parameter.

3. **Paged optimizer** — uses NVIDIA unified memory to swap optimizer state between GPU and CPU when GPU memory is exhausted, preventing peak-memory OOM.

Approximate memory comparison for a 7B model:

| Method | Base model memory | Total memory (rough) |
|---|---|---|
| Full bf16 fine-tuning | ~14 GB | ~60+ GB (depends on optimizer-state precision) |
| LoRA on bf16 base | ~14 GB | ~18 GB |
| QLoRA on 4-bit base | ~3.5 GB | ~6 GB |

Total memory varies with optimizer-state precision (fp32 master copies, bf16 master copies, paged state, etc.), gradient accumulation factor, and activation memory; the figures above assume a typical training-time configuration with fp32 optimizer state.

The quality cost of QLoRA: NF4 quantization introduces some precision loss. For SFT on small data, the impact is usually negligible; for long CPT runs the degradation is observable. When resources allow, prefer LoRA over QLoRA.

### 17.6 Recommended LoRA Configurations

By model size:

| Size | lora_r | lora_alpha | lora_dropout | Method |
|---|---|---|---|---|
| 1B–3B | 4–16 | r or 2r | 0.05 | LoRA |
| 7B–13B | 8–32 | 16–64 | 0.05 | LoRA / QLoRA |
| 30B–70B | 16–64 | 32–128 | 0.05 | LoRA (multi-GPU) / QLoRA (single GPU) |
| 100B+ | 32–64 | 64–128 | 0 | QLoRA usually required |

By training stage:

- **CPT.** Inject substantial new knowledge. r = 32–64, α = r (conservative ratio), dropout = 0, target attention + full MLP. Beyond a certain data scale, full fine-tuning is preferable; LoRA's low-rank constraint becomes the bottleneck.
- **SFT.** r = 8–32, α = 2r, dropout = 0.05, target Q/K/V/O + MLP.
- **DPO.** Smaller refinement. r = 4–16, α = r, dropout = 0.05; targeting only Q/V is sometimes sufficient.
- **PPO actor.** r = 8–16, α = 2r, dropout = 0.05. Critic models are often full-parameter or use a larger r.

### 17.7 Target Modules

A transformer layer's linear projections:

```
Attention:  Q, K, V, O
MLP/FFN:    gate_proj, up_proj, down_proj  (LLaMA-style)
            fc1, fc2                       (GPT-style)
Embedding:  token embedding
LM Head:    output projection
```

**Hugging Face PEFT.** Configure via `target_modules`. The recommended set for capability injection is:

```python
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]
```

Embedding and LM head are not LoRA-adapted by default; full-parameter training of these layers can be enabled via `modules_to_save` if desired.

**DeepSpeed.** No native LoRA implementation; relies on PEFT. ZeRO-3 with LoRA requires careful parameter aggregation (some frameworks like LLaMA-Factory provide handling); ZeRO-2 is simpler because LoRA parameters are not sharded.

**Megatron-LM.** Has its own LoRA implementation, defaulting to attention Q/K/V/O only and excluding MLP. The reason is that Megatron's MLP uses column-parallel + row-parallel splits; inserting LoRA across these tensor-parallel boundaries requires additional infrastructure that the default does not include.

| Framework | Default target | MLP support |
|---|---|---|
| Hugging Face PEFT | Q/V (configurable) | Yes, manual configuration |
| DeepSpeed | Inherits from PEFT | Inherits from PEFT |
| Megatron-LM | Q/K/V/O | Not by default; requires custom code |

In practice, covering attention + full MLP outperforms attention-only LoRA, particularly for CPT and larger r.

---

## 18. Mixed Precision Training

### 18.1 Common Precision Formats

| Format | Bits | Mantissa | Exponent range | Notes |
|---|---|---|---|---|
| fp32 | 32 | 23 | wide | Default, expensive, rarely used for activations |
| bf16 | 16 | 7 | same as fp32 | Modern LLM default; same dynamic range as fp32 |
| fp16 | 16 | 10 | narrow | More mantissa precision but limited range; prone to overflow |
| fp8 (E4M3, E5M2) | 8 | 3 / 2 | narrower | Emerging; requires per-tensor scaling |

The key distinction between bf16 and fp16: bf16 keeps fp32's exponent range, so it does not overflow on typical LLM gradients; fp16 has narrower range and can produce inf/NaN on large gradients. This is why bf16 has become the default for LLM training when hardware supports it (Ampere generation onward).

### 18.2 Master Weight and Optimizer State

A standard mixed-precision recipe:

- **Weights stored in bf16** for forward and backward passes.
- **Master weights and optimizer state (m, v) in fp32**, updated from accumulated gradients.
- **Gradients in bf16**, converted to fp32 for the optimizer update.

Memory implications: a 7B model uses ~14 GB for bf16 weights, ~28 GB for fp32 master weights, and ~28 GB for AdamW's two fp32 moments — so optimizer state alone can exceed model size by 4×. ZeRO-1 and similar state sharding exist primarily to manage this.

Bf16-only optimizer state (no fp32 master copies) saves substantial memory and is increasingly used in production training, though it requires careful loss-scaling and gradient-accumulation setup to avoid precision loss in the moment estimates.

### 18.3 Loss Scaling

Required for fp16, optional for bf16. Small gradients in fp16 underflow to zero. Loss scaling multiplies the loss by a large factor before backpropagation, then divides gradients by the same factor before the optimizer step:

$$g_{\text{scaled}} = \nabla(\text{scale} \cdot \mathcal{L}), \qquad g = g_{\text{scaled}} / \text{scale}$$

Dynamic loss scaling adjusts the factor based on overflow occurrence: increases when no overflow has been seen for many steps, halves on detected overflow. Bf16 typically does not need this because its exponent range covers the gradient magnitudes encountered in LLM training.

### 18.4 fp8 Training

H100-class hardware supports fp8 with two formats: E4M3 (more mantissa, less range, used for forward) and E5M2 (more range, less mantissa, used for backward gradients). Per-tensor scaling factors are required to map values into the limited fp8 range. Frameworks like Transformer Engine and FP8-LM handle this automatically. Gains are roughly 2× throughput over bf16 with carefully tuned setups; quality preservation depends heavily on per-tensor scaling correctness.

### 18.5 Mixed Precision and Hyperparameter Choice

- Gradient clipping is more important under fp16 (overflow protection) than under bf16.
- Learning rate is largely format-independent for bf16 and fp16; for fp8, slightly larger η can sometimes compensate for noise introduced by quantization.
- AdamW's ε may need to be raised (e.g., from 1e-8 to 1e-6) under fp8 to avoid division-by-near-zero issues in the per-parameter normalization.

---

## 19. Distributed Training Parameters

This section briefly summarizes distributed-training mechanisms relevant to hyperparameter tuning. A full treatment of distributed systems is beyond scope.

### 19.1 Parallelism Types

- **Data parallelism (DP).** Replicates the model on each device; each replica processes a different micro-batch; gradients are all-reduced across devices. Simple, scales by adding devices; bounded by per-device model size.
- **Tensor parallelism (TP).** Splits individual layers (matrix multiplies) across devices. Reduces per-device memory but adds intra-layer communication; typically used within a single node (NVLink) for bandwidth reasons.
- **Pipeline parallelism (PP).** Splits the model layer-wise across devices; each micro-batch flows through the pipeline. Reduces per-device memory; requires careful micro-batch scheduling to avoid bubble overhead.
- **Sequence parallelism (SP).** Splits the sequence dimension across devices; used for very long contexts where attention memory is the bottleneck even with Flash Attention.
- **Expert parallelism (EP).** Splits MoE experts across devices; relevant only for MoE architectures.
- **Context parallelism.** Generalization of sequence parallelism with overlapping schemes.

Real production training combines several of these. A 70B-parameter long-context training run might use TP=8 within a node, PP=4 across nodes, DP across the remaining devices, and SP for context length above 32K.

### 19.2 ZeRO and Optimizer-State Sharding

ZeRO (DeepSpeed) shards memory consumed by training:

- **ZeRO-1.** Partitions optimizer state across devices.
- **ZeRO-2.** Adds gradient partitioning.
- **ZeRO-3.** Adds parameter partitioning; each device holds only a shard of parameters at any time.

ZeRO-3 has the largest memory savings but the highest communication cost. ZeRO-2 is a common operating point for most training. FSDP (PyTorch's native equivalent) provides similar functionality.

### 19.3 Hyperparameter Implications

- **Effective batch size.** With DP, equivalent batch size scales with the number of devices; the linear-scaling rule applies for moderate scaling but breaks down at large device counts (Section 6.6).
- **Learning rate.** Should be tuned at the equivalent batch size that will be used in production; do not tune at single-device batch and assume scaling up the device count preserves quality without adjustment.
- **Gradient accumulation.** Interacts with parallelism: micro-batch × accumulation × DP size = equivalent batch. Adjust accumulation steps as DP changes to keep equivalent batch constant.
- **Warmup steps.** Specified in optimizer steps, not forward steps. With more aggressive parallelism, the same number of optimizer steps consumes more data; warmup duration in tokens scales accordingly.
- **Communication-bound training.** When gradient all-reduce dominates, smaller models with high TP may be inefficient; profile communication versus compute to identify bottlenecks before tuning hyperparameters.

---

## 20. Closing Notes

LLM hyperparameter tuning has matured significantly since the early scaling-law era. Most decisions now have clear default values that work across a wide range of conditions; the ranges in this document reflect that consensus. The remaining tuning effort is best directed at:

1. **Data quality and mixture.** This dominates final quality more than any single hyperparameter.
2. **Effective batch size and learning rate, jointly.** These interact and should be tuned together via learning-rate range tests at the actual production batch size.
3. **Stage-specific defaults.** CPT, SFT, and DPO have different recipes; do not transfer them between stages.
4. **Diagnostics.** The single most underused tool is logging — grad_norm, learning rate, per-capability evaluation, and gradient histograms reveal more than any amount of post-hoc analysis.

Hyperparameter tuning is rarely the bottleneck on a well-run training pipeline. Data quality, evaluation rigor, and disciplined ablation usually matter more. This document provides defensible defaults and the reasoning behind them so that tuning effort can be spent where it has the most leverage.

---

## References

- Dinh, L., Pascanu, R., Bengio, S., & Bengio, Y. (2017). *Sharp Minima Can Generalize for Deep Nets.* ICML 2017.
- Dubey, A., et al. (2024). *The Llama 3 Herd of Models.* arXiv:2407.21783.
- Goyal, P., et al. (2017). *Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour.* arXiv:1706.02677.
- Hu, E. J., et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* arXiv:2106.09685.
- Hu, S., et al. (2024). *MiniCPM: Unleashing the Potential of Small Language Models with Scalable Training Strategies.* arXiv:2404.06395.
- Keskar, N. S., et al. (2017). *On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima.* ICLR 2017.
- Loshchilov, I., & Hutter, F. (2017). *SGDR: Stochastic Gradient Descent with Warm Restarts.* ICLR 2017.
- Loshchilov, I., & Hutter, F. (2019). *Decoupled Weight Decay Regularization.* ICLR 2019.
- McCandlish, S., et al. (2018). *An Empirical Model of Large-Batch Training.* arXiv:1812.06162.
- Rafailov, R., et al. (2023). *Direct Preference Optimization: Your Language Model is Secretly a Reward Model.* NeurIPS 2023.
- Shao, Z., et al. (2024). *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models.* arXiv:2402.03300. (GRPO origin.)
- Vaswani, A., et al. (2017). *Attention Is All You Need.* NeurIPS 2017.
