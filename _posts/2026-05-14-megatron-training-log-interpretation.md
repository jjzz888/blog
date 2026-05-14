---
layout: post
title: "Megatron Training Log Interpretation"
date: 2026-05-14 00:00:00 +0800
categories: [llm, machine-learning, distributed-training]
---

## 1. Example of a Megatron Training Log

Below is a more realistic Megatron training log example. It includes:

- distributed initialization information
- argument dumps
- dataset index, sample index, and shuffle index
- fixed-format iteration lines during training
- periodic validation blocks
- periodic checkpoint saving

```text
[2026-05-14 08:59:51] launching training job
[2026-05-14 08:59:51] CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
[2026-05-14 08:59:51] MASTER_ADDR=10.10.0.12
[2026-05-14 08:59:51] MASTER_PORT=6000
[2026-05-14 08:59:51] NNODES=1
[2026-05-14 08:59:51] NODE_RANK=0
[2026-05-14 08:59:51] WORLD_SIZE=8
[2026-05-14 08:59:52] initializing torch distributed ...
[2026-05-14 08:59:53] global rank 0 initialized
[2026-05-14 08:59:53] global rank 1 initialized
[2026-05-14 08:59:53] global rank 2 initialized
[2026-05-14 08:59:53] global rank 3 initialized
[2026-05-14 08:59:53] global rank 4 initialized
[2026-05-14 08:59:53] global rank 5 initialized
[2026-05-14 08:59:53] global rank 6 initialized
[2026-05-14 08:59:53] global rank 7 initialized
[NCCL INFO] Bootstrap : Using eth0:10.10.0.12<0>
[NCCL INFO] cudaDriverVersion 12040
[NCCL INFO] ncclCommInitRank rank 0 nranks 8 cudaDev 0
[NCCL INFO] ncclCommInitRank rank 1 nranks 8 cudaDev 1
[NCCL INFO] ncclCommInitRank rank 2 nranks 8 cudaDev 2
[NCCL INFO] ncclCommInitRank rank 3 nranks 8 cudaDev 3
[NCCL INFO] ncclCommInitRank rank 4 nranks 8 cudaDev 4
[NCCL INFO] ncclCommInitRank rank 5 nranks 8 cudaDev 5
[NCCL INFO] ncclCommInitRank rank 6 nranks 8 cudaDev 6
[NCCL INFO] ncclCommInitRank rank 7 nranks 8 cudaDev 7
[NCCL INFO] Connected all rings
[NCCL INFO] Connected all trees
using world size: 8, data-parallel-size: 2, tensor-model-parallel-size: 2, pipeline-model-parallel-size: 2
setting global batch size to 512
using torch.bfloat16 for parameters ...
accumulate all-reduce gradients in fp32 for bfloat16 data type.
------------------------ arguments ------------------------
  num_layers ...................................... 32
  hidden_size ..................................... 4096
  num_attention_heads ............................. 32
  seq_length ...................................... 4096
  max_position_embeddings ......................... 4096
  micro_batch_size ................................ 4
  global_batch_size ............................... 512
  train_iters ..................................... 20000
  lr .............................................. 1.500000e-04
  min_lr .......................................... 1.000000e-05
  lr_decay_style .................................. cosine
  lr_warmup_iters ................................. 200
  weight_decay .................................... 1.000000e-01
  clip_grad ....................................... 1.0
  init_method_std ................................. 0.006
  fp16 ............................................ False
  bf16 ............................................ True
  tensor_model_parallel_size ...................... 2
  pipeline_model_parallel_size .................... 2
  save ............................................ /checkpoints/gpt-run01
  load ............................................ None
  data_path ....................................... /data/corpus_text_document
  split ........................................... 969,30,1
-----------------------------------------------------------
> building GPT model ...
 > number of parameters on (tensor, pipeline) model parallel rank (0, 0): 6.74B
 > number of parameters on (tensor, pipeline) model parallel rank (1, 0): 6.74B
 > number of parameters on (tensor, pipeline) model parallel rank (0, 1): 6.73B
 > number of parameters on (tensor, pipeline) model parallel rank (1, 1): 6.73B
> building train, validation, and test datasets ...
 > dataset split:
    train:
     document indices in [0, 968)
    validation:
     document indices in [968, 998)
    test:
     document indices in [998, 999)
 > loading indexed mapping from /data/corpus_text_document_train_indexmap_4096ns_1234s_decoder_packed.npy
 > loading sample index from /data/corpus_text_document_train_sample_index.npy
 > loading shuffle index from /data/corpus_text_document_train_shuffle_index.npy
 > total number of samples: 28573440
 > total number of epochs: 3.58
 > training ...
 iteration        1/20000 | consumed samples:          512 | elapsed time per iteration (ms): 2418.7 | learning rate: 7.500E-07 | global batch size:   512 | lm loss: 10.812457 | loss scale: 1.0 | grad norm: 3.781 | skipped iterations:   0 | nan iterations:   0
 iteration        2/20000 | consumed samples:         1024 | elapsed time per iteration (ms): 2334.1 | learning rate: 1.500E-06 | global batch size:   512 | lm loss: 10.562930 | loss scale: 1.0 | grad norm: 3.547 | skipped iterations:   0 | nan iterations:   0
 iteration        3/20000 | consumed samples:         1536 | elapsed time per iteration (ms): 2287.3 | learning rate: 2.250E-06 | global batch size:   512 | lm loss: 10.331184 | loss scale: 1.0 | grad norm: 3.426 | skipped iterations:   0 | nan iterations:   0
 iteration       10/20000 | consumed samples:         5120 | elapsed time per iteration (ms): 2210.6 | learning rate: 7.500E-06 | global batch size:   512 | lm loss:  9.114520 | loss scale: 1.0 | grad norm: 2.892 | skipped iterations:   0 | nan iterations:   0
 iteration       20/20000 | consumed samples:        10240 | elapsed time per iteration (ms): 2179.2 | learning rate: 1.500E-05 | global batch size:   512 | lm loss:  8.321774 | loss scale: 1.0 | grad norm: 2.564 | skipped iterations:   0 | nan iterations:   0
 iteration       30/20000 | consumed samples:        15360 | elapsed time per iteration (ms): 2154.4 | learning rate: 2.250E-05 | global batch size:   512 | lm loss:  7.845192 | loss scale: 1.0 | grad norm: 2.337 | skipped iterations:   0 | nan iterations:   0
 iteration       40/20000 | consumed samples:        20480 | elapsed time per iteration (ms): 2138.5 | learning rate: 3.000E-05 | global batch size:   512 | lm loss:  7.381922 | loss scale: 1.0 | grad norm: 2.149 | skipped iterations:   0 | nan iterations:   0
 iteration       50/20000 | consumed samples:        25600 | elapsed time per iteration (ms): 2127.8 | learning rate: 3.750E-05 | global batch size:   512 | lm loss:  7.012334 | loss scale: 1.0 | grad norm: 2.031 | skipped iterations:   0 | nan iterations:   0
 ------------------------------------------------------------------------------------------
  validation loss at iteration 50 | lm loss value: 6.987203E+00 | lm loss PPL: 1.084918E+03
 ------------------------------------------------------------------------------------------
 iteration       60/20000 | consumed samples:        30720 | elapsed time per iteration (ms): 2119.6 | learning rate: 4.500E-05 | global batch size:   512 | lm loss:  6.771202 | loss scale: 1.0 | grad norm: 1.962 | skipped iterations:   0 | nan iterations:   0
 iteration       70/20000 | consumed samples:        35840 | elapsed time per iteration (ms): 2111.9 | learning rate: 5.250E-05 | global batch size:   512 | lm loss:  6.552019 | loss scale: 1.0 | grad norm: 1.854 | skipped iterations:   0 | nan iterations:   0
 iteration       80/20000 | consumed samples:        40960 | elapsed time per iteration (ms): 2103.5 | learning rate: 6.000E-05 | global batch size:   512 | lm loss:  6.341775 | loss scale: 1.0 | grad norm: 1.792 | skipped iterations:   0 | nan iterations:   0
 iteration       90/20000 | consumed samples:        46080 | elapsed time per iteration (ms): 2098.8 | learning rate: 6.750E-05 | global batch size:   512 | lm loss:  6.178900 | loss scale: 1.0 | grad norm: 1.734 | skipped iterations:   0 | nan iterations:   0
 iteration      100/20000 | consumed samples:        51200 | elapsed time per iteration (ms): 2094.7 | learning rate: 7.500E-05 | global batch size:   512 | lm loss:  6.042661 | loss scale: 1.0 | grad norm: 1.688 | skipped iterations:   0 | nan iterations:   0
[Rank 0] saving checkpoint at iteration      100 to /checkpoints/gpt-run01
[Rank 0]   successfully saved checkpoint from iteration 100 to /checkpoints/gpt-run01
 iteration      110/20000 | consumed samples:        56320 | elapsed time per iteration (ms): 2088.0 | learning rate: 8.250E-05 | global batch size:   512 | lm loss:  5.901274 | loss scale: 1.0 | grad norm: 1.621 | skipped iterations:   0 | nan iterations:   0
 iteration      120/20000 | consumed samples:        61440 | elapsed time per iteration (ms): 2085.6 | learning rate: 9.000E-05 | global batch size:   512 | lm loss:  5.792509 | loss scale: 1.0 | grad norm: 1.577 | skipped iterations:   0 | nan iterations:   0
 iteration      130/20000 | consumed samples:        66560 | elapsed time per iteration (ms): 2081.1 | learning rate: 9.750E-05 | global batch size:   512 | lm loss:  5.694131 | loss scale: 1.0 | grad norm: 1.546 | skipped iterations:   0 | nan iterations:   0
 iteration      140/20000 | consumed samples:        71680 | elapsed time per iteration (ms): 2076.9 | learning rate: 1.050E-04 | global batch size:   512 | lm loss:  5.611074 | loss scale: 1.0 | grad norm: 1.503 | skipped iterations:   0 | nan iterations:   0
 iteration      150/20000 | consumed samples:        76800 | elapsed time per iteration (ms): 2075.3 | learning rate: 1.125E-04 | global batch size:   512 | lm loss:  5.522418 | loss scale: 1.0 | grad norm: 1.472 | skipped iterations:   0 | nan iterations:   0
 ------------------------------------------------------------------------------------------
  validation loss at iteration 150 | lm loss value: 5.487203E+00 | lm loss PPL: 2.417918E+02
 ------------------------------------------------------------------------------------------
 iteration      160/20000 | consumed samples:        81920 | elapsed time per iteration (ms): 2084.1 | learning rate: 1.200E-04 | global batch size:   512 | lm loss:  5.436822 | loss scale: 1.0 | grad norm: 1.458 | skipped iterations:   0 | nan iterations:   0
 iteration      170/20000 | consumed samples:        87040 | elapsed time per iteration (ms): 2071.7 | learning rate: 1.275E-04 | global batch size:   512 | lm loss:  5.361918 | loss scale: 1.0 | grad norm: 1.427 | skipped iterations:   0 | nan iterations:   0
 iteration      180/20000 | consumed samples:        92160 | elapsed time per iteration (ms): 2069.4 | learning rate: 1.350E-04 | global batch size:   512 | lm loss:  5.285631 | loss scale: 1.0 | grad norm: 1.402 | skipped iterations:   0 | nan iterations:   0
 iteration      190/20000 | consumed samples:        97280 | elapsed time per iteration (ms): 2067.0 | learning rate: 1.425E-04 | global batch size:   512 | lm loss:  5.213744 | loss scale: 1.0 | grad norm: 1.381 | skipped iterations:   0 | nan iterations:   0
 iteration      200/20000 | consumed samples:       102400 | elapsed time per iteration (ms): 2064.3 | learning rate: 1.500E-04 | global batch size:   512 | lm loss:  5.149203 | loss scale: 1.0 | grad norm: 1.349 | skipped iterations:   0 | nan iterations:   0
```

Typical characteristics of this kind of log:

- distributed and NCCL initialization appears first
- training arguments are printed next
- dataset indexing and sample/shuffle index loading follow
- training mainly consists of fixed-format iteration lines
- validation blocks appear periodically
- checkpoint save events appear periodically

## 2. Field-by-Field Explanation of Training Log Entries

Use the following iteration line as the main example:

```text
iteration      300/20000 | consumed samples: 153600 | elapsed time per iteration (ms): 2052.6 | learning rate: 1.498E-04 | global batch size: 512 | lm loss: 4.681209 | loss scale: 1.0 | grad norm: 1.181 | skipped iterations: 0 | nan iterations: 0
```

### `iteration 300/20000`

- The current training step is `300`.
- The planned total number of training steps is `20000`.
- In Megatron, `iteration` is essentially the optimizer update step.

### `consumed samples: 153600`

- A total of `153600` samples have been processed so far.
- A common approximation is:

```text
consumed samples ≈ iteration * global_batch_size
```

For example:

```text
300 * 512 = 153600
```

This is useful because many training plans are tracked by processed samples or tokens rather than by step count alone.

### `elapsed time per iteration (ms): 2052.6`

- The average time per training step is about `2052.6 ms`.
- That is about `0.487` steps per second.
- If this number stays stable, training is usually progressing normally.
- If it suddenly increases, common causes include:
  - slow dataloader performance
  - checkpoint saving overhead
  - communication slowdowns
  - one rank becoming slower than the others

### `learning rate: 1.498E-04`

- The current learning rate.
- In scientific notation:

```text
1.498E-04 = 0.0001498
```

- Early in training, this usually rises gradually during warmup.
- Later it may stay flat for a while and then decay, for example with cosine decay.

### `global batch size: 512`

- This is the global batch size, not the per-GPU batch size.
- A common interpretation is:

```text
global_batch_size = micro_batch_size * data_parallel_size * gradient_accumulation_steps
```

- With pipeline parallelism, the visible configuration may look more complicated, but the core meaning is still how many samples are aggregated before one optimizer update.

### `lm loss: 4.681209`

- The language modeling loss, typically cross-entropy loss.
- Lower is usually better.
- It is common for it to start high and then decrease over time.
- If it suddenly jumps from something like `4.x` to `10+`, or becomes `nan`, training stability should be investigated.

### `loss scale: 1.0`

- The loss scaling value used in mixed-precision training.
- In `bf16` training, this is often `1.0` because `bf16` has a larger numeric range and usually does not require dynamic loss scaling.
- In `fp16`, common values are things like `65536` or `32768`.
- If this value keeps getting halved, it usually indicates overflow.

### `grad norm: 1.181`

- The gradient norm.
- It reflects whether the overall gradient magnitude is stable.
- Some fluctuation is normal, but it should remain within a reasonable range.
- If it suddenly jumps from around `1~3` to tens or hundreds, that often suggests gradient explosion, bad data, too large a learning rate, or numeric instability.

### `skipped iterations: 0`

- The number of skipped steps.
- This often happens after `fp16` overflow, where the step does not result in a real parameter update.
- Occasional skips may not be critical, but continuous growth usually indicates unstable training.

### `nan iterations: 0`

- The number of iterations that produced `NaN`.
- If this starts growing, there is usually a real problem.
- Common causes include:
  - learning rate too high
  - bad data
  - numeric overflow
  - unstable custom operators

### Other common fields

#### `number of parameters on (tensor, pipeline) model parallel rank ...`

For example:

```text
> number of parameters on (tensor, pipeline) model parallel rank (0, 0): 6.74B
```

Meaning:

- This is how many parameters are stored on a particular tensor-parallel rank and pipeline-parallel rank.
- It is usually not the total model parameter count, but the shard held by that rank.

#### `validation loss`

For example:

```text
validation loss at iteration 150 | lm loss value: 5.487203E+00 | lm loss PPL: 2.417918E+02
```

Meaning:

- `validation loss`: loss on the validation set
- `PPL`: perplexity, usually corresponding approximately to `exp(loss)`
- If training loss decreases and validation loss also improves, training is usually healthy.
- If training loss decreases but validation loss worsens, overfitting should be considered.

#### `saving checkpoint`

For example:

```text
[Rank 0] saving checkpoint at iteration 200 to /checkpoints/gpt-run01
```

Meaning:

- A checkpoint is being saved.
- Usually the main process or `rank 0` prints this.
- Step time may temporarily increase during checkpointing.

## 3. Example Error Logs When Training Goes Wrong

### CUDA OOM

```text
 iteration      418/20000 | consumed samples:       214016 | elapsed time per iteration (ms): 2488.9 | learning rate: 1.492E-04 | global batch size:   512 | lm loss:  4.312944 | loss scale: 1.0 | grad norm: 1.103 | skipped iterations:   0 | nan iterations:   0
Traceback (most recent call last):
  File "pretrain_gpt.py", line 312, in <module>
    pretrain(...)
  File "/workspace/megatron/training.py", line 1456, in pretrain
    train(...)
  File "/workspace/megatron/training.py", line 987, in train
    losses_dict, skipped_iter, grad_norm, num_zeros_in_grad = train_step(...)
  File "/workspace/megatron/training.py", line 612, in train_step
    forward_backward_func(...)
  File "/workspace/megatron/core/pipeline_parallel/schedules.py", line 421, in forward_backward_pipelining_without_interleaving
    output_tensor = forward_step(...)
  File "/workspace/megatron/model/gpt_model.py", line 233, in forward
    hidden_states = self.decoder(hidden_states, attention_mask, ...)
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.26 GiB.
GPU 3 has a total capacity of 79.15 GiB of which 842.32 MiB is free.
Including non-PyTorch memory, this process has 77.91 GiB memory in use.
If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.
```

Common fixes:

- reduce `micro_batch_size`
- reduce `seq_length`
- enable or increase activation checkpointing
- check whether one rank has abnormal memory growth

### NCCL timeout / communication stall

```text
[Rank 5] Watchdog caught collective operation timeout: WorkNCCL(SeqNum=18421, OpType=ALLREDUCE, NumelIn=524288, NumelOut=524288, Timeout(ms)=600000) ran for 600183 milliseconds before timing out.
[Rank 5] Exception detected by watchdog at work: 18421, last enqueued NCCL work: 18421, last completed NCCL work: 18420.
[E ProcessGroupNCCL.cpp:587] [Rank 5] Some NCCL operations have failed or timed out.
[E ProcessGroupNCCL.cpp:593] [Rank 5] To avoid data inconsistency, we are taking the entire process down.
RuntimeError: [Rank 5] NCCL communicator was aborted on rank 5.
torch.distributed.elastic.multiprocessing.errors.ChildFailedError:
============================================================
pretrain_gpt.py FAILED
------------------------------------------------------------
Failures:
  <NO_OTHER_FAILURES>
------------------------------------------------------------
Root Cause (first observed failure):
[0]:
  time      : 2026-05-14_10:33:48
  host      : node-a
  rank      : 5
  exitcode  : 1
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
============================================================
```

Common causes:

- one GPU hit OOM or failed earlier and the others are waiting
- network instability between nodes
- a dataloader process on one rank is stuck
- inconsistent parallel configuration

### fp16 overflow / loss becomes NaN

```text
 iteration      822/20000 | consumed samples:       420864 | elapsed time per iteration (ms): 2097.4 | learning rate: 1.470E-04 | global batch size:   512 | lm loss:  3.992881 | loss scale: 65536.0 | grad norm: 2.841 | skipped iterations:   0 | nan iterations:   0
 iteration      823/20000 | consumed samples:       421376 | elapsed time per iteration (ms): 2101.6 | learning rate: 1.470E-04 | global batch size:   512 | lm loss:  7.281394 | loss scale: 65536.0 | grad norm: 48.337 | skipped iterations:   0 | nan iterations:   0
 WARNING: Overflow detected, setting loss scale to 32768.0
 iteration      824/20000 | consumed samples:       421888 | elapsed time per iteration (ms): 2113.8 | learning rate: 1.470E-04 | global batch size:   512 | lm loss:       nan | loss scale: 32768.0 | grad norm: nan | skipped iterations:   1 | nan iterations:   1
 WARNING: Overflow detected, setting loss scale to 16384.0
 iteration      825/20000 | consumed samples:       422400 | elapsed time per iteration (ms): 2109.9 | learning rate: 1.470E-04 | global batch size:   512 | lm loss:       nan | loss scale: 16384.0 | grad norm: nan | skipped iterations:   2 | nan iterations:   2
```

Common fixes:

- lower the learning rate
- enable `clip_grad`
- inspect bad samples
- switch to `bf16`, which is usually more stable
- inspect initialization, parallel partitioning, or custom kernels

### checkpoint save failure

```text
[Rank 0] saving checkpoint at iteration     1000 to /checkpoints/gpt-run01
Traceback (most recent call last):
  File "/workspace/megatron/checkpointing.py", line 412, in save_checkpoint
    torch.save(state_dict, checkpoint_name)
  File "/opt/conda/lib/python3.10/site-packages/torch/serialization.py", line 850, in save
    with _open_zipfile_writer(f) as opened_zipfile:
OSError: [Errno 28] No space left on device
```

This is usually straightforward: the disk is full.

## 4. How to Quickly Tell Whether a Log Looks Healthy

Use the following as a quick reference for judging whether training appears normal.

### Check whether `loss` is decreasing overall

- Fast drop early and slower drop later is usually normal.
- Flat or rising loss over a long period usually means learning rate, data, or batch configuration should be checked.

### Check whether `grad norm` is stable

- Small fluctuations are normal.
- If it frequently spikes to very large values, such as tens or hundreds, training is usually unstable.

### Check `skipped iterations` and `nan iterations`

- One or two occasional skipped steps may not be severe.
- Continuous growth usually means numeric stability is getting worse.
- If `nan iterations` keeps growing, that is usually a clear warning sign.

### Check whether `loss scale` keeps dropping

- In `fp16`, occasional adjustment is normal.
- If it keeps getting halved, overflow is probably frequent.

### Check whether `elapsed time per iteration` suddenly becomes slow

- If it was stable around `~2000ms` and later often becomes `4000ms+`, check:
  - dataloader
  - checkpoint I/O
  - communication
  - whether one rank is lagging

### Check whether `validation loss` improves as well

- If both training loss and validation loss improve, that is usually a healthy trend.
- If training loss improves but validation loss worsens, overfitting or a data distribution issue may be involved.

### Example of a healthy trend

```text
iter 100   | loss 6.48 | grad norm 1.98 | skipped 0
iter 500   | loss 4.77 | grad norm 1.12 | skipped 1
iter 1000  | loss 4.21 | grad norm 0.96 | skipped 1
iter 2000  | loss 3.89 | grad norm 0.88 | skipped 1
iter 5000  | loss 3.42 | grad norm 0.79 | skipped 2
```

This usually means:

- loss is decreasing
- gradient norm is not exploding
- skipped steps are rare
- training is progressing steadily

### Example of an unhealthy trend

```text
iter 100   | loss 6.52 | grad norm 2.1   | skipped 0
iter 120   | loss 8.93 | grad norm 48.7  | skipped 3
iter 121   | loss nan  | grad norm nan   | skipped 4 | nan iterations: 1
iter 122   | loss nan  | grad norm nan   | skipped 5 | nan iterations: 2
```

This usually suggests:

- learning rate may be too high
- mixed precision is unstable
- there may be bad samples in the data
- gradient clipping, initialization, or parallel configuration may be problematic

## 5. Megatron Log Diagnosis Template

You can treat the following as a practical troubleshooting checklist.

### A. First check whether training is actually making progress

Focus on:

- `iteration`
- `consumed samples`
- `elapsed time per iteration (ms)`

Normal behavior:

- `iteration` keeps increasing
- `consumed samples` grows as expected
- step time is roughly stable

Abnormal behavior:

- steps stop moving
- no new logs appear after a certain point
- step time jumps from `~2000ms` to `10000ms+`

Check first:

- dataloader stall
- one rank disconnected or failed
- NCCL communication blockage
- checkpoint saving is too slow
- storage or network jitter

### B. Then check whether loss looks healthy

Focus on:

- `lm loss`
- `validation loss`

Normal behavior:

- fast drop early in training
- slower decline later
- validation loss roughly improves as well

Abnormal behavior:

- loss does not decrease for a long time
- loss suddenly jumps from `4.x` to `10+`
- loss becomes `nan`

Check first:

- learning rate too high
- bad or dirty samples in the data
- tokenization or label alignment errors
- mixed-precision overflow
- incorrect model parallel setup

### C. Check whether gradients are stable

Focus on:

- `grad norm`
- `clip_grad` configuration

Normal behavior:

- gradient norm fluctuates, but within a stable scale
- for example, long-term values around `0.5 ~ 5`

Abnormal behavior:

- sudden jump to `20 / 50 / 100+`
- directly becomes `nan`

Check first:

- learning rate too high
- gradient clipping disabled or too loose
- bad data
- unstable initialization
- numeric issues in custom operators

### D. Check mixed-precision stability

Focus on:

- `loss scale`
- `skipped iterations`
- `nan iterations`

Normal behavior:

- in `bf16`, `loss scale: 1.0` is common
- in `fp16`, occasional loss scale adjustment is okay, but it should not collapse continuously
- occasional skipped steps may be acceptable

Abnormal behavior:

- `loss scale` keeps getting halved
- `skipped iterations` keeps increasing
- `nan iterations` starts accumulating

Check first:

- switch to `bf16` if possible
- reduce learning rate
- reduce `micro_batch_size`
- inspect bad data
- inspect fused kernels or custom operators

### E. Check memory and batch configuration

If the logs show:

- `CUDA out of memory`
- one GPU has very little free memory
- one rank OOMs earlier than the others

Check first:

- `micro_batch_size`
- `seq_length`
- activation checkpointing
- whether tensor/pipeline partitioning is balanced
- whether one rank is holding extra state

### F. Check distributed communication

Key errors:

- `NCCL timeout`
- `communicator was aborted`
- `address already in use`
- one rank failed

Reasoning pattern:

- if OOM appears first and NCCL timeout appears later, OOM is often the root cause
- if one rank is stuck, the others often report NCCL timeout as well
- if startup fails, first check `MASTER_ADDR/PORT`, world size, and rank mapping

### G. A very practical troubleshooting order

When something goes wrong, the following order is usually efficient:

1. Find the first real error, not the last one.
2. Check whether `OOM` happened first.
3. Check whether `NaN` or `overflow` happened first.
4. Check whether one rank failed first.
5. Then look at follow-up `NCCL timeout` errors.

In many cases, the final visible error is a communication failure, while the real root cause happened earlier on one GPU.
