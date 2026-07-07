# CS336 Assignment 1 任务整理

来源：[cs336_assignment1_basics.pdf](./cs336_assignment1_basics.pdf)，Version 26.0.3，Spring 2026。

本文只整理需要完成的任务、测试入口和提交物，不包含作业解法。

## 总体目标

本作业要求从零实现并训练一个标准 Transformer 语言模型。主要工作包括：

- 实现 byte-level BPE tokenizer。
- 实现 Transformer LM 的核心模块。
- 实现 cross-entropy loss、AdamW optimizer、learning rate schedule 和 gradient clipping。
- 实现训练循环、数据加载、checkpoint 保存/恢复。
- 在 TinyStories 和 OpenWebText 上训练、评估、生成文本，并完成实验报告。

## 仓库工作方式

- 主要代码写在 `cs336_basics/` 下。
- `tests/adapters.py` 是测试适配层，需要把测试调用转发到你自己的实现。
- `tests/test_*.py` 是官方测试，不应修改。
- `adapters.py` 只应包含胶水代码，不应放主要逻辑。

## 提交物

Gradescope 需要提交：

- `writeup.pdf`：所有书面题、实验结果、学习曲线和分析。
- `code.zip`：你写的全部代码。运行 `make_submission.sh` 生成。

Leaderboard 需要向以下仓库提交 PR：

- `github.com/stanford-cs336/assignment1-basics-leaderboard`

## 数据准备

需要使用两个预处理后的纯文本数据集：

- TinyStories
- OpenWebText sample

README 中给出了下载命令。后续 tokenizer 和 LM 训练都依赖这些数据。

## 第 2 部分：BPE Tokenizer

### unicode1 - Understanding Unicode - 1 分

书面题：

- 说明 `chr(0)` 返回什么 Unicode 字符。
- 比较该字符的 `__repr__()` 表示和 `print()` 表示。
- 说明该字符出现在文本中时会发生什么。

交付物：

- 三个一两句以内的回答，写入 `writeup.pdf`。

### unicode2 - Unicode Encodings - 3 分

书面题：

- 说明为什么 tokenizer 更适合基于 UTF-8 bytes 训练，而不是 UTF-16 或 UTF-32。
- 给出一个让错误 UTF-8 解码函数产生错误结果的 byte string，并说明原因。
- 给出一个无法解码成 Unicode 字符的两字节序列，并说明原因。

交付物：

- 三个简短回答，写入 `writeup.pdf`。

### train_bpe - BPE Tokenizer Training - 15 分

代码任务：

- 实现一个 byte-level BPE 训练函数。
- 输入参数至少包括：
  - `input_path: str`
  - `vocab_size: int`
  - `special_tokens: list[str]`
- 返回：
  - `vocab: dict[int, bytes]`
  - `merges: list[tuple[bytes, bytes]]`
- special tokens 要作为硬边界，不能跨越它们做 merge，也不要把它们计入 merge statistics。

测试入口：

```sh
uv run pytest tests/test_train_bpe.py
```

需要实现的 adapter：

- `adapters.run_train_bpe`

### train_bpe_tinystories - TinyStories 上训练 BPE - 2 分

实验任务：

- 在 TinyStories 上训练 byte-level BPE tokenizer。
- 最大词表大小：`10000`。
- special token：`<|endoftext|>`。
- 序列化保存 vocab 和 merges。
- 记录训练时间、内存占用、最长 token，并判断是否合理。
- profile 代码，找出 tokenizer 训练最耗时的部分。

交付物：

- 两段一两句的实验回答，写入 `writeup.pdf`。

资源目标：

- 不使用 GPU。
- 训练时间不超过 30 分钟。
- 内存不超过 30 GB。

### train_bpe_expts_owt - OpenWebText 上训练 BPE - 2 分

实验任务：

- 在 OpenWebText 上训练 byte-level BPE tokenizer。
- 最大词表大小：`32000`。
- 序列化保存 vocab 和 merges。
- 记录最长 token，并判断是否合理。
- 对比 TinyStories tokenizer 和 OpenWebText tokenizer。

交付物：

- 两段一两句的实验回答，写入 `writeup.pdf`。

资源目标：

- 不使用 GPU。
- 训练时间不超过 12 小时。
- 内存不超过 100 GB。

### tokenizer - Implementing the tokenizer - 15 分

代码任务：

- 实现 `Tokenizer` 类，能够根据 vocab 和 merges 编码/解码文本。
- 支持用户提供 special tokens；如果 special tokens 不在 vocab 中，需要追加。
- 支持大文件流式编码，避免一次性把整份文本读入内存。
- 解码非法 byte 序列时应使用 Unicode replacement character。

推荐接口：

```python
__init__(self, vocab, merges, special_tokens=None)
from_files(cls, vocab_filepath, merges_filepath, special_tokens=None)
encode(self, text: str) -> list[int]
encode_iterable(self, iterable) -> Iterator[int]
decode(self, ids: list[int]) -> str
```

测试入口：

```sh
uv run pytest tests/test_tokenizer.py
```

需要实现的 adapter：

- `adapters.get_tokenizer`

### tokenizer_experiments - Tokenizer 实验 - 4 分

实验任务：

- 从 TinyStories 和 OpenWebText 各采样 10 个文档。
- 分别用 TinyStories 10K tokenizer 和 OpenWebText 32K tokenizer 编码样本。
- 计算 bytes/token 压缩率。
- 用 TinyStories tokenizer 编码 OpenWebText 样本，比较压缩率和现象。
- 估算 tokenizer throughput，例如 bytes/second。
- 估算 tokenize 825 GB Pile 数据集需要多久。
- 用对应 tokenizer 把 TinyStories 和 OpenWebText 的 train/dev 数据编码成整数 token ID 序列。
- 推荐保存为 `uint16` NumPy array，并解释为什么 `uint16` 合适。

交付物：

- 四段简短实验回答，写入 `writeup.pdf`。

## 第 3 部分：Transformer LM Architecture

### linear - Linear module - 1 分

代码任务：

- 实现不带 bias 的 `Linear` module，继承 `torch.nn.Module`。
- 参数按 `W` 存储，不存 `W.T`。
- 不使用 `nn.Linear` 或 `nn.functional.linear`。
- 使用 `torch.nn.init.trunc_normal_` 初始化。

推荐接口：

```python
__init__(self, in_features, out_features, device=None, dtype=None)
forward(self, x: torch.Tensor) -> torch.Tensor
```

测试入口：

```sh
uv run pytest -k test_linear
```

需要实现的 adapter：

- `adapters.run_linear`

### embedding - Embedding module - 1 分

代码任务：

- 实现 `Embedding` module，继承 `torch.nn.Module`。
- 不使用 `nn.Embedding` 或 `nn.functional.embedding`。
- embedding matrix 形状为 `(vocab_size, d_model)`。
- 使用 `torch.nn.init.trunc_normal_` 初始化。

推荐接口：

```python
__init__(self, num_embeddings, embedding_dim, device=None, dtype=None)
forward(self, token_ids: torch.Tensor) -> torch.Tensor
```

测试入口：

```sh
uv run pytest -k test_embedding
```

需要实现的 adapter：

- `adapters.run_embedding`

### rmsnorm - RMSNorm - 1 分

代码任务：

- 实现 `RMSNorm`。
- forward 中先 upcast 到 `torch.float32`，计算后再转回原 dtype。

推荐接口：

```python
__init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None)
forward(self, x: torch.Tensor) -> torch.Tensor
```

测试入口：

```sh
uv run pytest -k test_rmsnorm
```

需要实现的 adapter：

- `adapters.run_rmsnorm`

### positionwise_feedforward - SwiGLU FFN - 2 分

代码任务：

- 实现 SwiGLU feed-forward network。
- 由 SiLU activation 和 GLU 组成。
- 可使用 `torch.sigmoid` 保持数值稳定。
- `d_ff` 约为 `8 / 3 * d_model`，并取为 64 的倍数。

测试入口：

```sh
uv run pytest -k test_swiglu
```

需要实现的 adapter：

- `adapters.run_swiglu`

### rope - Rotary Positional Embedding - 2 分

代码任务：

- 实现 `RotaryPositionalEmbedding`。
- 对输入 tensor 应用 RoPE。
- 没有可学习参数。
- 可预计算 sin/cos buffer，并用 `register_buffer(persistent=False)` 保存。
- forward 需要支持任意 batch-like dimensions。

推荐接口：

```python
__init__(self, theta: float, d_k: int, max_seq_len: int, device=None)
forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor
```

测试入口：

```sh
uv run pytest -k test_rope
```

需要实现的 adapter：

- `adapters.run_rope`

### softmax - Stable softmax - 1 分

代码任务：

- 实现指定维度上的 softmax。
- 需要先减去该维度最大值以保证数值稳定。

测试入口：

```sh
uv run pytest -k test_softmax_matches_pytorch
```

需要实现的 adapter：

- `adapters.run_softmax`

### scaled_dot_product_attention - 5 分

代码任务：

- 实现 scaled dot-product attention。
- 支持 query/key shape 为 `(batch_size, ..., seq_len, d_k)`。
- 支持 value shape 为 `(batch_size, ..., seq_len, d_v)`。
- 返回 shape 为 `(batch_size, ..., seq_len, d_v)`。
- 支持可选 boolean mask，mask 中 `True` 表示可 attend，`False` 表示不可 attend。

测试入口：

```sh
uv run pytest -k test_scaled_dot_product_attention
uv run pytest -k test_4d_scaled_dot_product_attention
```

需要实现的 adapter：

- `adapters.run_scaled_dot_product_attention`

### multihead_self_attention - Causal MHA - 5 分

代码任务：

- 实现 causal multi-head self-attention。
- 防止 token attend 到未来位置。
- RoPE 只应用到 query 和 key，不应用到 value。
- head dimension 应作为 batch-like dimension 处理。
- `d_k = d_v = d_model / num_heads`。

推荐参数：

```python
d_model: int
num_heads: int
```

测试入口：

```sh
uv run pytest -k test_multihead_self_attention
```

需要实现的 adapter：

- `adapters.run_multihead_self_attention`

### transformer_block - Transformer block - 3 分

代码任务：

- 实现 pre-norm Transformer block。
- 每个 block 包含：
  - RMSNorm + causal MHA + residual
  - RMSNorm + SwiGLU FFN + residual

推荐参数：

```python
d_model: int
num_heads: int
d_ff: int
```

测试入口：

```sh
uv run pytest -k test_transformer_block
```

需要实现的 adapter：

- `adapters.run_transformer_block`

### transformer_lm - Full Transformer LM - 3 分

代码任务：

- 组装完整 Transformer language model。
- 包含 token embedding、多个 Transformer block、最终 RMSNorm、LM head。
- 输出 vocab logits。

额外推荐参数：

```python
vocab_size: int
context_length: int
num_layers: int
```

测试入口：

```sh
uv run pytest -k test_transformer_lm
```

需要实现的 adapter：

- `adapters.run_transformer_lm`

### transformer_accounting - 资源计算 - 5 分

书面题：

- 对 GPT-2 XL-shaped assignment 架构计算参数量和加载模型所需内存。
- 列出一次 forward pass 中的所有 matrix multiplies，并计算总 FLOPs。
- 判断哪些组件占用最多 FLOPs。
- 对 GPT-2 small、medium、large 重复 FLOPs 组成分析，并比较模型变大后各组件占比变化。
- 将 GPT-2 XL 的 context length 增加到 `16384`，分析总 FLOPs 和各组件占比变化。

交付物：

- 参数量、内存、FLOPs 明细和简短分析，写入 `writeup.pdf`。

## 第 4 部分：Training a Transformer LM

### cross_entropy - 1 分

代码任务：

- 实现 cross-entropy loss。
- 输入 predicted logits 和 targets。
- 需要数值稳定处理：
  - 减去最大 logit。
  - 尽量消去不必要的 `log(exp(...))`。
- 支持 batch-like dimensions。
- 返回 batch 平均 loss。

测试入口：

```sh
uv run pytest -k test_cross_entropy
```

需要实现的 adapter：

- `adapters.run_cross_entropy`

### learning_rate_tuning - SGD 学习率观察 - 1 分

实验任务：

- 运行讲义中的 SGD toy example。
- 分别测试 learning rate：`1e1`、`1e2`、`1e3`。
- 每个只跑 10 次迭代。
- 观察 loss 是更快下降、更慢下降，还是发散。

交付物：

- 一两句实验观察，写入 `writeup.pdf`。

### adamw - AdamW optimizer - 2 分

代码任务：

- 实现 AdamW，继承 `torch.optim.Optimizer`。
- `__init__` 支持 learning rate、`beta`、`eps`、weight decay。
- 使用 `self.state` 保存每个参数的一阶、二阶矩估计等状态。

测试入口：

```sh
uv run pytest -k test_adamw
```

需要实现的 adapter：

- `adapters.get_adamw_cls`

### adamw_accounting - AdamW 训练资源计算 - 2 分

书面题：

- 用 float32 假设计算 AdamW peak memory，并按 parameters、activations、gradients、optimizer state 分解。
- 将表达式实例化到 GPT-2 XL-shaped model，得到只依赖 batch size 的表达式，并计算 80 GB 下最大 batch size。
- 计算 AdamW 单步 FLOPs。
- 假设 H100 50% MFU，计算 GPT-2 XL 训练 400K steps、batch size 1024 需要多少小时。

交付物：

- 代数表达式、数值表达式、最大 batch size、训练小时数和简短说明，写入 `writeup.pdf`。

### learning_rate_schedule - Cosine schedule with warmup - 1 分

代码任务：

- 实现 cosine learning rate schedule with warmup。
- 输入包括当前 step、最大学习率、最小学习率、warmup steps、cosine decay 结束 step。
- 输出当前 step 使用的 learning rate。

测试入口：

```sh
uv run pytest -k test_get_lr_cosine_schedule
```

需要实现的 adapter：

- `adapters.get_lr_cosine_schedule`

### gradient_clipping - 1 分

代码任务：

- 实现 gradient clipping。
- 输入参数列表和最大 L2 norm。
- 原地修改每个参数的 gradient。
- 使用 `eps = 1e-6`。

测试入口：

```sh
uv run pytest -k test_gradient_clipping
```

需要实现的 adapter：

- `adapters.run_gradient_clipping`

## 第 5 部分：Training loop

### data_loading - 2 分

代码任务：

- 实现 batch 采样函数。
- 输入：
  - token ID NumPy array
  - `batch_size`
  - `context_length`
  - PyTorch device string，例如 `cpu`、`cuda:0`、`mps`
- 输出：
  - inputs tensor，shape `(batch_size, context_length)`
  - targets tensor，shape `(batch_size, context_length)`
- outputs 应放到指定 device。
- 大数据建议使用 `np.memmap` 或 `np.load(..., mmap_mode="r")`。

测试入口：

```sh
uv run pytest -k test_get_batch
```

需要实现的 adapter：

- `adapters.run_get_batch`

### checkpointing - 1 分

代码任务：

- 实现 checkpoint 保存和加载。
- 保存内容至少包括：
  - model state
  - optimizer state
  - iteration number
- 加载时恢复 model 和 optimizer，并返回 iteration number。

推荐接口：

```python
save_checkpoint(model, optimizer, iteration, out)
load_checkpoint(src, model, optimizer)
```

测试入口：

```sh
uv run pytest -k test_checkpointing
```

需要实现的 adapter：

- `adapters.run_save_checkpoint`
- `adapters.run_load_checkpoint`

### training_together - 完整训练脚本 - 4 分

代码任务：

- 写一个训练脚本，把 tokenizer 产物、数据加载、Transformer LM、loss、optimizer、schedule、checkpoint 和 logging 串起来。
- 推荐支持：
  - 配置 model 和 optimizer hyperparameters。
  - 用 `np.memmap` 高效加载大型 train/validation 数据。
  - 保存 checkpoint 到用户指定路径。
  - 定期记录 training loss 和 validation loss。
  - 可记录到 console 或 Weights and Biases。

交付物：

- 可运行训练脚本。

## 第 6 部分：Generating text

### decoding - 3 分

代码任务：

- 实现从 language model 生成文本的 decode 函数。
- 支持：
  - 用户提供 prompt。
  - 生成直到 `<|endoftext|>` 或达到最大 token 数。
  - 控制最大生成 token 数。
  - temperature scaling。
  - top-p / nucleus sampling。

交付物：

- decoding 代码。

## 第 7 部分：Experiments

### experiment_log - 3 分

实验基础设施：

- 为训练和评估建立 experiment tracking。
- 记录 loss curves，横轴至少包括 gradient steps 和 wall-clock time。
- 记录实验日志，说明每次尝试了什么。

交付物：

- logging infrastructure code。
- experiment log 文档，覆盖本节后续实验。

### TinyStories 基础配置

讲义建议的起始配置：

- `vocab_size = 10000`
- `context_length = 256`
- `d_model = 512`
- `d_ff = 1344`
- RoPE `theta = 10000`
- `num_layers = 4`
- `num_heads = 16`
- 总 token 数约 `327,680,000`

需要调的超参数：

- learning rate
- warmup steps
- AdamW `beta1`、`beta2`、`eps`
- weight decay
- batch size

低资源设备可把总 token 数降到约 `40,000,000`。

### learning_rate - Tune learning rate - 3 分

实验任务：

- 对 TinyStories base model 做 learning rate sweep。
- 报告多个 learning rate 的 learning curves。
- 说明搜索策略。
- 至少包含一个 divergent run，用于分析 edge of stability。
- 训练出 TinyStories validation loss 不高于 `1.45` 的模型。

低资源目标：

- 如果使用 CPU 或 Apple Silicon，可把目标改为 validation loss 不高于 `2.00`。

交付物：

- 多条 learning curves。
- 搜索策略说明。
- 最佳模型结果。
- 包含发散学习率的曲线和分析。

### batch_size_experiment - Batch size variations - 1 分

实验任务：

- 从 batch size 1 到 GPU memory limit 做 batch size 实验。
- 至少包含几个中间值，例如 64 和 128。
- 必要时重新调 learning rate。

交付物：

- 不同 batch size 的 learning curves。
- 几句关于 batch size 对训练影响的讨论。

### generate - Generate text - 1 分

实验任务：

- 使用 decoder 和训练好的 checkpoint 生成文本。
- 可调整 temperature、top-p 等参数提高流畅度。

交付物：

- 至少 256 tokens 的文本输出，或生成到第一个 `<|endoftext|>` 为止。
- 简短评论输出流畅度。
- 至少说明两个影响输出质量的因素。

### layer_norm_ablation - Remove RMSNorm - 1 分

实验任务：

- 从 Transformer 中移除所有 RMSNorm 后训练。
- 观察原最佳 learning rate 下会发生什么。
- 尝试更低 learning rate 是否能稳定训练。

交付物：

- 移除 RMSNorm 后的 learning curve。
- 最佳 learning rate 下的 learning curve。
- 几句关于 RMSNorm 影响的评论。

### pre_norm_ablation - Post-norm Transformer - 1 分

实验任务：

- 将 pre-norm Transformer 改成 post-norm Transformer。
- 训练并比较结果。

交付物：

- post-norm 与 pre-norm 的 learning curve 对比。

### no_pos_emb - NoPE - 1 分

实验任务：

- 移除 RoPE，不使用任何 position embedding。
- 与 RoPE base model 比较。

交付物：

- RoPE 与 NoPE 的 learning curve 对比。

### swiglu_ablation - SwiGLU vs SiLU - 1 分

实验任务：

- 将 SwiGLU FFN 与不带 GLU 的 SiLU FFN 对比。
- SiLU baseline 的 inner dimension 设为约 `4 * d_model`，以近似匹配参数量。

交付物：

- SwiGLU 与 SiLU FFN 的 learning curve 对比。
- 几句实验发现。

### main_experiment - OpenWebText - 2 分

实验任务：

- 用与 TinyStories 相同的模型架构和训练迭代数在 OpenWebText 上训练。
- 可能需要重新调 learning rate 或 batch size。
- 比较 OpenWebText 与 TinyStories 的 loss。
- 生成 OpenWebText LM 文本。

交付物：

- OpenWebText learning curve。
- 关于 OWT loss 与 TinyStories loss 差异的解释。
- OpenWebText 模型生成文本。
- 评论生成质量，并解释为什么同样模型和计算预算下质量更差。

### leaderboard - 6 分

实验任务：

- 在 leaderboard 规则内改进模型或训练策略，目标是降低 OpenWebText validation loss。

规则：

- 单次提交最多运行 45 分钟 B200。
- 只能使用课程提供的 OpenWebText training dataset。
- 其他修改自由。
- 目标是在 `0.75` B200-hours 内尽量降低 validation loss。
- 期望至少优于 naive baseline：validation loss `5.0`。

交付物：

- 最终 validation loss。
- 横轴明确为 wall-clock time 且小于 45 分钟的 learning curve。
- 修改内容说明。
- 向 leaderboard 仓库提交 PR。

## 建议完成顺序

1. 阅读 PDF 和测试文件，确认所有 adapter 名称。
2. 完成 Unicode 书面题。
3. 实现 BPE training，并通过 `test_train_bpe.py`。
4. 训练 TinyStories tokenizer，再训练 OpenWebText tokenizer。
5. 实现 `Tokenizer`，通过 `test_tokenizer.py`。
6. 完成 tokenizer experiments，并保存 tokenized datasets。
7. 依次实现 Transformer 基础模块：Linear、Embedding、RMSNorm、SwiGLU、RoPE、softmax、attention、MHA。
8. 组装 Transformer block 和 Transformer LM，并通过模型相关测试。
9. 完成 Transformer resource accounting 书面题。
10. 实现 cross-entropy、AdamW、LR schedule、gradient clipping。
11. 实现 data loading、checkpointing、training script。
12. 实现 decoding。
13. 在 TinyStories 上完成学习率、batch size、生成文本和 ablation 实验。
14. 在 OpenWebText 上完成主实验。
15. 做 leaderboard 实验。
16. 整理 `writeup.pdf`、运行测试、执行 `make_submission.sh`。

## 测试清单

按模块逐步运行：

```sh
uv run pytest tests/test_train_bpe.py
uv run pytest tests/test_tokenizer.py
uv run pytest -k test_linear
uv run pytest -k test_embedding
uv run pytest -k test_rmsnorm
uv run pytest -k test_swiglu
uv run pytest -k test_rope
uv run pytest -k test_softmax_matches_pytorch
uv run pytest -k test_scaled_dot_product_attention
uv run pytest -k test_4d_scaled_dot_product_attention
uv run pytest -k test_multihead_self_attention
uv run pytest -k test_transformer_block
uv run pytest -k test_transformer_lm
uv run pytest -k test_cross_entropy
uv run pytest -k test_adamw
uv run pytest -k test_get_lr_cosine_schedule
uv run pytest -k test_gradient_clipping
uv run pytest -k test_get_batch
uv run pytest -k test_checkpointing
```

最终完整测试：

```sh
uv run pytest
```

## 最终检查清单

- 所有 required tests 通过。
- `tests/adapters.py` 只包含适配逻辑。
- `writeup.pdf` 覆盖所有书面题和实验题。
- TinyStories tokenizer、OpenWebText tokenizer、tokenized datasets 已保存。
- 训练脚本支持可配置超参数、checkpoint、validation 和 logging。
- 关键实验有 learning curve，并记录 wall-clock time。
- 生成文本满足 token 数要求。
- 大文件、checkpoint、数据集没有误打包进 `code.zip`。
- 已运行 `make_submission.sh` 生成提交包。
