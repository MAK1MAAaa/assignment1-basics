# CS336 2025 春季作业 1：基础

作业的完整说明请阅读作业讲义：
[cs336_assignment1_basics.pdf](./cs336_assignment1_basics.pdf)

如果你发现作业讲义或代码存在问题，可以在 GitHub 上提交 issue，或打开 pull request 提交修复。

## 环境配置

### 环境

本项目使用 `uv` 管理环境，以保证可复现性、可移植性和易用性。
你可以在[这里](https://github.com/astral-sh/uv#installation)安装 `uv`，这是推荐方式；
也可以运行 `pip install uv` 或 `brew install uv`。

建议阅读 `uv` 的项目管理文档：
[Managing dependencies](https://docs.astral.sh/uv/guides/projects/#managing-dependencies)。

之后可以用下面的命令运行仓库中的任意 Python 文件：

```sh
uv run <python_file_path>
```

必要时，`uv` 会自动解析并激活对应环境。

### 运行单元测试

```sh
uv run pytest
```

初始状态下，所有测试都会因为 `NotImplementedError` 失败。
要把你的实现连接到测试，需要完成 [./tests/adapters.py](./tests/adapters.py) 中的函数。

### BPE 训练脚手架

Part 2 的 BPE 训练脚手架位于
[./cs336_basics/Part2/train_bpe.py](./cs336_basics/Part2/train_bpe.py)。
该文件暴露 `train_bpe(input_path, vocab_size, special_tokens)`，并将实现拆分为输入校验、特殊 token 边界切分、预分词计数、字节 token 初始化、相邻 pair 统计、merge 选择、merge 应用和词表构建等步骤。

`tests.adapters.run_train_bpe` 已接入该函数。补全或调整 `train_bpe.py` 后，可以运行：

```sh
uv run pytest tests/test_train_bpe.py
```

如果只想先验证 Part 2 目录下的本地辅助测试，可以运行：

```sh
uv run pytest cs336_basics/Part2/test_train_bpe.py
```

#### BPE 训练性能优化记录

`train_bpe.py` 中已在相关函数旁用注释保留原始朴素写法，并实现了下面两处优化：

- `count_adjacent_pairs`：原始写法使用 `Counter` 和 `zip(token, token[1:])`，每次新增 key 都会触发 `Counter.__missing__`，并且 `token[1:]` 会产生切片。当前改为普通 `dict` 加 `get` 累加，并使用下标遍历，减少额外对象分配。
- `apply_merge`：原始写法会无条件创建 `merged_parts` 列表，即使当前 token 不包含要 merge 的 pair。当前改为懒构造，只有遇到目标 pair 时才创建新列表；没有变化的 token 直接复用原 tuple。

当前验证结果：

```sh
uv run pytest cs336_basics/Part2/test_train_bpe.py
uv run pytest tests/test_train_bpe.py
```

结果分别为 `1 passed` 和 `3 passed`，官方 BPE 速度测试已通过。

#### TinyStories 10k BPE 训练任务

新增脚本 [train_bpe_tinystories.py](./cs336_basics/Part2/train_bpe_tinystories.py)，用于在完整 TinyStories
训练集上训练最大词表大小为 10,000 的 byte-level BPE tokenizer，并显式加入
`<|endoftext|>` special token。该脚本独立于
[train_bpe.py](./cs336_basics/Part2/train_bpe.py)：它复制了原始 BPE 的基础校验、special token
切分、GPT-2 预分词和词表初始化逻辑，并在脚本内用注释标出针对 TinyStories 任务的修改点。
针对 2.1GB 训练文件，脚本将原始整文件读取改为按 `<|endoftext|>` 文档边界流式切分，将单进程预分词改为多进程并行计数，并将每轮全量扫描 pair 的 BPE 训练替换为增量 pair 倒排索引。

运行命令：

```sh
uv run python cs336_basics/Part2/train_bpe_tinystories.py \
  --input data/TinyStoriesV2-GPT4-train.txt \
  --output-dir artifacts/tinystories_bpe \
  --vocab-size 10000
```

输出文件：

- `artifacts/tinystories_bpe/vocab.json`：词表，按 token id 存储 token 的十六进制表示、UTF-8 检视文本和字节长度。
- `artifacts/tinystories_bpe/merges.json`：merge 列表，按生成顺序存储左右 token。
- `artifacts/tinystories_bpe/summary.json`：训练参数、资源占用和最长 token 统计。

本机实测结果：

```text
vocab_size=10000
merges=9743
total_seconds=99.963
pretokenization_seconds=85.299
bpe_training_seconds=14.621
serialization_seconds=0.035
peak_rss_mb=3737.2
unique_pretokens=59933
total_pretokens=536592168
longest_token=id=9379 bytes=15 utf8=' responsibility'
```

训练耗时约 100 秒，进程树峰值 RSS 约 3.65 GiB，满足“不超过 30 分钟、30GB RAM、无 GPU”的资源要求。

Profiling 结果显示，tokenizer 训练中最耗时的是预分词计数阶段：`pretokenization_seconds=85.299`，
约占总耗时的 85%。主要开销来自对 2.1GB TinyStories 文本按 `<|endoftext|>` 边界流式切分、
UTF-8 解码、GPT-2 正则预分词，以及跨进程汇总 `Counter[bytes]`。实际 BPE merge 训练阶段耗时
`14.621` 秒，序列化耗时只有 `0.035` 秒，因此当前瓶颈不是 merge 选择或产物写入，而是大规模文本预分词和计数。

最长 token 是 `" responsibility"`，长度为 15 字节。这个结果合理：GPT-2 风格预分词会把英文单词前的空格保留在同一个 pretoken 中，BPE 会优先把高频片段逐步合并成完整词或带前导空格的完整词。TinyStories 中这类常见词被学习成单个 token 符合预期。

### 下载数据

下载 TinyStories 数据和 OpenWebText 的一个子样本：

```sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

如果在 macOS 上执行上述命令时出现下面的错误：

```text
zsh: command not found: wget
```

原因是 macOS 默认通常不自带 `wget`。

可以用下面两种方式解决：

1. 直接改用系统自带的 `curl`：

```sh
mkdir -p data
cd data

curl -L -O https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
curl -L -O https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

curl -L -O https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
curl -L -O https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

2. 如果希望继续使用原文中的命令，可以先安装 `wget`：

```sh
brew install wget
```

下载完成后可以检查文件是否齐全：

```sh
ls -lh data
```

## `uv sync` 下载 Python 超时的原因

你遇到的错误：

```text
error: Request failed after 3 retries in 48.7s
  Caused by: Failed to download https://mirrors.ustc.edu.cn/github-release/astral-sh/python-build-standalone/20260414/cpython-3.13.13%2B20260414-aarch64-apple-darwin-install_only_stripped.tar.gz
  Caused by: error sending request for url (...)
  Caused by: client error (Connect)
  Caused by: operation timed out
```

根本原因不是项目代码问题，而是 `uv` 在同步环境时需要下载一个受它管理的 Python 运行时，但连接镜像站超时。

本项目的 [pyproject.toml](./pyproject.toml) 中有两个相关配置：

```toml
requires-python = ">=3.12,<3.14"

[tool.uv]
python-preference = "managed"
```

含义如下：

- 项目要求 Python 版本必须大于等于 3.12 且小于 3.14。
- `python-preference = "managed"` 表示 `uv` 会优先使用自己管理的 Python。
- 如果本机没有可用的匹配版本，或者 `uv` 决定使用托管 Python，它会从 `python-build-standalone` 下载对应的 CPython 包。
- 你的机器是 Apple Silicon macOS，因此它尝试下载 `aarch64-apple-darwin` 架构的 CPython 3.13.13。
- 下载地址被配置到了中科大镜像 `mirrors.ustc.edu.cn`，但该连接在重试 3 次后仍然超时，所以 `uv sync` 失败。

可以按下面顺序排查：

1. 确认网络能访问该镜像地址，或稍后重试。
2. 如果你本机已经安装了符合要求的 Python 3.12 或 3.13，可以改用系统 Python：

```sh
uv sync --python-preference only-system
```

3. 也可以显式指定一个已安装的 Python：

```sh
uv sync --python 3.12 --python-preference only-system
```

4. 如果问题来自当前镜像，可以清理或调整 `uv`/环境里的镜像配置，让它改用官方源或其他可用镜像。

这个报错的关键判断点是最后的 `operation timed out`：它说明失败发生在网络连接阶段，而不是依赖版本冲突、测试失败或 Python 代码错误。
