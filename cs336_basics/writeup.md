### Problem(unicode1)

#### (a)

空字符NUL

#### (b)

可以输出内容`'\x00'`

#### (c)

`chr(0)`不会显示出来，当他在句子中时候，下一个字符会直接输出在他的位置（ghostty+zsh）

### Problem(unicode2)

#### (a)

| 编码类型 | 占字节数 |
| -------- | -------- |
| UTF-8    | 1-4      |
| UTF-16   | 2-4      |
| UTF-32   | 4        |

超过98%的网页都在使用utf8编码，而且英文在utf8下仅占1个字节，可以显著的节约空间。

#### (b)

这个函数仅能在纯英文环境下才能正常运行，例如使用中文或者emoji都会导致报错。

```text
mac $makima: ~/Desktop/code/python projects/stf_course ❯ uv run "/Users/makima/Desktop/code/python projects/stf_course/assignment1-basics/cs336_basics/Part 2/temp_2_2.py"
Traceback (most recent call last):
  File "/Users/makima/Desktop/code/python projects/stf_course/assignment1-basics/cs336_basics/Part 2/temp_2_2.py", line 4, in <module>
    decode_utf8_bytes_to_str_wrong("原神启动！".encode("utf-8"))
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/makima/Desktop/code/python projects/stf_course/assignment1-basics/cs336_basics/Part 2/temp_2_2.py", line 2, in decode_utf8_bytes_to_str_wrong
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])
                    ~~~~~~~~~~~~~~~~~^^^^^^^^^
UnicodeDecodeError: 'utf-8' codec can't decode byte 0xe5 in position 0: unexpected end of data
```

#### (c)

```text
mac $makima: ~/Desktop/code/python projects/stf_course ❯ uv run "/Users/makima/Desktop/code/python projects/stf_course/assignment1-basics/cs336_basics/Part 2/temp_2_2.py"
Traceback (most recent call last):
  File "/Users/makima/Desktop/code/python projects/stf_course/assignment1-basics/cs336_basics/Part 2/temp_2_2.py", line 6, in <module>
    b"\x80\x80".decode("utf-8")
    ~~~~~~~~~~~~~~~~~~^^^^^^^^^
UnicodeDecodeError: 'utf-8' codec can't decode byte 0x80 in position 0: invalid start byte
```

例如上面的后续字节。

### Problem(train_bpe)

见文件 [train_bpe.py](Part2/train_bpe.py) 和文件 [adapters.py](../tests/adapters.py)

### Problem(train_bpe_tinystories)

#### (a)

```json
{
  "input_path": "data/TinyStoriesV2-GPT4-train.txt",
  "vocab_size": 10000,
  "special_tokens": ["<|endoftext|>"],
  "workers": 9,
  "batch_bytes": 67108864,
  "unique_pretokens": 59933,
  "total_pretokens": 536592168,
  "actual_vocab_size": 10000,
  "merge_count": 9743,
  "pretokenization_seconds": 32.5644668749992,
  "bpe_training_seconds": 10.176330375001271,
  "longest_token": {
    "id": 9379,
    "hex": "20726573706f6e736962696c697479",
    "utf8": " responsibility",
    "byte_length": 15
  },
  "total_seconds": 42.76510929199867,
  "serialization_seconds": 0.015650207998987753,
  "peak_rss_mb": 3846.734375
}
```

代码见[train_bpe_tinystories.py](Part2/train_bpe_tinystories.py)。产物见[vocab.json](../artifacts/tinystories_bpe/vocab.json)、[merges.json](../artifacts/tinystories_bpe/merges.json)、[summary.json](../artifacts/tinystories_bpe/summary.json)。用时约42.765s，内存消耗约3846MB。最长的token是`responsibility`，他就是正常单词的含义。

#### (b)

根据[summary.json](../artifacts/tinystories_bpe/summary.json)中的数据，最耗时的是`预分词计数阶段（pretokenization）`

### Problem(train_bpe_expts_owt)

#### (a)

```json
{
  "input_path": "data/owt_train.txt",
  "vocab_size": 32000,
  "special_tokens": ["<|endoftext|>"],
  "workers": 9,
  "batch_bytes": 67108864,
  "progress_interval": 1000,
  "heap_rebuild_multiplier": 4,
  "show_progress": true,
  "unique_pretokens": 6601892,
  "total_pretokens": 2471753092,
  "actual_vocab_size": 32000,
  "merge_count": 31743,
  "pretokenization_seconds": 252.36145837499862,
  "bpe_training_seconds": 861.1913226249999,
  "longest_token": {
    "id": 25836,
    "hex": "2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d",
    "utf8": "----------------------------------------------------------------",
    "byte_length": 64
  },
  "total_seconds": 1113.658848250001,
  "serialization_seconds": 0.09106141699885484,
  "peak_rss_mb": 6677.4375
}
```

代码见[train_bpe_expts_owt.py](Part2/train_bpe_expts_owt.py)。产物见[vocab.json](../artifacts/owt_bpe/vocab.json)、[merges.json](../artifacts/owt_bpe/merges.json)、[summary.json](../artifacts/owt_bpe/summary.json)。最长的token是`----------------------------------------------------------------`，因为数据来自网页文本，markdown中分隔线在html上渲染后的效果就是如此。

#### (b)

TinyStories tokenizer更贴近儿童故事语域，词表更干净，长 token 多是常见英文完整词；OpenWebText tokenizer覆盖面更广，但会学习网页文本中的分隔线、重复标点、编码噪声等高频格式片段。

### Problem(tokenizer)

源码见[tokenizer.py](Part2/tokenizer.py)

```bash
mac $makima: ~/Desktop/code/python projects/stf_course/assignment1-basics on main ≡ ❯ uv run pytest tests/test_tokenizer.py                          75.63% 12/16GB

=============================== test session starts ===============================
platform darwin -- Python 3.13.13, pytest-9.0.2, pluggy-1.6.0
rootdir: /Users/makima/Desktop/code/python projects/stf_course/assignment1-basics
configfile: pyproject.toml
plugins: jaxtyping-0.3.9, timeout-2.4.0
collected 25 items

tests/test_tokenizer.py::test_roundtrip_empty PASSED
tests/test_tokenizer.py::test_empty_matches_tiktoken PASSED
tests/test_tokenizer.py::test_roundtrip_single_character PASSED
tests/test_tokenizer.py::test_single_character_matches_tiktoken PASSED
tests/test_tokenizer.py::test_roundtrip_single_unicode_character PASSED
tests/test_tokenizer.py::test_single_unicode_character_matches_tiktoken PASSED
tests/test_tokenizer.py::test_roundtrip_ascii_string PASSED
tests/test_tokenizer.py::test_ascii_string_matches_tiktoken PASSED
tests/test_tokenizer.py::test_roundtrip_unicode_string PASSED
tests/test_tokenizer.py::test_unicode_string_matches_tiktoken PASSED
tests/test_tokenizer.py::test_roundtrip_unicode_string_with_special_tokens PASSED
tests/test_tokenizer.py::test_unicode_string_with_special_tokens_matches_tiktoken PASSED
tests/test_tokenizer.py::test_overlapping_special_tokens PASSED
tests/test_tokenizer.py::test_address_roundtrip PASSED
tests/test_tokenizer.py::test_address_matches_tiktoken PASSED
tests/test_tokenizer.py::test_german_roundtrip PASSED
tests/test_tokenizer.py::test_german_matches_tiktoken PASSED
tests/test_tokenizer.py::test_tinystories_sample_roundtrip PASSED
tests/test_tokenizer.py::test_tinystories_matches_tiktoken PASSED
tests/test_tokenizer.py::test_encode_special_token_trailing_newlines PASSED
tests/test_tokenizer.py::test_encode_special_token_double_newline_non_whitespace PASSED
tests/test_tokenizer.py::test_encode_iterable_tinystories_sample_roundtrip PASSED
tests/test_tokenizer.py::test_encode_iterable_tinystories_matches_tiktoken PASSED
tests/test_tokenizer.py::test_encode_iterable_memory_usage SKIPPED (rli...)
tests/test_tokenizer.py::test_encode_memory_usage SKIPPED (rlimit suppo...)

========================== 23 passed, 2 skipped in 1.17s ==========================
```

其中2个skipped原因是因为我使用的是macos而非linux

```python
@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="rlimit support for non-linux systems is spotty.",
)
def test_encode_iterable_memory_usage():
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )
    with open(FIXTURES_PATH / "tinystories_sample_5M.txt") as f:
        ids = []
        for _id in _encode_iterable(tokenizer, f):
            ids.append(_id)


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="rlimit support for non-linux systems is spotty.",
)
@pytest.mark.xfail(reason="Tokenizer.encode is expected to take more memory than allotted (1MB).")
def test_encode_memory_usage():
    """
    We expect this test to fail, since Tokenizer.encode is not expected to be memory efficient.
    """
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )
    with open(FIXTURES_PATH / "tinystories_sample_5M.txt") as f:
        contents = f.read()
        _ = _encode(tokenizer, contents)

```

### Problem(tokenizer_experiments)

采用每个训练集前10个非空文档作为可复现样本，数据摘要在[tokenizer_experiments_report.json](../artifacts/tokenizer_experiments/tokenizer_experiments_report.json)中。

```text
TinyStories sample + TinyStories 10K tokenizer:
  bytes=7435
  tokens=1808
  bytes/token=4.112

OpenWebText sample + OpenWebText 32K tokenizer:
  bytes=31487
  tokens=6712
  bytes/token=4.691

OpenWebText sample + TinyStories 10K tokenizer:
  bytes=31487
  tokens=9873
  bytes/token=3.189

TinyStories 10K throughput on a 2MiB TinyStories prefix:
  2,111,372 bytes/s
  513,947 tokens/s

OpenWebText 32K throughput on a 2MiB OWT prefix:
  1,725,461 bytes/s
  389,811 tokens/s

Estimated time for 825GB Pile using the OWT 32K measured throughput:
  132.81 hours, about 5.53 days
```

#### (a)

由上面的数据可得知，TinyStories 10K tokenizer在TinyStories 10文档样本上的压缩率是4.112bytes/token；OpenWebText 32K tokenizer在OpenWebText 10文档样本上的压缩率是4.691 bytes/token。

#### (b)

TinyStories 10K tokenizer在OpenWebText 10文档样本上的压缩率是3.189 bytes/token。压缩率显著降低。

#### (c)

按照OpenWebText 32K tokenizer吞吐量1725461bytes/s来计算的话，大约需要132.81小时，约为5.53天。

#### (d)

[tokenizer_experiments.py](Part2/tokenizer_experiments.py)脚本提供完整数据集编码子命令:

```sh
uv run python cs336_basics/Part2/tokenizer_experiments.py \
  --mode encode-datasets \
  --output-dir artifacts/tokenizer_experiments \
  --overwrite
```

uint16最大可以表示65535，而两个tokenizer词表的最大大小为10000和32000，都小于uint16的上限。而uint16只需要int32一半的大小，可以显著减少磁盘和内存占用，且无损保存全部token id。

---

### Problem(linear)

源码见[linear.py](Part3/linear.py)。运行结果如下

```text
mac $makima: ~/Desktop/code/python projects/stf_course/assignment1-basics on main ↑1 ❯ uv run pytest -k test_linear                                  70.13% 11/16GB
=============================== test session starts ===============================
platform darwin -- Python 3.13.13, pytest-9.0.2, pluggy-1.6.0
rootdir: /Users/makima/Desktop/code/python projects/stf_course/assignment1-basics
configfile: pyproject.toml
plugins: jaxtyping-0.3.9, timeout-2.4.0
collected 49 items / 48 deselected / 1 selected

tests/test_model.py::test_linear PASSED

======================== 1 passed, 48 deselected in 0.07s =========================
```

### Problem(embedding)

源码见[embedding.py](Part3/embedding.py)。运行结果如下

```text
mac $makima: ~/Desktop/code/python projects/stf_course/assignment1-basics on main ↑1  ~3 ❯ uv run pytest -k test_embedding                          69.83% 11/16GB
=============================== test session starts ===============================
platform darwin -- Python 3.13.13, pytest-9.0.2, pluggy-1.6.0
rootdir: /Users/makima/Desktop/code/python projects/stf_course/assignment1-basics
configfile: pyproject.toml
plugins: jaxtyping-0.3.9, timeout-2.4.0
collected 49 items / 48 deselected / 1 selected

tests/test_model.py::test_embedding PASSED

======================== 1 passed, 48 deselected in 0.07s =========================
```

### Problem(rmsnorm)

源码见[rmsnorm.py](Part3/rmsnorm.py)。运行结果如下

```text
mac $makima: ~/Desktop/code/python projects/stf_course/assignment1-basics on main ↑1  ~4 ❯ uv run pytest -k test_rmsnorm                            70.74% 11/16GB
=============================== test session starts ===============================
platform darwin -- Python 3.13.13, pytest-9.0.2, pluggy-1.6.0
rootdir: /Users/makima/Desktop/code/python projects/stf_course/assignment1-basics
configfile: pyproject.toml
plugins: jaxtyping-0.3.9, timeout-2.4.0
collected 49 items / 48 deselected / 1 selected

tests/test_model.py::test_rmsnorm PASSED

======================== 1 passed, 48 deselected in 0.05s =========================
```

### Problem(positionwise_feedforward)

源码见[positionwise_feedforward.py](Part3/positionwise_feedforward.py)。运行结果如下

```text
mac $makima: ~/Desktop/code/python projects/stf_course/assignment1-basics on main ↑1  ~4 ❯ uv run pytest -k test_swiglu                              70.7% 11/16GB
=============================== test session starts ===============================
platform darwin -- Python 3.13.13, pytest-9.0.2, pluggy-1.6.0
rootdir: /Users/makima/Desktop/code/python projects/stf_course/assignment1-basics
configfile: pyproject.toml
plugins: jaxtyping-0.3.9, timeout-2.4.0
collected 49 items / 48 deselected / 1 selected

tests/test_model.py::test_swiglu PASSED

======================== 1 passed, 48 deselected in 0.05s =========================
```

### Problem(rope)

源码见[rope.py](Part3/rope.py)。运行结果如下

```text
mac $makima: ~/Desktop/code/python projects/stf_course/assignment1-basics on main ≡ ❯ uv run pytest -k test_rope                                     69.01% 11/16GB

=============================== test session starts ===============================
platform darwin -- Python 3.13.13, pytest-9.0.2, pluggy-1.6.0
rootdir: /Users/makima/Desktop/code/python projects/stf_course/assignment1-basics
configfile: pyproject.toml
plugins: jaxtyping-0.3.9, timeout-2.4.0
collected 49 items / 48 deselected / 1 selected

tests/test_model.py::test_rope PASSED

======================== 1 passed, 48 deselected in 0.04s =========================
```

### Problem(softmax)

源码见[softmax.py](Part3/softmax.py)。运行结果如下

```text
mac $makima: ~/Desktop/code/python projects/stf_course/assignment1-basics on main ↑1  ?1 ~2 ❯ uv run pytest -k test_softmax_matches_pytorch         72.74% 11/16GB
=============================== test session starts ===============================
platform darwin -- Python 3.13.13, pytest-9.0.2, pluggy-1.6.0
rootdir: /Users/makima/Desktop/code/python projects/stf_course/assignment1-basics
configfile: pyproject.toml
plugins: jaxtyping-0.3.9, timeout-2.4.0
collected 49 items / 48 deselected / 1 selected

tests/test_nn_utils.py::test_softmax_matches_pytorch PASSED

======================== 1 passed, 48 deselected in 0.05s =========================
```

### Problem(scaled_dot_product_attention)

源码见[scaled_dot_product_attention.py](Part3/scaled_dot_product_attention.py)。运行结果如下

```text
mac $makima: ~/Desktop/code/python projects/stf_course/assignment1-basics on main ≡  ?1 ~2 -2 ❯ uv run pytest -k test_scaled_dot_product_attention  65.29% 10/16GB
=============================== test session starts ===============================
platform darwin -- Python 3.13.13, pytest-9.0.2, pluggy-1.6.0
rootdir: /Users/makima/Desktop/code/python projects/stf_course/assignment1-basics
configfile: pyproject.toml
plugins: jaxtyping-0.3.9, timeout-2.4.0
collected 49 items / 48 deselected / 1 selected

tests/test_model.py::test_scaled_dot_product_attention PASSED

======================== 1 passed, 48 deselected in 0.06s =========================
mac $makima: ~/Desktop/code/python projects/stf_course/assignment1-basics on main ≡  ?1 ~2 -2 ❯ uv run pytest -k test_4d_scaled_dot_product_attention
=============================== test session starts ===============================
platform darwin -- Python 3.13.13, pytest-9.0.2, pluggy-1.6.0
rootdir: /Users/makima/Desktop/code/python projects/stf_course/assignment1-basics
configfile: pyproject.toml
plugins: jaxtyping-0.3.9, timeout-2.4.0
collected 49 items / 48 deselected / 1 selected

tests/test_model.py::test_4d_scaled_dot_product_attention PASSED

======================== 1 passed, 48 deselected in 0.04s =========================
```

### Problem(multihead_self_attention)

源码见[multihead_self_attention.py](Part3/multihead_self_attention.py)和[test_multihead_self_attention_with_rope.py](Part3/test_multihead_self_attention_with_rope.py)。运行结果如下

```text
mac $makima: ~/Desktop/code/python projects/stf_course/assignment1-basics on main ↑1  ?2 ~3 ❯ uv run pytest -k test_multihead_self_attention        65.73% 10/16GB
=============================== test session starts ===============================
platform darwin -- Python 3.13.13, pytest-9.0.2, pluggy-1.6.0
rootdir: /Users/makima/Desktop/code/python projects/stf_course/assignment1-basics
configfile: pyproject.toml
plugins: jaxtyping-0.3.9, timeout-2.4.0
collected 49 items / 47 deselected / 2 selected

tests/test_model.py::test_multihead_self_attention PASSED
tests/test_model.py::test_multihead_self_attention_with_rope PASSED

======================== 2 passed, 47 deselected in 0.05s =========================
```

### Problem(transformer_block)

源码见[transformer_block.py](Part3/transformer_block.py)。运行结果如下

```text
mac $makima: ~/Desktop/code/python projects/stf_course/assignment1-basics on main ↑2 ❯ uv run pytest -k test_transformer_block                       65.64% 10/16GB
=============================== test session starts ===============================
platform darwin -- Python 3.13.13, pytest-9.0.2, pluggy-1.6.0
rootdir: /Users/makima/Desktop/code/python projects/stf_course/assignment1-basics
configfile: pyproject.toml
plugins: jaxtyping-0.3.9, timeout-2.4.0
collected 49 items / 48 deselected / 1 selected

tests/test_model.py::test_transformer_block PASSED

======================== 1 passed, 48 deselected in 0.08s =========================
```

### Problem(transformer_lm)

源码见[transformer_lm.py](Part3/transformer_lm.py)。运行结果如下

```text
mac $makima: ~/Desktop/code/python projects/stf_course/assignment1-basics on main ↑3 ❯ uv run pytest -k test_transformer_lm                          64.39% 10/16GB
=============================== test session starts ===============================
platform darwin -- Python 3.13.13, pytest-9.0.2, pluggy-1.6.0
rootdir: /Users/makima/Desktop/code/python projects/stf_course/assignment1-basics
configfile: pyproject.toml
plugins: jaxtyping-0.3.9, timeout-2.4.0
collected 49 items / 47 deselected / 2 selected

tests/test_model.py::test_transformer_lm PASSED
tests/test_model.py::test_transformer_lm_truncated_input PASSED

======================== 2 passed, 47 deselected in 0.10s =========================
```

### Problem(transformer_accounting)

#### (a)

参数量为：

$$
2Vd + L(4d^2 + 3d\,d_{ff} + 2d) + d
= 1,640,452,800
$$

以FP32加载仅参数需要 `6,561,811,200 bytes`。

#### (b)

GPT-2 XL配置：`V=50,257`、`T=1,024`、`L=48`、`d=1,600`、`d_ff=4,288`。

| 矩阵乘法                              | 每层/模型 FLOPs |        总 FLOPs |
| ------------------------------------- | --------------: | --------------: |
| Q、K、V 投影：3 × `(T × d) @ (d × d)` |          `6Td²` |   754.97 GFLOPs |
| 注意力分数：各头 `Q @ Kᵀ`             |          `2T²d` |   161.06 GFLOPs |
| 注意力聚合：各头 `P @ V`              |          `2T²d` |   161.06 GFLOPs |
| 注意力输出投影：`(T × d) @ (d × d)`   |          `2Td²` |   251.66 GFLOPs |
| SwiGLU 三个投影：`W1`、`W3`、`W2`     |       `6Tdd_ff` | 2,023.33 GFLOPs |
| LM head：`(T × d) @ (d × V)`          |          `2TdV` |   164.68 GFLOPs |

总计为`3,516,769,894,400 FLOPs`。

#### (c)

XL在长度1,024时，SwiGLU前馈网络占约`57.5%`的FLOPs，是主要开销；注意力投影合计约`28.6%`。二次复杂度的注意力分数和加权求和仅约`9.2%`，因为此时`T`相对模型宽度仍较小。

#### (d)

| 模型                                     |     总 FLOPs |           注意力投影 |  注意力 `QKᵀ + PV` |           SwiGLU FFN |            LM head |
| ---------------------------------------- | -----------: | -------------------: | -----------------: | -------------------: | -----------------: |
| GPT-2 small (`L=12, d=768, d_ff=2048`)   | 0.292 TFLOPs |    57.98 GF (19.88%) |  38.65 GF (13.25%) |   115.96 GF (39.76%) |  79.05 GF (27.10%) |
| GPT-2 medium (`L=24, d=1024, d_ff=2752`) | 0.830 TFLOPs |   206.16 GF (24.83%) | 103.08 GF (12.42%) |   415.54 GF (50.05%) | 105.40 GF (12.70%) |
| GPT-2 large (`L=36, d=1280, d_ff=3392`)  | 1.769 TFLOPs |   483.18 GF (27.32%) | 193.27 GF (10.93%) |   960.33 GF (54.30%) |  131.75 GF (7.45%) |
| GPT-2 XL (`L=48, d=1600, d_ff=4288`)     | 3.517 TFLOPs | 1,006.63 GF (28.62%) |  322.12 GF (9.16%) | 2,023.33 GF (57.53%) |  164.68 GF (4.68%) |

随着模型加宽、加深，按`d²`或`d·d_ff`增长的注意力投影与FFN占比上升，其中FFN始终增长最快；LM head只随`d`增长，因此占比明显下降。固定context length时，按`T²d`增长的注意力矩阵乘法占比也会逐渐下降。

#### (e)

将XL的context length从1024提高到16384后，单次前向总量变为约`133.58 TFLOPs`，是原来的约`37.98倍`。注意力`QKᵀ + PV`的占比从`9.16%`增至`61.73%`，成为主导开销；FFN、注意力投影和LM head的占比分别降至约`24.24%`、`12.06%`与`1.97%`。

---

### Problem(cross_entropy)

源码见[cross_entropy.py](Part4/cross_entropy.py)。运行结果如下

```text
mac $makima: ~/Desktop/code/python projects/stf_course/assignment1-basics on main ≡  ?1 ❯ uv run pytest -k test_cross_entropy                       66.45% 10/16GB
=============================== test session starts ===============================
platform darwin -- Python 3.13.13, pytest-9.0.2, pluggy-1.6.0
rootdir: /Users/makima/Desktop/code/python projects/stf_course/assignment1-basics
configfile: pyproject.toml
plugins: jaxtyping-0.3.9, timeout-2.4.0
collected 49 items / 48 deselected / 1 selected

tests/test_nn_utils.py::test_cross_entropy PASSED

======================== 1 passed, 48 deselected in 0.06s =========================
```

### Problem(learning_rate_tuning)

学习率`1e1`的loss稳定且较快下降；`1e2`在第一步基本不变后迅速降至接近零。学习率`1e3`则明显发散，loss在10次迭代内从约`3e1`增长到约`3e18`。

### Problem(adamw)

源码见[adamw.py](Part4/adamw.py)。运行结果如下

```text
mac $makima: ~/Desktop/code/python projects/stf_course/assignment1-basics on main ↑1 ❯ uv run pytest -k test_adamw                                   66.11% 10/16GB
=============================== test session starts ===============================
platform darwin -- Python 3.13.13, pytest-9.0.2, pluggy-1.6.0
rootdir: /Users/makima/Desktop/code/python projects/stf_course/assignment1-basics
configfile: pyproject.toml
plugins: jaxtyping-0.3.9, timeout-2.4.0
collected 49 items / 48 deselected / 1 selected

tests/test_optimizer.py::test_adamw PASSED

======================== 1 passed, 48 deselected in 0.47s =========================
```
