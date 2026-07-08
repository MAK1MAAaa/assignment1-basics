### Problem(unicode1)

- (a)
  空字符NUL

- (b)
  可以输出内容`'\x00'`

- (c)
  `chr(0)`不会显示出来，当他在句子中时候，下一个字符会直接输出在他的位置（ghostty+zsh）

### Problem(unicode2)

- (a)
  | 编码类型 | 占字节数 |
  | -------- | -------- |
  | UTF-8 | 1-4 |
  | UTF-16 | 2-4 |
  | UTF-32 | 4 |

  超过98%的网页都在使用utf8编码，而且英文在utf8下仅占1个字节，可以显著的节约空间。

- (b)
  这个函数仅能在纯英文环境下才能正常运行，例如使用中文或者emoji都会导致报错。

  ```bash
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

- (c)
  ```bash
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

- (a)

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

- (b)
  根据[summary.json](../artifacts/tinystories_bpe/summary.json)中的数据，最耗时的是`预分词计数阶段（pretokenization）`

### Problem(train_bpe_expts_owt)

- (a)
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
