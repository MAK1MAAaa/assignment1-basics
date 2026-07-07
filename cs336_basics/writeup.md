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
  ```text
  total_seconds=99.963
  pretokenization_seconds=85.299
  bpe_training_seconds=14.621
  serialization_seconds=0.035
  peak_rss_mb=3737.2
  vocab_size=10000
  merges=9743
  longest_token=id=9379 bytes=15 utf8=' responsibility'
  ```

  产物见 [vocab.json](../artifacts/tinystories_bpe/vocab.json) 、 [merges.json](../artifacts/tinystories_bpe/merges.json) 、 [summary.json](../artifacts/tinystories_bpe/summary.json)