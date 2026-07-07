from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import regex as re


GPT2_PRETOKENIZATION_PATTERN = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


Token = tuple[bytes, ...]
Pair = tuple[bytes, bytes]


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """训练字节级 BPE tokenizer，并返回词表和按生成顺序排列的 merges。

    该函数保留作业 Part 2 的朴素训练流程：
    1. 读取输入语料。
    2. 按 special token 切分文本，确保 merge 不会跨过 special token 的边界。
    3. 使用 GPT-2 正则表达式对普通文本片段做预分词。
    4. 将每个 pretoken 初始化为字节 token 序列。
    5. 每轮全量统计相邻 token pair，选择最佳 pair 并执行 merge。
    6. 基于初始字节 token、special token 和学习到的 merge 构建最终词表。
    """
    normalized_special_tokens = validate_train_bpe_inputs(input_path, vocab_size, special_tokens)
    text = read_training_text(input_path)
    text_segments = split_on_special_tokens(text, normalized_special_tokens)
    word_counts = count_pretokens(text_segments)
    token_counts = initialize_byte_tokens(word_counts)
    vocab = initialize_vocab(normalized_special_tokens)
    merges: list[Pair] = []

    while len(vocab) < vocab_size:
        pair_counts = count_adjacent_pairs(token_counts)
        if not pair_counts:
            break

        best_pair = select_best_pair(pair_counts)
        token_counts = apply_merge(token_counts, best_pair)
        merges.append(best_pair)
        vocab[len(vocab)] = best_pair[0] + best_pair[1]

    return vocab, merges


def validate_train_bpe_inputs(
    input_path: str,
    vocab_size: int,
    special_tokens: Sequence[str],
) -> list[str]:
    """校验公开 API 的输入，并返回去重后的 special tokens。

    special tokens 按首次出现顺序保留，保证词表 ID 分配结果确定。
    """
    if not Path(input_path).is_file():
        raise FileNotFoundError(f"BPE 训练输入文件不存在: {input_path}")
    if vocab_size <= 0:
        raise ValueError("vocab_size 必须是正整数")
    if not all(isinstance(token, str) for token in special_tokens):
        raise TypeError("special_tokens 只能包含字符串")

    deduplicated_special_tokens = list(dict.fromkeys(special_tokens))
    minimum_vocab_size = 256 + len(deduplicated_special_tokens)
    if vocab_size < minimum_vocab_size:
        raise ValueError(
            "vocab_size 至少需要等于 256 加上唯一 special token 的数量 "
            f"({minimum_vocab_size})"
        )

    return deduplicated_special_tokens


def read_training_text(input_path: str) -> str:
    """读取 UTF-8 编码的训练数据。"""
    return Path(input_path).read_text(encoding="utf-8")


def split_on_special_tokens(text: str, special_tokens: Sequence[str]) -> list[str]:
    """按 special token 将文本切分为普通片段，并丢弃 special token 片段。

    special tokens 是硬边界：它们不能参与 merge 统计，merge 也不能跨过它们的范围。
    """
    if not special_tokens:
        return [text]

    escaped_tokens = [re.escape(token) for token in sorted(special_tokens, key=len, reverse=True)]
    special_token_pattern = "|".join(escaped_tokens)
    return [segment for segment in re.split(special_token_pattern, text) if segment]


def count_pretokens(text_segments: Iterable[str]) -> Counter[bytes]:
    """对文本片段做预分词，并按 UTF-8 bytes 统计每个 pretoken。"""
    pretoken_counts: Counter[bytes] = Counter()
    pattern = re.compile(GPT2_PRETOKENIZATION_PATTERN)

    for segment in text_segments:
        for match in pattern.finditer(segment):
            pretoken_counts[match.group(0).encode("utf-8")] += 1

    return pretoken_counts


def initialize_byte_tokens(word_counts: Mapping[bytes, int]) -> Counter[Token]:
    """将每个已计数的 pretoken 表示为单字节 BPE token 组成的元组。"""
    token_counts: Counter[Token] = Counter()
    for word, count in word_counts.items():
        token_counts[tuple(bytes([byte]) for byte in word)] += count
    return token_counts


def initialize_vocab(special_tokens: Sequence[str]) -> dict[int, bytes]:
    """创建初始字节级词表，并追加 special tokens。"""
    vocab: dict[int, bytes] = {byte: bytes([byte]) for byte in range(256)}
    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode("utf-8")
    return vocab


def count_adjacent_pairs(token_counts: Counter[Token]) -> dict[Pair, int]:
    """统计相邻 BPE token pair，并按 pretoken 频次加权。"""
    pair_counts: dict[Pair, int] = {}
    for token, count in token_counts.items():
        for index in range(len(token) - 1):
            pair = (token[index], token[index + 1])
            pair_counts[pair] = pair_counts.get(pair, 0) + count
    return pair_counts


def select_best_pair(pair_counts: Mapping[Pair, int]) -> Pair:
    """选择下一组要 merge 的 token pair：先比较频次，再按字节字典序打破平局。"""
    return max(pair_counts.items(), key=lambda item: (item[1], item[0]))[0]


def apply_merge(token_counts: Counter[Token], pair_to_merge: Pair) -> Counter[Token]:
    """将选中的 merge 应用到每个已 tokenized 的 pretoken。"""
    merged_token_counts: dict[Token, int] = {}
    left_token, right_token = pair_to_merge
    merged_token = left_token + right_token

    for token, count in token_counts.items():
        merged_parts: list[bytes] | None = None
        index = 0
        token_length = len(token)

        while index < token_length:
            if (
                index < token_length - 1
                and token[index] == left_token
                and token[index + 1] == right_token
            ):
                if merged_parts is None:
                    merged_parts = list(token[:index])
                merged_parts.append(merged_token)
                index += 2
            else:
                if merged_parts is not None:
                    merged_parts.append(token[index])
                index += 1

        if merged_parts is None:
            merged_token_counts[token] = merged_token_counts.get(token, 0) + count
        else:
            merged_token_tuple = tuple(merged_parts)
            merged_token_counts[merged_token_tuple] = merged_token_counts.get(merged_token_tuple, 0) + count

    return Counter(merged_token_counts)
