from __future__ import annotations

import heapq
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import regex as re


GPT2_PRETOKENIZATION_PATTERN = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


Token = tuple[bytes, ...]
Pair = tuple[bytes, bytes]
IdPair = tuple[int, int]


@dataclass(frozen=True)
class _DescendingPairKey:
    """Heap key that preserves BPE's max-count, max-lexicographic tie-break."""

    pair: Pair

    def __lt__(self, other: _DescendingPairKey) -> bool:
        return self.pair > other.pair


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """训练字节级 BPE tokenizer，并返回词表和按生成顺序排列的 merges。"""
    normalized_special_tokens = validate_train_bpe_inputs(input_path, vocab_size, special_tokens)
    text = read_training_text(input_path)
    text_segments = split_on_special_tokens(text, normalized_special_tokens)
    word_counts = count_pretokens(text_segments)
    return train_bpe_from_pretoken_counts(
        word_counts=word_counts,
        vocab_size=vocab_size,
        special_tokens=normalized_special_tokens,
    )


def train_bpe_from_pretoken_counts(
    word_counts: Mapping[bytes, int],
    vocab_size: int,
    special_tokens: Sequence[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """基于已统计好的 pretoken 频次训练字节级 BPE。

    该入口用于大语料训练：调用方可以用多进程或流式方式先完成预分词计数，
    然后复用这里的增量 pair 统计和 merge 逻辑。
    """
    normalized_special_tokens = list(dict.fromkeys(special_tokens))
    minimum_vocab_size = 256 + len(normalized_special_tokens)
    if vocab_size < minimum_vocab_size:
        raise ValueError(
            "vocab_size 至少需要等于 256 加上唯一 special token 的数量 "
            f"({minimum_vocab_size})"
        )

    token_counts = initialize_byte_tokens(word_counts)
    vocab = initialize_vocab(normalized_special_tokens)
    merges: list[Pair] = []

    return train_bpe_from_token_counts(token_counts, vocab, merges, vocab_size)


def train_bpe_from_token_counts(
    token_counts: Counter[Token],
    vocab: dict[int, bytes],
    merges: list[Pair],
    vocab_size: int,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """使用增量 pair 索引训练 BPE merge。

    朴素实现每轮 merge 都重新扫描所有 pretoken。这里维护 pair -> word 的倒排索引，
    每次只更新包含当前 best pair 的 pretoken，避免 10k 词表训练时反复全量扫描。
    """
    word_tokens: list[list[int]] = []
    word_weights: list[int] = []
    pair_counts: dict[IdPair, int] = {}
    pair_to_word_counts: dict[IdPair, dict[int, int]] = {}
    heap: list[tuple[int, _DescendingPairKey, IdPair]] = []
    token_to_id = {token: token_id for token_id, token in vocab.items()}

    for token, count in token_counts.items():
        if count <= 0:
            continue
        word_id = len(word_tokens)
        token_ids = [token_to_id[part] for part in token]
        word_tokens.append(token_ids)
        word_weights.append(count)

        for pair, occurrences in count_token_id_pairs(token_ids).items():
            weighted_count = occurrences * count
            pair_counts[pair] = pair_counts.get(pair, 0) + weighted_count
            pair_to_word_counts.setdefault(pair, {})[word_id] = occurrences

    for pair, count in pair_counts.items():
        push_pair(heap, pair, count, vocab)

    while len(vocab) < vocab_size:
        best_pair = pop_best_pair(heap, pair_counts, vocab)
        if best_pair is None:
            break

        left_id, right_id = best_pair
        left_bytes = vocab[left_id]
        right_bytes = vocab[right_id]
        merged_token_id = len(vocab)
        vocab[merged_token_id] = left_bytes + right_bytes
        merges.append((left_bytes, right_bytes))

        affected_word_ids = list(pair_to_word_counts.get(best_pair, {}).keys())
        for word_id in affected_word_ids:
            old_token_ids = word_tokens[word_id]
            old_pair_counts = count_token_id_pairs(old_token_ids)
            if best_pair not in old_pair_counts:
                continue

            word_weight = word_weights[word_id]
            for pair, occurrences in old_pair_counts.items():
                weighted_count = occurrences * word_weight
                updated_count = pair_counts.get(pair, 0) - weighted_count
                if updated_count > 0:
                    pair_counts[pair] = updated_count
                    push_pair(heap, pair, updated_count, vocab)
                else:
                    pair_counts.pop(pair, None)

                word_map = pair_to_word_counts.get(pair)
                if word_map is not None:
                    word_map.pop(word_id, None)
                    if not word_map:
                        pair_to_word_counts.pop(pair, None)

            new_token_ids = merge_token_id_pair(old_token_ids, best_pair, merged_token_id)
            word_tokens[word_id] = new_token_ids

            for pair, occurrences in count_token_id_pairs(new_token_ids).items():
                weighted_count = occurrences * word_weight
                updated_count = pair_counts.get(pair, 0) + weighted_count
                pair_counts[pair] = updated_count
                pair_to_word_counts.setdefault(pair, {})[word_id] = occurrences
                push_pair(heap, pair, updated_count, vocab)

    return vocab, merges


def count_token_id_pairs(token_ids: Sequence[int]) -> dict[IdPair, int]:
    """统计单个 pretoken 内的相邻 token-id pair。"""
    pair_counts: dict[IdPair, int] = {}
    for index in range(len(token_ids) - 1):
        pair = (token_ids[index], token_ids[index + 1])
        pair_counts[pair] = pair_counts.get(pair, 0) + 1
    return pair_counts


def merge_token_id_pair(token_ids: Sequence[int], pair_to_merge: IdPair, merged_token_id: int) -> list[int]:
    """按 BPE 规则从左到右合并非重叠 pair。"""
    merged_token_ids: list[int] = []
    index = 0
    token_count = len(token_ids)

    while index < token_count:
        if index < token_count - 1 and (token_ids[index], token_ids[index + 1]) == pair_to_merge:
            merged_token_ids.append(merged_token_id)
            index += 2
        else:
            merged_token_ids.append(token_ids[index])
            index += 1

    return merged_token_ids


def push_pair(
    heap: list[tuple[int, _DescendingPairKey, IdPair]],
    pair: IdPair,
    count: int,
    vocab: Mapping[int, bytes],
) -> None:
    """将 pair 的当前计数压入堆，旧计数通过 lazy deletion 处理。"""
    if count <= 0:
        return
    pair_bytes = (vocab[pair[0]], vocab[pair[1]])
    heapq.heappush(heap, (-count, _DescendingPairKey(pair_bytes), pair))


def pop_best_pair(
    heap: list[tuple[int, _DescendingPairKey, IdPair]],
    pair_counts: Mapping[IdPair, int],
    vocab: Mapping[int, bytes],
) -> IdPair | None:
    """弹出仍然有效的最高频 pair，并按字节字典序处理平局。"""
    while heap:
        negative_count, pair_key, pair = heapq.heappop(heap)
        count = -negative_count
        current_count = pair_counts.get(pair, 0)
        if current_count != count:
            continue
        if pair_key.pair != (vocab[pair[0]], vocab[pair[1]]):
            continue
        return pair
    return None


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
    # 原始朴素版本：
    # pair_counts: Counter[Pair] = Counter()
    # for token, count in token_counts.items():
    #     for left, right in zip(token, token[1:]):
    #         pair_counts[(left, right)] += count
    #
    # 优化点：普通 dict + get 避免 Counter.__missing__；下标遍历避免 token[1:] 切片。
    pair_counts: dict[Pair, int] = {}
    for token, count in token_counts.items():
        for index in range(len(token) - 1):
            pair = (token[index], token[index + 1])
            pair_counts[pair] = pair_counts.get(pair, 0) + count
    return pair_counts


def select_best_pair(pair_counts: Mapping[Pair, int]) -> Pair:
    """选择下一组要 merge 的 token pair。"""
    return max(pair_counts.items(), key=lambda item: (item[1], item[0]))[0]
    # raise NotImplementedError("请实现 BPE pair 选择和平局处理逻辑。")


def apply_merge(token_counts: Counter[Token], pair_to_merge: Pair) -> Counter[Token]:
    """将选中的 merge 应用到每个已 tokenized 的 pretoken。"""
    merged_token_counts: dict[Token, int] = {}
    left_token, right_token = pair_to_merge
    merged_token = left_token + right_token

    for token, count in token_counts.items():
        # 原始朴素版本会无条件创建 list 并逐项 append：
        # merged_parts: list[bytes] = []
        # 即使当前 token 不包含 pair_to_merge，也会产生大量无效分配。
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
    # raise NotImplementedError("请实现 BPE merge 应用逻辑。")
