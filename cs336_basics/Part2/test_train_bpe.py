from __future__ import annotations

from collections import Counter

from cs336_basics.Part2.train_bpe import Token, apply_merge


def test_apply_merge_replaces_non_overlapping_pairs_and_preserves_counts() -> None:
    """验证 apply_merge 会合并目标 pair，并保留每个 pretoken 的出现次数。"""
    token_counts: Counter[Token] = Counter(
        {
            (b"l", b"o", b"w"): 3,
            (b"l", b"o", b"w", b"e", b"r"): 2,
            (b"a", b"a", b"a"): 4,
        }
    )

    merged_token_counts = apply_merge(token_counts, (b"o", b"w"))

    assert merged_token_counts == Counter(
        {
            (b"l", b"ow"): 3,
            (b"l", b"ow", b"e", b"r"): 2,
            (b"a", b"a", b"a"): 4,
        }
    )

    overlapping_token_counts: Counter[Token] = Counter({(b"a", b"a", b"a"): 1})

    assert apply_merge(overlapping_token_counts, (b"a", b"a")) == Counter({(b"aa", b"a"): 1})
