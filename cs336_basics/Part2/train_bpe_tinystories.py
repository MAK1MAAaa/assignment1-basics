from __future__ import annotations

import argparse
import json
import os
import queue
import threading
import time
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import psutil

from cs336_basics.Part2.train_bpe import (
    count_pretokens,
    split_on_special_tokens,
    train_bpe_from_pretoken_counts,
    validate_train_bpe_inputs,
)


DEFAULT_INPUT_PATH = Path("data/TinyStoriesV2-GPT4-train.txt")
DEFAULT_OUTPUT_DIR = Path("artifacts/tinystories_bpe")
DEFAULT_VOCAB_SIZE = 10_000
DEFAULT_SPECIAL_TOKEN = "<|endoftext|>"
DEFAULT_BATCH_BYTES = 64 * 1024 * 1024
DEFAULT_SAMPLE_INTERVAL_SECONDS = 0.2

_worker_special_tokens: tuple[str, ...] = ()


@dataclass(frozen=True)
class TrainingMetrics:
    total_seconds: float
    pretokenization_seconds: float
    bpe_training_seconds: float
    serialization_seconds: float
    peak_rss_mb: float
    unique_pretokens: int
    total_pretokens: int


class ProcessTreeMemorySampler:
    """Periodically samples RSS for the current process and its children."""

    def __init__(self, interval_seconds: float = DEFAULT_SAMPLE_INTERVAL_SECONDS) -> None:
        self.interval_seconds = interval_seconds
        self.peak_rss_bytes = 0
        self._process = psutil.Process(os.getpid())
        self._stop_event = threading.Event()
        self._errors: queue.SimpleQueue[Exception] = queue.SimpleQueue()
        self._thread = threading.Thread(target=self._sample_until_stopped, daemon=True)

    def __enter__(self) -> ProcessTreeMemorySampler:
        self._thread.start()
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self._stop_event.set()
        self._thread.join()

    @property
    def peak_rss_mb(self) -> float:
        return self.peak_rss_bytes / (1024 * 1024)

    def _sample_until_stopped(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.peak_rss_bytes = max(self.peak_rss_bytes, self._current_tree_rss_bytes())
            except Exception as exc:  # pragma: no cover - defensive sampler path
                self._errors.put(exc)
            self._stop_event.wait(self.interval_seconds)

    def _current_tree_rss_bytes(self) -> int:
        rss_bytes = self._process.memory_info().rss
        for child in self._process.children(recursive=True):
            try:
                rss_bytes += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return rss_bytes


def initialize_worker(special_tokens: Sequence[str]) -> None:
    global _worker_special_tokens
    _worker_special_tokens = tuple(special_tokens)


def count_batch_pretokens(batch_bytes: bytes) -> Counter[bytes]:
    text = batch_bytes.decode("utf-8")
    text_segments = split_on_special_tokens(text, _worker_special_tokens)
    return count_pretokens(text_segments)


def iter_special_token_batches(
    input_path: Path,
    special_token: str,
    target_batch_bytes: int,
) -> Iterator[bytes]:
    delimiter = special_token.encode("utf-8")
    read_size = min(target_batch_bytes, 16 * 1024 * 1024)
    remainder = b""
    batch = bytearray()

    with input_path.open("rb") as input_file:
        while chunk := input_file.read(read_size):
            data = remainder + chunk
            documents = data.split(delimiter)
            remainder = documents.pop()

            for document in documents:
                batch.extend(document)
                batch.extend(delimiter)
                if len(batch) >= target_batch_bytes:
                    yield bytes(batch)
                    batch.clear()

    if remainder:
        batch.extend(remainder)
    if batch:
        yield bytes(batch)


def count_pretokens_parallel(
    input_path: Path,
    special_tokens: Sequence[str],
    workers: int,
    batch_bytes: int,
) -> Counter[bytes]:
    pretoken_counts: Counter[bytes] = Counter()
    special_token = special_tokens[0]
    batches = iter_special_token_batches(input_path, special_token, batch_bytes)

    if workers == 1:
        initialize_worker(special_tokens)
        for batch_counts in map(count_batch_pretokens, batches):
            pretoken_counts.update(batch_counts)
        return pretoken_counts

    with Pool(processes=workers, initializer=initialize_worker, initargs=(tuple(special_tokens),)) as pool:
        for batch_counts in pool.imap_unordered(count_batch_pretokens, batches, chunksize=1):
            pretoken_counts.update(batch_counts)

    return pretoken_counts


def serialize_training_artifacts(
    output_dir: Path,
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    summary: dict[str, Any],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    vocab_path = output_dir / "vocab.json"
    merges_path = output_dir / "merges.json"
    summary_path = output_dir / "summary.json"

    vocab_payload = {
        str(token_id): {
            "hex": token_bytes.hex(),
            "utf8": decode_for_inspection(token_bytes),
            "byte_length": len(token_bytes),
        }
        for token_id, token_bytes in vocab.items()
    }
    merges_payload = [
        {
            "rank": rank,
            "left_hex": left.hex(),
            "right_hex": right.hex(),
            "left_utf8": decode_for_inspection(left),
            "right_utf8": decode_for_inspection(right),
        }
        for rank, (left, right) in enumerate(merges)
    ]

    vocab_path.write_text(json.dumps(vocab_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    merges_path.write_text(json.dumps(merges_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    return {"vocab": vocab_path, "merges": merges_path, "summary": summary_path}


def decode_for_inspection(token_bytes: bytes) -> str:
    return token_bytes.decode("utf-8", errors="replace")


def find_longest_token(vocab: dict[int, bytes]) -> tuple[int, bytes]:
    token_id, token_bytes = max(vocab.items(), key=lambda item: (len(item[1]), item[0]))
    return token_id, token_bytes


def train_tinystories_bpe(
    input_path: Path,
    output_dir: Path,
    vocab_size: int,
    special_tokens: Sequence[str],
    workers: int,
    batch_bytes: int,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]], TrainingMetrics, dict[str, Path]]:
    normalized_special_tokens = validate_train_bpe_inputs(
        input_path=str(input_path),
        vocab_size=vocab_size,
        special_tokens=list(special_tokens),
    )

    total_start = time.perf_counter()
    with ProcessTreeMemorySampler() as memory_sampler:
        pretokenization_start = time.perf_counter()
        word_counts = count_pretokens_parallel(
            input_path=input_path,
            special_tokens=normalized_special_tokens,
            workers=workers,
            batch_bytes=batch_bytes,
        )
        pretokenization_seconds = time.perf_counter() - pretokenization_start

        bpe_training_start = time.perf_counter()
        vocab, merges = train_bpe_from_pretoken_counts(
            word_counts=word_counts,
            vocab_size=vocab_size,
            special_tokens=normalized_special_tokens,
        )
        bpe_training_seconds = time.perf_counter() - bpe_training_start

        longest_token_id, longest_token = find_longest_token(vocab)
        serialization_start = time.perf_counter()
        metrics_without_serialization = {
            "input_path": str(input_path),
            "vocab_size": vocab_size,
            "special_tokens": normalized_special_tokens,
            "workers": workers,
            "batch_bytes": batch_bytes,
            "unique_pretokens": len(word_counts),
            "total_pretokens": sum(word_counts.values()),
            "actual_vocab_size": len(vocab),
            "merge_count": len(merges),
            "pretokenization_seconds": pretokenization_seconds,
            "bpe_training_seconds": bpe_training_seconds,
            "longest_token": {
                "id": longest_token_id,
                "hex": longest_token.hex(),
                "utf8": decode_for_inspection(longest_token),
                "byte_length": len(longest_token),
            },
        }
        artifact_paths = serialize_training_artifacts(
            output_dir=output_dir,
            vocab=vocab,
            merges=merges,
            summary=metrics_without_serialization,
        )
        serialization_seconds = time.perf_counter() - serialization_start

    total_seconds = time.perf_counter() - total_start
    metrics = TrainingMetrics(
        total_seconds=total_seconds,
        pretokenization_seconds=pretokenization_seconds,
        bpe_training_seconds=bpe_training_seconds,
        serialization_seconds=serialization_seconds,
        peak_rss_mb=memory_sampler.peak_rss_mb,
        unique_pretokens=len(word_counts),
        total_pretokens=sum(word_counts.values()),
    )

    longest_token_id, longest_token = find_longest_token(vocab)
    final_summary = {
        **metrics_without_serialization,
        "total_seconds": total_seconds,
        "serialization_seconds": serialization_seconds,
        "peak_rss_mb": memory_sampler.peak_rss_mb,
        "longest_token": {
            "id": longest_token_id,
            "hex": longest_token.hex(),
            "utf8": decode_for_inspection(longest_token),
            "byte_length": len(longest_token),
        },
    }
    artifact_paths["summary"].write_text(json.dumps(final_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    return vocab, merges, metrics, artifact_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a 10k byte-level BPE tokenizer on TinyStories.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH, help="TinyStories training text path.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for JSON artifacts.")
    parser.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE, help="Maximum vocabulary size.")
    parser.add_argument(
        "--special-token",
        action="append",
        default=[DEFAULT_SPECIAL_TOKEN],
        help="Special token to add. May be passed multiple times.",
    )
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1), help="Pretokenizer workers.")
    parser.add_argument("--batch-bytes", type=int, default=DEFAULT_BATCH_BYTES, help="Bytes per pretokenization batch.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    vocab, merges, metrics, artifact_paths = train_tinystories_bpe(
        input_path=args.input,
        output_dir=args.output_dir,
        vocab_size=args.vocab_size,
        special_tokens=args.special_token,
        workers=args.workers,
        batch_bytes=args.batch_bytes,
    )
    longest_token_id, longest_token = find_longest_token(vocab)

    print(f"vocab_size={len(vocab)}")
    print(f"merges={len(merges)}")
    print(f"total_seconds={metrics.total_seconds:.3f}")
    print(f"pretokenization_seconds={metrics.pretokenization_seconds:.3f}")
    print(f"bpe_training_seconds={metrics.bpe_training_seconds:.3f}")
    print(f"serialization_seconds={metrics.serialization_seconds:.3f}")
    print(f"peak_rss_mb={metrics.peak_rss_mb:.1f}")
    print(f"unique_pretokens={metrics.unique_pretokens}")
    print(f"total_pretokens={metrics.total_pretokens}")
    print(
        "longest_token="
        f"id={longest_token_id} "
        f"bytes={len(longest_token)} "
        f"utf8={decode_for_inspection(longest_token)!r} "
        f"hex={longest_token.hex()}"
    )
    print(f"vocab_path={artifact_paths['vocab']}")
    print(f"merges_path={artifact_paths['merges']}")
    print(f"summary_path={artifact_paths['summary']}")


if __name__ == "__main__":
    main()
