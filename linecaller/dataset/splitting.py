from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetSplit:
    train: tuple[str, ...]
    val: tuple[str, ...]
    test: tuple[str, ...]


def split_clips(
    clip_ids: list[str],
    *,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> DatasetSplit:

    if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio >= 1:
        raise ValueError("Invalid split ratios")

    ids = list(clip_ids)
    random.Random(seed).shuffle(ids)

    n = len(ids)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))

    n_train = min(n_train, n)
    n_val = min(n_val, n - n_train)

    train = ids[:n_train]
    val = ids[n_train:n_train+n_val]
    test = ids[n_train+n_val:]

    return DatasetSplit(
        train=tuple(train),
        val=tuple(val),
        test=tuple(test),
    )
