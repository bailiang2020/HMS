#!/usr/bin/env python3
import argparse
import pickle
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a fixed stratified subset of the DGM4 test pickle for faster masking analysis."
    )
    parser.add_argument("--input_pkl", type=str, required=True, help="Path to the original DGM4 test.pkl")
    parser.add_argument("--output_pkl", type=str, required=True, help="Path to the output subset pickle")
    parser.add_argument("--subset_size", type=int, default=5000, help="Number of samples to keep")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def summarize_labels(data, prefix: str):
    for key in ("label", "text_label", "img_label"):
        values = data.get(key)
        if isinstance(values, (list, tuple)) and values:
            try:
                counts = Counter(int(v) for v in values)
            except Exception:
                counts = Counter(values)
            print(f"[{prefix}] {key}: total={len(values)} counts={dict(counts)}")


def infer_length(data: dict) -> int:
    labels = data.get("label")
    if not isinstance(labels, (list, tuple, np.ndarray)):
        raise ValueError("Expected 'label' to be a sequence in the pickle.")
    return len(labels)


def build_group_keys(data: dict, n: int):
    available_keys = [key for key in ("label", "text_label", "img_label") if isinstance(data.get(key), (list, tuple, np.ndarray))]
    if not available_keys:
        raise ValueError("None of label/text_label/img_label is available for stratification.")

    group_keys = []
    for idx in range(n):
        key = tuple(
            int(data[field][idx]) if field in data and isinstance(data[field], (list, tuple, np.ndarray)) else None
            for field in available_keys
        )
        group_keys.append(key)
    return available_keys, group_keys


def stratified_sample_indices(group_keys, subset_size: int, seed: int):
    rng = random.Random(seed)
    buckets = defaultdict(list)
    for idx, key in enumerate(group_keys):
        buckets[key].append(idx)

    total = len(group_keys)
    if subset_size > total:
        raise ValueError(f"subset_size={subset_size} exceeds total size={total}")

    exact_targets = {}
    base_targets = {}
    remainders = []
    assigned = 0

    for key, indices in buckets.items():
        exact = len(indices) * subset_size / total
        base = min(len(indices), int(exact))
        exact_targets[key] = exact
        base_targets[key] = base
        assigned += base
        remainders.append((exact - base, key))

    remaining = subset_size - assigned
    remainders.sort(reverse=True)
    for _, key in remainders:
        if remaining <= 0:
            break
        if base_targets[key] < len(buckets[key]):
            base_targets[key] += 1
            remaining -= 1

    if remaining != 0:
        raise RuntimeError(f"Failed to allocate exact subset size. remaining={remaining}")

    selected = []
    for key, indices in buckets.items():
        take = base_targets[key]
        if take <= 0:
            continue
        selected.extend(rng.sample(indices, take))

    selected.sort()
    if len(selected) != subset_size:
        raise RuntimeError(f"Selected {len(selected)} indices, expected {subset_size}")
    return selected


def subset_value(value, indices):
    if isinstance(value, list):
        return [value[i] for i in indices]
    if isinstance(value, tuple):
        return tuple(value[i] for i in indices)
    if isinstance(value, np.ndarray):
        return value[indices]
    return value


def main():
    args = parse_args()
    input_pkl = Path(args.input_pkl)
    output_pkl = Path(args.output_pkl)

    with open(input_pkl, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected pickle to contain a dict, got {type(data)}")

    n = infer_length(data)
    summarize_labels(data, prefix="full")
    available_keys, group_keys = build_group_keys(data, n)
    print(f"[subset] stratifying on keys={available_keys}")
    indices = stratified_sample_indices(group_keys, subset_size=args.subset_size, seed=args.seed)

    subset = {}
    for key, value in data.items():
        try:
            if len(value) == n:
                subset[key] = subset_value(value, indices)
            else:
                subset[key] = value
        except Exception:
            subset[key] = value

    output_pkl.parent.mkdir(parents=True, exist_ok=True)
    with open(output_pkl, "wb") as f:
        pickle.dump(subset, f)

    summarize_labels(subset, prefix="subset")
    print(f"[subset] saved {args.subset_size} samples to {output_pkl}")


if __name__ == "__main__":
    main()
