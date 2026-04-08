#!/usr/bin/env python3
"""Generate a random control head_roles.json from an existing probing file.

The generated file preserves:
- number of layers
- number of heads per layer
- per-layer counts of img/text/shared heads

It changes the concrete head ids by applying a layer-wise random derangement
of head indices. We additionally search multiple derangements and keep the one
that minimizes overlap with the original role assignments for the same layer.
"""

from __future__ import annotations

import argparse
import json
import random
from copy import deepcopy
from pathlib import Path
from typing import Dict, List


def random_derangement(n: int, rng: random.Random) -> List[int]:
    """Return a random derangement of range(n)."""
    items = list(range(n))
    while True:
        perm = items[:]
        rng.shuffle(perm)
        if all(i != perm[i] for i in items):
            return perm


def remap_roles_with_best_derangement(
    layer_roles: Dict[str, List[int]],
    num_heads: int,
    seed: int,
    search_steps: int,
) -> Dict[str, List[int]]:
    """Search derangements and keep the one with the smallest same-role overlap."""
    rng = random.Random(seed)
    best_perm = None
    best_score = None

    old_img = set(layer_roles.get("img_heads", []))
    old_text = set(layer_roles.get("text_heads", []))
    old_shared = set(layer_roles.get("shared_heads", []))

    for _ in range(search_steps):
        perm = random_derangement(num_heads, rng)
        new_img = {perm[h] for h in old_img}
        new_text = {perm[h] for h in old_text}
        new_shared = {perm[h] for h in old_shared}

        # Minimize staying in the same semantic bucket first, then union overlap.
        same_role_overlap = len(new_img & old_img) + len(new_text & old_text) + len(new_shared & old_shared)
        union_overlap = len((new_img | new_text | new_shared) & (old_img | old_text | old_shared))
        score = (same_role_overlap, union_overlap)

        if best_score is None or score < best_score:
            best_score = score
            best_perm = perm
            if score == (0, 0):
                break

    assert best_perm is not None
    return {
        "img_heads": [best_perm[h] for h in layer_roles.get("img_heads", [])],
        "text_heads": [best_perm[h] for h in layer_roles.get("text_heads", [])],
        "shared_heads": [best_perm[h] for h in layer_roles.get("shared_heads", [])],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to the original head_roles.json")
    parser.add_argument("--output", required=True, help="Path to the generated control head_roles.json")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument(
        "--search_steps",
        type=int,
        default=4000,
        help="Number of random derangements searched per layer",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    with in_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    meta = deepcopy(data.get("meta", {}))
    head_roles = data.get("head_roles", {})
    num_heads = int(meta["num_heads"])

    new_head_roles = {}
    overlap_report = {}

    for layer_str, layer_roles in head_roles.items():
        new_roles = remap_roles_with_best_derangement(
            layer_roles=layer_roles,
            num_heads=num_heads,
            seed=args.seed + int(layer_str),
            search_steps=args.search_steps,
        )

        old_img = set(layer_roles.get("img_heads", []))
        old_text = set(layer_roles.get("text_heads", []))
        old_shared = set(layer_roles.get("shared_heads", []))
        new_img = set(new_roles.get("img_heads", []))
        new_text = set(new_roles.get("text_heads", []))
        new_shared = set(new_roles.get("shared_heads", []))

        overlap_report[layer_str] = {
            "img_overlap": len(old_img & new_img),
            "text_overlap": len(old_text & new_text),
            "shared_overlap": len(old_shared & new_shared),
        }

        new_head_roles[layer_str] = {
            "img_heads": new_roles["img_heads"],
            "text_heads": new_roles["text_heads"],
            "shared_heads": new_roles["shared_heads"],
        }

    meta["role_assignment_method"] = "layerwise_random_derangement_control"
    meta["source_head_roles"] = str(in_path)
    meta["random_seed"] = args.seed
    meta["search_steps"] = args.search_steps

    out = {
        "meta": meta,
        "head_roles": new_head_roles,
        "overlap_report": overlap_report,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    total_img = sum(len(v.get("img_heads", [])) for v in new_head_roles.values())
    total_text = sum(len(v.get("text_heads", [])) for v in new_head_roles.values())
    img_overlap = sum(v["img_overlap"] for v in overlap_report.values())
    text_overlap = sum(v["text_overlap"] for v in overlap_report.values())
    shared_overlap = sum(v["shared_overlap"] for v in overlap_report.values())

    print(f"Saved random control head roles to: {out_path}")
    print(f"Total roles: img={total_img}, text={total_text}")
    print(
        "Same-role overlaps with original: "
        f"img={img_overlap}, text={text_overlap}, shared={shared_overlap}"
    )


if __name__ == "__main__":
    main()
