#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot masking curves for image/text critical and random head masking."
    )
    parser.add_argument("--text_critical_csv", type=Path, required=True)
    parser.add_argument("--text_random_csv", type=Path, required=True)
    parser.add_argument("--img_critical_csv", type=Path, required=True)
    parser.add_argument("--img_random_csv", type=Path, required=True)
    parser.add_argument("--text_baseline", type=float, required=True)
    parser.add_argument("--img_baseline", type=float, required=True)
    parser.add_argument("--title", type=str, default="")
    parser.add_argument("--paper_style", action="store_true")
    parser.add_argument(
        "--output_stem",
        type=Path,
        required=True,
        help="Output path without extension, e.g. results/mask_sweeps/plots/mix_mask_curves",
    )
    return parser.parse_args()


def load_curve(csv_path: Path) -> list[tuple[int, float]]:
    rows: list[tuple[int, float]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get("return_code", "")).strip() != "0":
                continue
            k_str = str(row.get("mask_count", "")).strip()
            f1_str = str(row.get("macro_avg_f1", "")).strip()
            if not k_str or not f1_str:
                continue
            rows.append((int(k_str), float(f1_str)))
    rows.sort(key=lambda x: x[0])
    dedup: dict[int, float] = {}
    for k, v in rows:
        dedup[k] = v
    return sorted(dedup.items(), key=lambda x: x[0])


def with_baseline(curve: list[tuple[int, float]], baseline: float) -> tuple[list[int], list[float]]:
    xs = [0]
    ys = [baseline]
    for k, v in curve:
        if k == 0:
            ys[0] = v
        else:
            xs.append(k)
            ys.append(v)
    return xs, ys


def write_clean_csv(
    out_csv: Path,
    text_critical: tuple[list[int], list[float]],
    text_random: tuple[list[int], list[float]],
    img_critical: tuple[list[int], list[float]],
    img_random: tuple[list[int], list[float]],
) -> None:
    k_values = sorted(
        set(text_critical[0]) | set(text_random[0]) | set(img_critical[0]) | set(img_random[0])
    )

    def to_map(curve: tuple[list[int], list[float]]) -> dict[int, float]:
        return {k: v for k, v in zip(curve[0], curve[1])}

    tc = to_map(text_critical)
    tr = to_map(text_random)
    ic = to_map(img_critical)
    ir = to_map(img_random)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "mask_count",
                "text_critical_macro_f1",
                "text_random_macro_f1",
                "img_critical_macro_f1",
                "img_random_macro_f1",
            ]
        )
        for k in k_values:
            writer.writerow(
                [
                    k,
                    "" if k not in tc else f"{tc[k]:.6f}",
                    "" if k not in tr else f"{tr[k]:.6f}",
                    "" if k not in ic else f"{ic[k]:.6f}",
                    "" if k not in ir else f"{ir[k]:.6f}",
                ]
            )


def plot_panel(
    ax,
    critical_curve: tuple[list[int], list[float]],
    random_curve: tuple[list[int], list[float]],
    ylabel: str,
    panel_title: str,
    paper_style: bool,
) -> None:
    crit_x, crit_y = critical_curve
    rand_x, rand_y = random_curve
    all_x = sorted(set(crit_x) | set(rand_x))

    crit_marker_size = 4.8 if paper_style else 6.5
    rand_marker_size = 4.6 if paper_style else 6.0
    crit_width = 2.0 if paper_style else 2.6
    rand_width = 1.9 if paper_style else 2.3
    title_size = 11 if paper_style else 15
    axis_label_size = 10 if paper_style else 13
    tick_size = 9 if paper_style else 11

    ax.plot(
        crit_x,
        crit_y,
        marker="o",
        markersize=crit_marker_size,
        linewidth=crit_width,
        color="#C44E52",
        label="Critical heads",
    )
    ax.plot(
        rand_x,
        rand_y,
        marker="s",
        markersize=rand_marker_size,
        linewidth=rand_width,
        color="#4C72B0",
        label="Random heads",
    )
    ax.axvline(0, linestyle="--", linewidth=1.0, alpha=0.45, color="#777777")
    ax.set_title(panel_title, fontsize=title_size)
    ax.set_xlabel("Masked heads", fontsize=axis_label_size)
    ax.set_ylabel(ylabel, fontsize=axis_label_size)
    ax.set_xticks(all_x)
    ax.tick_params(axis="both", labelsize=tick_size)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)


def main() -> None:
    args = parse_args()
    plt.style.use("seaborn-v0_8-whitegrid")

    text_critical = with_baseline(load_curve(args.text_critical_csv), args.text_baseline)
    text_random = with_baseline(load_curve(args.text_random_csv), args.text_baseline)
    img_critical = with_baseline(load_curve(args.img_critical_csv), args.img_baseline)
    img_random = with_baseline(load_curve(args.img_random_csv), args.img_baseline)

    out_stem = args.output_stem
    out_stem.parent.mkdir(parents=True, exist_ok=True)

    figsize = (7.1, 2.9) if args.paper_style else (10.8, 4.2)
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=False)
    plot_panel(
        axes[0],
        text_critical,
        text_random,
        ylabel="Macro F1",
        panel_title="(a) Text side",
        paper_style=args.paper_style,
    )
    plot_panel(
        axes[1],
        img_critical,
        img_random,
        ylabel="Macro F1",
        panel_title="(b) Image side",
        paper_style=args.paper_style,
    )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.01 if args.paper_style else 1.02),
        frameon=False,
        fontsize=9 if args.paper_style else 12,
    )
    if args.title:
        fig.suptitle(args.title, fontsize=12 if args.paper_style else 16, y=1.06 if args.paper_style else 1.08)
    top_rect = 0.87 if args.paper_style and not args.title else 0.82 if args.paper_style else 0.94
    fig.tight_layout(rect=(0, 0, 1, top_rect))
    fig.savefig(str(out_stem) + ".png", dpi=320 if args.paper_style else 260, bbox_inches="tight")
    fig.savefig(str(out_stem) + ".pdf", bbox_inches="tight")
    plt.close(fig)

    write_clean_csv(
        Path(str(out_stem) + "_clean.csv"),
        text_critical=text_critical,
        text_random=text_random,
        img_critical=img_critical,
        img_random=img_random,
    )

    print(f"[MaskPlot] Saved figure to {out_stem}.png and {out_stem}.pdf")
    print(f"[MaskPlot] Saved cleaned curve data to {out_stem}_clean.csv")


if __name__ == "__main__":
    main()
