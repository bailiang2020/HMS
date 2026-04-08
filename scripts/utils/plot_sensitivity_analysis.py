#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


TAU_DATA = {
    0.1: {"multi": 0.792543333, "img": 0.78929, "text": 0.649223333},
    0.2: {"multi": 0.78911, "img": 0.787983333, "text": 0.65136},
}

GAMMA_DATA = {
    0.1: {"multi": 0.78915, "img": 0.783793333, "text": 0.63929},
    0.2: {"multi": 0.790653333, "img": 0.786616667, "text": 0.6508},
    0.3: {"multi": 0.787023333, "img": 0.784393333, "text": 0.650936667},
    0.4: {"multi": 0.789746667, "img": 0.783636667, "text": 0.6572},
    0.5: {"multi": 0.79003, "img": 0.787743333, "text": 0.646706667},
    0.6: {"multi": 0.790586667, "img": 0.786186667, "text": 0.64566},
    0.7: {"multi": 0.79266, "img": 0.788056667, "text": 0.653633333},
    0.8: {"multi": 0.793086667, "img": 0.785416667, "text": 0.644923333},
    0.9: {"multi": 0.791756667, "img": 0.78831, "text": 0.6377},
}


def main() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    out_dir = Path("results/analysis_plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_stem = out_dir / "sensitivity_analysis_dgm4"

    colors = {"multi": "#4C72B0", "img": "#C44E52", "text": "#55A868"}
    labels = {"multi": "Multimodal", "img": "Image only", "text": "Text only"}

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.65))

    tau_vals = list(TAU_DATA.keys())
    x = np.arange(len(tau_vals))
    width = 0.22
    for offset, key in zip((-width, 0, width), ("multi", "img", "text")):
        axes[0].bar(
            x + offset,
            [TAU_DATA[t][key] for t in tau_vals],
            width=width,
            color=colors[key],
            label=labels[key],
        )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"{t:.1f}" for t in tau_vals])
    axes[0].set_xlabel(r"HMS threshold $\tau$", fontsize=10)
    axes[0].set_ylabel("Macro F1", fontsize=10)
    axes[0].set_title(r"(a) Effect of $\tau$", fontsize=11)
    axes[0].tick_params(axis="both", labelsize=9)
    axes[0].set_ylim(0.63, 0.81)

    gamma_vals = list(GAMMA_DATA.keys())
    for key in ("multi", "img", "text"):
        axes[1].plot(
            gamma_vals,
            [GAMMA_DATA[g][key] for g in gamma_vals],
            marker="o",
            markersize=3.8,
            linewidth=1.8,
            color=colors[key],
            label=labels[key],
        )
    axes[1].set_xlabel(r"UKR factor $\gamma$", fontsize=10)
    axes[1].set_ylabel("Macro F1", fontsize=10)
    axes[1].set_title(r"(b) Effect of $\gamma$", fontsize=11)
    axes[1].tick_params(axis="both", labelsize=9)
    axes[1].set_xticks(gamma_vals)
    axes[1].set_ylim(0.635, 0.795)

    handles, legend_labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=3,
        frameon=False,
        fontsize=9,
    )
    for ax in axes:
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)

    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(str(out_stem) + ".png", dpi=320, bbox_inches="tight")
    fig.savefig(str(out_stem) + ".pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"Saved sensitivity figure to {out_stem}.png and {out_stem}.pdf")


if __name__ == "__main__":
    main()
