#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
COLORS = {"ToolACE": "#2a6fbb", "ReTool": "#d95f02", "Glaive": "#2b8c4b"}


def style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)


def box(ax, x, y, w, h, title, body, color="#17324d"):
    patch = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.015,rounding_size=0.018",
        facecolor="white", edgecolor=color, linewidth=1.5,
    )
    ax.add_patch(patch)
    ax.text(x + 0.02, y + h - 0.055, title, weight="bold", color=color, va="top")
    ax.text(x + 0.02, y + h - 0.12, body, va="top", linespacing=1.35, fontsize=8)


def figure_schematic() -> None:
    fig, ax = plt.subplots(figsize=(10.5, 4.1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    box(ax, 0.015, 0.54, 0.21, 0.36, "1  Stored trajectory",
        "USER  find order A\nASSISTANT  call lookup\nTOOL  confirmation = X7Q\nASSISTANT  VALUE=X7Q")
    box(ax, 0.265, 0.54, 0.21, 0.36, "2  Render + mask",
        "system / user / tool:  · · ·\nassistant tool call:       █ █\nassistant answer:          █ █\n\n█ = supervised token")
    box(ax, 0.515, 0.54, 0.21, 0.36, "3  Length handling",
        "A split can place the target\nand its tool result in different\nattention groups.", color="#a33b20")
    box(ax, 0.765, 0.54, 0.22, 0.36, "4  Post-render audit",
        "Target still labeled:       YES\nVerified source reachable:  NO\n\n→ unsupported supervision", color="#a33b20")
    for left, right in [(0.225, 0.265), (0.475, 0.515), (0.725, 0.765)]:
        ax.add_patch(FancyArrowPatch((left, 0.72), (right, 0.72), arrowstyle="-|>", mutation_scale=14, color="#52606d"))

    ax.text(0.02, 0.38, "Boundary-breaking view", weight="bold", color="#a33b20")
    segments = [
        (0.02, 0.18, "user + call", "#d9eaf7"),
        (0.205, 0.19, "tool: X7Q", "#cfe8df"),
        (0.415, 0.08, "split", "#f5d3ca"),
        (0.515, 0.21, "target: VALUE=X7Q", "#eadcf0"),
    ]
    for x, w, label, color in segments:
        ax.add_patch(FancyBboxPatch((x, 0.25), w, 0.09, boxstyle="round,pad=0.01", facecolor=color, edgecolor="#52606d"))
        ax.text(x + w / 2, 0.295, label, ha="center", va="center")
    ax.text(0.02, 0.10, "Grouped closure", weight="bold", color="#087f5b")
    ax.add_patch(FancyBboxPatch((0.19, 0.055), 0.57, 0.10, boxstyle="round,pad=0.012", facecolor="#e1f3eb", edgecolor="#087f5b", linewidth=1.5))
    ax.text(0.475, 0.105, "minimal structure  +  tool: X7Q  +  complete target: VALUE=X7Q", ha="center", va="center", fontsize=8)
    ax.add_patch(FancyArrowPatch((0.78, 0.105), (0.94, 0.105), arrowstyle="-|>", mutation_scale=14, color="#087f5b"))
    ax.text(0.98, 0.105, "valid\nexample", ha="right", va="center", color="#087f5b", weight="bold", fontsize=8)
    save(fig, "figure1_schematic")


def combined_incidence() -> pd.DataFrame:
    frames = [
        pd.read_csv(ROOT / "analysis/corpus_audit/toolace_retool/incidence.csv"),
        pd.read_csv(ROOT / "analysis/corpus_audit/glaive/incidence.csv"),
    ]
    frame = pd.concat(frames, ignore_index=True)
    frame = frame[frame.stratum == "all_admitted"].copy()
    frame["Corpus"] = frame.corpus.map({"toolace": "ToolACE", "retool": "ReTool", "glaive": "Glaive"})
    return frame


def figure_incidence() -> None:
    data = combined_incidence()
    policies = {"bfd_split": ("BFD-split", "-"), "wrapped_single_batch": ("Wrapped (intended)", "--")}
    markers = {"bfd": "s", "bfd_split": "o", "wrapped_single_batch": "D"}
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.65))
    ax = axes[0]
    for policy, (_, linestyle) in policies.items():
        subset = data[data.policy == policy]
        for corpus in ["ToolACE", "ReTool", "Glaive"]:
            group = subset[subset.Corpus == corpus].sort_values("max_length")
            ax.plot(
                group.max_length,
                100 * group.conditional_prevalence,
                marker="o",
                linestyle=linestyle,
                lw=1.8,
                ms=3.8,
                color=COLORS[corpus],
            )
    ax.axhline(0, color="#4d5965", lw=1.0)
    ax.text(280, 1.5, "Default BFD: 0% in all cells", color="#4d5965", fontsize=7.5)
    ax.set_xscale("log", base=2)
    ax.set_xticks([256, 512, 1024, 2048, 4096, 8192], ["256", "512", "1k", "2k", "4k", "8k"])
    ax.set_xlabel("Token budget")
    ax.set_ylabel("Unsupported / trained verified targets (%)")
    ax.set_title("A  Boundary-breaking policies across budgets", loc="left", weight="bold")
    ax.grid(axis="y", alpha=0.25)
    legend = [Line2D([0], [0], color=COLORS[name], lw=2, label=name) for name in ["ToolACE", "ReTool", "Glaive"]]
    legend += [Line2D([0], [0], color="#4d5965", lw=2, linestyle=style, label=name) for name, style in policies.values()]
    ax.legend(handles=legend, frameon=False, ncol=2, fontsize=7, loc="upper right")

    ax = axes[1]
    subset = data[(data.max_length == 512) & data.policy.isin(["bfd", *policies])]
    for row in subset.itertuples():
        ax.scatter(
            100 * row.supervised_token_retention,
            100 * row.conditional_prevalence,
            marker=markers[row.policy],
            s=65,
            color=COLORS[row.Corpus],
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
    ax.set_xlabel("Nominal assistant supervision retained (%)")
    ax.set_ylabel("Unsupported / trained targets (%)")
    ax.set_title("B  Integrity--retention tradeoff at 512 tokens", loc="left", weight="bold")
    ax.grid(alpha=0.25)
    policy_legend = [
        Line2D([0], [0], marker=markers[p], linestyle="none", color="#4d5965", markersize=6, label=n)
        for p, n in [("bfd", "BFD"), ("bfd_split", "BFD-split"), ("wrapped_single_batch", "Wrapped")]
    ]
    ax.legend(handles=policy_legend, frameon=False, fontsize=7, loc="upper left")
    fig.suptitle("Sequence preservation and supervision retention are separate objectives", y=1.03, weight="bold")
    fig.tight_layout()
    save(fig, "figure2_current_policy_incidence")


def figure_denominators() -> None:
    data = combined_incidence()
    data = data[(data.policy == "bfd_split") & (data.max_length == 512)].set_index("Corpus")
    metrics = [
        ("conditional_prevalence", "Among trained verified targets"),
        ("conversation_incidence", "Among all conversations"),
        ("nominal_objective_exposure", "Among nominal assistant tokens"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(9.7, 3.2))
    corpora = ["ToolACE", "ReTool", "Glaive"]
    for ax, (metric, title) in zip(axes, metrics, strict=True):
        values = [100 * data.loc[corpus, metric] for corpus in corpora]
        bars = ax.bar(corpora, values, color=[COLORS[c] for c in corpora], width=0.65)
        ax.set_title(title)
        ax.set_ylabel("Percent")
        ax.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, values, strict=True):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{value:.2f}%", ha="center", va="bottom", fontsize=8)
    fig.suptitle("The same 512-token failure has three different denominators", y=1.03, weight="bold")
    save(fig, "figure3_denominators")


def figure_grouped_repair() -> None:
    data = pd.read_csv(ROOT / "outputs/materialization/summary.csv")
    data = data[
        data.max_length.isin([256, 512, 1024, 2048])
        & data.policy.isin(["source_target_round", "single_target_closure", "grouped_closure"])
    ]
    labels = {"source_target_round": "Source + target round", "single_target_closure": "Single-target closure", "grouped_closure": "Grouped closure"}
    markers = {"source_target_round": "s", "single_target_closure": "o", "grouped_closure": "D"}
    policy_colors = {"source_target_round": "#7b8794", "single_target_closure": "#8b5a9b", "grouped_closure": "#2a6fbb"}
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.65), sharex=False, sharey=True)
    for ax, corpus in zip(axes, ["toolace", "retool"], strict=True):
        subset = data[data.corpus == corpus]
        for policy in labels:
            group = subset[subset.policy == policy].sort_values("max_length")
            ax.plot(
                100 * group.input_token_ratio,
                100 * group.target_retention,
                marker=markers[policy],
                lw=1.8,
                ms=5,
                color=policy_colors[policy],
                label=labels[policy],
            )
            for row in group.itertuples():
                budget = {256: "256", 512: "512", 1024: "1k", 2048: "2k"}[row.max_length]
                offsets = {
                    "source_target_round": (4, 4),
                    "single_target_closure": (4, -11),
                    "grouped_closure": (7, 5),
                }
                ax.annotate(
                    budget,
                    (100 * row.input_token_ratio, 100 * row.target_retention),
                    xytext=offsets[policy],
                    textcoords="offset points",
                    fontsize=6.5,
                )
        ax.set_title({"toolace": "ToolACE", "retool": "ReTool"}[corpus])
        ax.set_xlabel("Retained input-token ratio (%)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Verified-target retention (%)")
    axes[1].legend(frameon=False, fontsize=7, loc="lower right")
    fig.suptitle("Support-preserving materializers trace distinct Pareto trajectories", y=1.03, weight="bold")
    fig.tight_layout()
    save(fig, "figure4_grouped_repair")


def figure_schema_repair() -> None:
    data = pd.read_csv(ROOT / "outputs/schema_repair/summary.csv")
    modes = ["whole_schema", "invoked_tool", "parameter_slice"]
    names = {"whole_schema": "Whole schema", "invoked_tool": "Invoked tool", "parameter_slice": "Parameter slice"}
    colors = {"whole_schema": "#8c96a0", "invoked_tool": "#2a6fbb", "parameter_slice": "#2b8c4b"}
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.55), sharey=True)
    for ax, corpus in zip(axes, ["toolace", "retool"], strict=True):
        subset = data[data.corpus == corpus]
        budgets = sorted(subset.max_length.unique())
        x = np.arange(len(budgets))
        width = 0.24
        for offset, mode in zip([-width, 0, width], modes, strict=True):
            group = subset[subset["mode"] == mode].set_index("max_length")
            ax.bar(x + offset, [100 * group.loc[b, "repair_rate"] for b in budgets], width=width, color=colors[mode], label=names[mode])
        ax.set_xticks(x, [str(b) if b < 1000 else f"{b//1024}k" for b in budgets])
        ax.set_xlabel("Token budget")
        ax.set_title({"toolace": "ToolACE", "retool": "ReTool"}[corpus])
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Action targets repaired (%)")
    axes[1].legend(frameon=False, loc="upper left")
    fig.suptitle("Schema slicing repairs short-budget ToolACE actions; compact ReTool schemas are a boundary", y=1.03, weight="bold")
    save(fig, "figure5_schema_repair")


def figure_model_study() -> bool:
    path = ROOT / "analysis/model_study/per_seed.csv"
    if not path.exists():
        return False
    data = pd.read_csv(path)
    x_values = [0, 25, 50, 100]
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.25))
    test = data[data.split == "test"]
    conflict = data[data.split == "conflict"]
    for seed, group in test.groupby("seed"):
        ordered = group.set_index("corruption").reindex(x_values)
        axes[0].plot(x_values, 100 * ordered.exact, color="#2a6fbb", alpha=0.28, lw=1)
        axes[1].plot(x_values, ordered.value_token_nll, color="#d95f02", alpha=0.28, lw=1)
    test_mean = test.groupby("corruption").mean(numeric_only=True).reindex(x_values)
    axes[0].plot(x_values, 100 * test_mean.exact, marker="o", color="#2a6fbb", lw=2.2)
    axes[1].plot(x_values, test_mean.value_token_nll, marker="o", color="#d95f02", lw=2.2)
    conflict_mean = conflict.groupby("corruption").mean(numeric_only=True).reindex(x_values)
    axes[2].plot(x_values, 100 * conflict_mean.exact, marker="o", color="#2b8c4b", lw=2.2, label="Correct fresh value")
    axes[2].plot(x_values, 100 * conflict_mean.copied_stale, marker="s", color="#a33b20", lw=2.2, label="Copied stale value")
    axes[0].set_ylabel("Exact generation (%)")
    axes[1].set_ylabel("Value-token NLL")
    axes[2].set_ylabel("Conflict outcome (%)")
    axes[0].set_ylim(-2, 102)
    axes[2].set_ylim(-2, 102)
    for ax, title in zip(axes, ["Held-out exact", "Held-out likelihood", "Stale-value conflict"], strict=True):
        ax.set_xticks(x_values)
        ax.set_xlabel("Unsupported training examples (%)")
        ax.set_title(title)
        ax.grid(alpha=0.25)
    axes[2].legend(frameon=False, fontsize=7)
    fig.suptitle("Controlled support-corruption dose response (three optimization seeds)", y=1.03, weight="bold")
    save(fig, "figure6_model_study")
    return True


def main() -> None:
    style()
    figure_schematic()
    figure_incidence()
    figure_denominators()
    figure_grouped_repair()
    figure_schema_repair()
    figure_model_study()
    manifest = {path.name: path.stat().st_size for path in sorted(OUT.glob("figure*"))}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
