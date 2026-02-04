"""
plot_results.py — Paper-ready figures
Generates 6 publication-quality plots:
  1. Learning curves (reward over episodes)
  2. Final benchmark comparison (bar chart — 5 metrics)
  3. CPU & Memory utilisation over training
  4. SLA violation rate over training
  5. Energy consumption comparison
  6. Ablation study
"""

import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

HERE        = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
PLOTS_DIR   = os.path.join(HERE, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── Colours & labels ──────────────────────────────────────────────────────
COLORS = {
    "IQL+Fuzzy+LLM":   "#2E86AB",
    "IQL+Fuzzy":       "#A23B72",
    "IQL_Base":        "#F18F01",
    "Round-Robin":     "#C73E1D",
    "First-Fit":       "#3B1F2B",
    "Single-DQN":      "#44BBA4",
    "Single-PPO":      "#E94F37",
    "H-MARL":          "#6A0572",
    "Federated-MARL":  "#1B998B",
}

LABELS = {
    "IQL+Fuzzy+LLM":   "IQL + Fuzzy + LLM (Ours)",
    "IQL+Fuzzy":       "IQL + Fuzzy (Ablation)",
    "IQL_Base":        "IQL Baseline (Ablation)",
    "Round-Robin":     "Round Robin",
    "First-Fit":       "First Fit",
    "Single-DQN":      "Single-Agent DQN",
    "Single-PPO":      "Single-Agent PPO",
    "H-MARL":          "H-MARL (Hierarchical)",
    "Federated-MARL":  "Federated MARL",
}

# Drawing order — IQL+Fuzzy+LLM drawn last so it sits on top
DRAW_ORDER = [
    "Round-Robin", "First-Fit",
    "IQL_Base", "IQL+Fuzzy",
    "Single-DQN", "Single-PPO",
    "H-MARL", "Federated-MARL",
    "IQL+Fuzzy+LLM",
]

ABLATION_KEYS = {"IQL+Fuzzy", "IQL_Base"}


def smooth(arr, w=10):
    return uniform_filter1d(arr, size=w)


def load():
    path = os.path.join(RESULTS_DIR, "all_results.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\n  Missing: {path}\n  Run python prepare_results.py first.\n")
    with open(path) as f:
        return json.load(f)


def eval_mean(data, key):
    return float(np.mean([e[key] for e in data["eval"]]))


# ─────────────────────────────────────────────────────────────────────────
# Fig 1 — Learning Curves
# ─────────────────────────────────────────────────────────────────────────
def plot_learning_curves(results):
    fig, ax = plt.subplots(figsize=(12, 5))

    ordered = [k for k in DRAW_ORDER if k in results]

    for method in ordered:
        data    = results[method]
        rewards = [e["total_reward"] for e in data["train"]]
        s       = smooth(rewards, 12)
        is_abl  = method in ABLATION_KEYS
        is_ours = method == "IQL+Fuzzy+LLM"
        ax.plot(s,
                color=COLORS.get(method, "grey"),
                label=LABELS.get(method, method),
                linewidth=2.8 if is_ours else (1.6 if is_abl else 1.5),
                linestyle="--" if is_abl else "-",
                alpha=1.0 if is_ours else 0.80,
                zorder=10 if is_ours else 1)

    ax.set_xlabel("Training Episode", fontsize=13)
    ax.set_ylabel("Total Reward",     fontsize=13)
    ax.set_title("Learning Curves — MARL Cloud Benchmark\n(Google Cluster Trace)", fontsize=14)
    ax.legend(fontsize=8, loc="lower right", ncol=2, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "fig1_learning_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────
# Fig 2 — Benchmark Bar Chart (5 metrics, no ablation variants)
# ─────────────────────────────────────────────────────────────────────────
def plot_benchmark_bars(results):
    methods = [k for k in DRAW_ORDER if k in results and k not in ABLATION_KEYS]

    metrics      = ["avg_cpu_util", "avg_mem_util", "sla_violation_rate",
                    "avg_wait_time", "total_energy"]
    titles       = ["CPU Utilisation", "Memory Utilisation",
                    "SLA Violation Rate", "Avg Wait Time (s)", "Total Energy"]
    scales       = [100, 100, 100, 1, 1]
    units        = ["%", "%", "%", "s", "J"]
    lower_better = [False, False, True, True, True]

    fig, axes = plt.subplots(1, 5, figsize=(20, 5))

    for ax, metric, title, scale, unit, lb in zip(
            axes, metrics, titles, scales, units, lower_better):
        vals   = [eval_mean(results[m], metric) * scale for m in methods]
        colors = [COLORS.get(m, "grey") for m in methods]

        bars = ax.bar(range(len(methods)), vals, color=colors,
                      edgecolor="white", linewidth=0.5)

        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(
            [LABELS.get(m, m)
             .replace(" (Ours)", "")
             .replace(" (Hierarchical)", "")
             for m in methods],
            rotation=35, ha="right", fontsize=7.5
        )
        ax.set_title(f"{title}\n({'↓ better' if lb else '↑ better'})", fontsize=10)
        ax.set_ylabel(unit)
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

        # Gold border on the best bar
        best_idx = int(np.argmin(vals) if lb else np.argmax(vals))
        bars[best_idx].set_edgecolor("gold")
        bars[best_idx].set_linewidth(3)

        # Y-axis: start from 0, give a little headroom
        ax.set_ylim(0, max(vals) * 1.15)

    fig.suptitle(
        "Benchmark Comparison — All Methods vs Baselines\n(Evaluation on Google Cluster Trace)",
        fontsize=13, y=1.02)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "fig2_benchmark_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────
# Fig 3 — CPU & Memory Utilisation over training
# ─────────────────────────────────────────────────────────────────────────
def plot_utilisation(results):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ordered = [k for k in DRAW_ORDER if k in results]

    for method in ordered:
        data    = results[method]
        cpu     = smooth([e["avg_cpu_util"] * 100 for e in data["train"]], 10)
        mem     = smooth([e["avg_mem_util"] * 100 for e in data["train"]], 10)
        is_ours = method == "IQL+Fuzzy+LLM"
        is_abl  = method in ABLATION_KEYS
        lw = 2.5 if is_ours else 1.5
        ls = "--" if is_abl else "-"
        al = 1.0 if is_ours else 0.75
        zo = 10  if is_ours else 1

        ax1.plot(cpu, color=COLORS.get(method, "grey"),
                 label=LABELS.get(method, method),
                 linewidth=lw, linestyle=ls, alpha=al, zorder=zo)
        ax2.plot(mem, color=COLORS.get(method, "grey"),
                 linewidth=lw, linestyle=ls, alpha=al, zorder=zo)

    for ax, title in [(ax1, "CPU Utilisation (%)"), (ax2, "Memory Utilisation (%)")]:
        ax.set_xlabel("Episode", fontsize=11)
        ax.set_ylabel(title,     fontsize=11)
        ax.set_title(title,      fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    ax1.legend(fontsize=7.5, ncol=2, framealpha=0.9)
    plt.suptitle("Resource Utilisation over Training", fontsize=13)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "fig3_utilisation.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────
# Fig 4 — SLA Violation Rate over training
# ─────────────────────────────────────────────────────────────────────────
def plot_sla(results):
    fig, ax = plt.subplots(figsize=(12, 5))

    ordered = [k for k in DRAW_ORDER if k in results]

    for method in ordered:
        data    = results[method]
        sla     = smooth([e["sla_violation_rate"] * 100 for e in data["train"]], 10)
        is_ours = method == "IQL+Fuzzy+LLM"
        is_abl  = method in ABLATION_KEYS
        lw = 2.5 if is_ours else 1.5
        ls = "--" if is_abl else "-"
        al = 1.0 if is_ours else 0.78
        zo = 10  if is_ours else 1
        ax.plot(sla, color=COLORS.get(method, "grey"),
                label=LABELS.get(method, method),
                linewidth=lw, linestyle=ls, alpha=al, zorder=zo)

    ax.axhline(5, color="red", linestyle=":", linewidth=1.8,
               label="5% SLA Target", zorder=5)
    ax.set_xlabel("Episode",                fontsize=12)
    ax.set_ylabel("SLA Violation Rate (%)", fontsize=12)
    ax.set_title("SLA Violation Rate over Training\n(lower is better)", fontsize=13)
    ax.legend(fontsize=8, ncol=2, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "fig4_sla_violations.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────
# Fig 5 — Energy Consumption (no ablation)
# ─────────────────────────────────────────────────────────────────────────
def plot_energy(results):
    methods = [k for k in DRAW_ORDER if k in results and k not in ABLATION_KEYS]
    # Reverse so IQL+Fuzzy+LLM appears at top of horizontal bar chart
    methods = list(reversed(methods))

    energy  = [eval_mean(results[m], "total_energy") for m in methods]
    colors  = [COLORS.get(m, "grey") for m in methods]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(range(len(methods)), energy, color=colors, edgecolor="white")
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels([LABELS.get(m, m) for m in methods], fontsize=10)
    ax.set_xlabel("Total Energy Consumed (J)", fontsize=12)
    ax.set_title("Energy Consumption Comparison (↓ better)", fontsize=13)
    ax.grid(axis="x", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    for i, v in enumerate(energy):
        ax.text(v + max(energy) * 0.008, i, f"{v:.0f}",
                va="center", fontsize=9)

    # Gold border on best (lowest)
    best_idx = int(np.argmin(energy))
    bars[best_idx].set_edgecolor("gold")
    bars[best_idx].set_linewidth(3)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "fig5_energy.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────
# Fig 6 — Ablation Study
# ─────────────────────────────────────────────────────────────────────────
def plot_ablation(results):
    abl_methods = ["IQL_Base", "IQL+Fuzzy", "IQL+Fuzzy+LLM"]
    abl_labels  = ["IQL\n(Base)", "IQL +\nFuzzy", "IQL + Fuzzy\n+ LLM (Full)"]
    bar_colors  = ["#F18F01", "#A23B72", "#2E86AB"]

    metrics     = ["total_reward", "avg_cpu_util", "sla_violation_rate", "total_energy"]
    titles      = ["Total Reward ↑", "CPU Util % ↑", "SLA Violation % ↓", "Energy (J) ↓"]
    scales      = [1, 100, 100, 1]
    lower_b     = [False, False, True, True]

    fig, axes = plt.subplots(1, 4, figsize=(14, 5))
    x = np.arange(len(abl_methods))

    for ax, metric, title, scale, lb in zip(axes, metrics, titles, scales, lower_b):
        vals = [eval_mean(results[m], metric) * scale
                if m in results else 0.0
                for m in abl_methods]

        all_negative = all(v <= 0 for v in vals)

        if all_negative:
            # For negative-valued metrics (reward), use absolute values and flip axis label
            abs_vals  = [abs(v) for v in vals]
            bars = ax.bar(x, abs_vals, color=bar_colors, edgecolor="white", width=0.55)
            ax.set_title(title + "\n(|value|, smaller = better)", fontsize=10)
            ax.set_ylim(0, max(abs_vals) * 1.2)
            best = int(np.argmax(abs_vals) if lb else np.argmin(abs_vals))
            for bar, v, av in zip(bars, vals, abs_vals):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + max(abs_vals)*0.02,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=8)
        else:
            bars = ax.bar(x, vals, color=bar_colors, edgecolor="white", width=0.55)
            ax.set_title(title, fontsize=11)
            ax.set_ylim(0, max(vals) * 1.2)
            best = int(np.argmin(vals) if lb else np.argmax(vals))
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + max(vals)*0.02,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(abl_labels, fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

        # Gold border on best
        bars[best].set_edgecolor("gold")
        bars[best].set_linewidth(3)

    fig.suptitle("Ablation Study: Contribution of Each Component", fontsize=13)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "fig6_ablation.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading results...")
    results = load()
    print(f"  Methods: {list(results.keys())}\n")
    print("Generating figures...\n")
    plot_learning_curves(results)
    plot_benchmark_bars(results)
    plot_utilisation(results)
    plot_sla(results)
    plot_energy(results)
    plot_ablation(results)
    print(f"\n✓ All 6 figures saved to: {PLOTS_DIR}/")