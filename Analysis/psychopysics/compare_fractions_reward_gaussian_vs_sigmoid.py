#!/usr/bin/env python3
"""
Compare Gaussian vs Sigmoid overlap with reward-responsive neurons
within the psychophysics dataset only.

Expected keys in each comparison_fractions_*.pkl:
- Gaussian_Hit(reward)_All
- Sigmoid_Hit(reward)_All

Each per-mouse entry is expected to contain:
- set1_only
- set2_only
- both

Interpretation:
For Gaussian_Hit(reward)_All:
- set1_only = Gaussian only
- set2_only = Reward only (not Gaussian)
- both = Gaussian ∩ Reward

For Sigmoid_Hit(reward)_All:
- set1_only = Sigmoid only
- set2_only = Reward only (not Sigmoid)
- both = Sigmoid ∩ Reward
"""

import os
import pickle
from os.path import join

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib_venn import venn2
from scipy import stats

plt.rcParams["font.family"] = "Arial"
plt.rcParams["pdf.fonttype"] = 42


# ---------------------------------------------------------------------------
# Data locations (set these before running)
# ---------------------------------------------------------------------------
DATA_ROOT    = r'path\to\2p_data'  # folder with the imaging sessions (<mouse>/<session>/ML/...)
RESULTS_ROOT = r'path\to\results'  # folder where the across-mice results are saved


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------

def mean_sem(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    mean_val = np.mean(values)
    sem_val = np.std(values, ddof=1) / np.sqrt(n) if n > 1 else 0.0
    return float(mean_val), float(sem_val)


def _shapiro_safe(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return None, None
    if np.allclose(x, x[0]):
        return None, None
    try:
        return stats.shapiro(x)
    except Exception:
        return None, None


def choose_paired_test(x, y):
    """
    Auto choose paired t-test or Wilcoxon based on Shapiro on paired differences.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    d = x - y
    n = len(d)

    result = {
        "n": n,
        "test": None,
        "stat_name": None,
        "stat_value": np.nan,
        "p_value": np.nan,
        "shapiro_W": np.nan,
        "shapiro_p": np.nan,
        "mean_x": np.mean(x) if n > 0 else np.nan,
        "mean_y": np.mean(y) if n > 0 else np.nan,
    }

    if n < 2:
        return result

    W, p_norm = _shapiro_safe(d)
    if W is not None:
        result["shapiro_W"] = float(W)
        result["shapiro_p"] = float(p_norm)

    normal = (p_norm is not None) and (p_norm >= 0.05) and (np.std(d, ddof=1) > 0)

    if normal:
        t_stat, p_val = stats.ttest_rel(x, y, nan_policy="omit")
        result["test"] = "paired t-test"
        result["stat_name"] = "t"
        result["stat_value"] = float(t_stat)
        result["p_value"] = float(p_val)
    else:
        try:
            w_stat, p_val = stats.wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
        except ValueError:
            w_stat, p_val = 0.0, 1.0
        result["test"] = "Wilcoxon signed-rank"
        result["stat_name"] = "W"
        result["stat_value"] = float(w_stat)
        result["p_value"] = float(p_val)

    return result


def p_to_stars(p):
    if not np.isfinite(p):
        return "n.a."
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def add_sig_bracket(ax, x1, x2, y, p, text_y_pad=0.02):
    yr = ax.get_ylim()[1] - ax.get_ylim()[0]
    ypad = yr * text_y_pad
    ax.plot([x1, x1, x2, x2], [y, y + ypad, y + ypad, y], color='black', linewidth=1)
    ax.text((x1 + x2) / 2, y + ypad * 1.1, f"{p_to_stars(p)}  p={p:.3g}",
            ha='center', va='bottom', fontsize=10)


# ---------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------

def load_comparison_data(pkl_files):
    """
    Loads all 2-set fraction data from psychophysics pickle files.
    Returns:
      data[comparison_name] = list of per-mouse dicts
    """
    data_out = {}

    print("Loading psychophysics data...")
    for pkl_file in pkl_files:
        print(f"  Loading: {pkl_file}")
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
            if "2set" in data:
                for comp_name, fraction_dict in data["2set"].items():
                    data_out.setdefault(comp_name, []).append(fraction_dict)
        except FileNotFoundError:
            print(f"Warning: file not found: {pkl_file}")
        except Exception as e:
            print(f"Error loading {pkl_file}: {e}")

    return data_out


def extract_overlap_arrays(data_dict, key_name):
    """
    Extract set1_only, set2_only, both arrays from one comparison key.
    """
    if key_name not in data_dict:
        print(f"Key not found: {key_name}")
        print(f"Available keys: {list(data_dict.keys())}")
        return None, None, None

    entries = data_dict[key_name]
    set1_only = [d.get("set1_only", 0) for d in entries]
    set2_only = [d.get("set2_only", 0) for d in entries]
    both = [d.get("both", 0) for d in entries]

    return set1_only, set2_only, both


# ---------------------------------------------------------------------
# venn plots
# ---------------------------------------------------------------------

def plot_gaussian_sigmoid_venns(data_dict, save_dir="."):
    """
    Side-by-side Venns:
    - Gaussian vs Reward
    - Sigmoid vs Reward
    """
    g_only, g_reward_only, g_both = extract_overlap_arrays(data_dict, "Gaussian_Hit(reward)_All")
    s_only, s_reward_only, s_both = extract_overlap_arrays(data_dict, "Sigmoid_Hit(reward)_All")

    if g_only is None or s_only is None:
        print("Missing Gaussian or Sigmoid data. Cannot plot venns.")
        return

    g_only_mean, g_only_sem = mean_sem(g_only)
    g_reward_only_mean, g_reward_only_sem = mean_sem(g_reward_only)
    g_both_mean, g_both_sem = mean_sem(g_both)

    s_only_mean, s_only_sem = mean_sem(s_only)
    s_reward_only_mean, s_reward_only_sem = mean_sem(s_reward_only)
    s_both_mean, s_both_sem = mean_sem(s_both)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # Gaussian venn
    venn_g = venn2(
        subsets=(g_reward_only_mean, g_only_mean, g_both_mean),
        set_labels=("Reward", "Gaussian"),
        ax=ax1
    )

    if venn_g.get_patch_by_id("10"):
        venn_g.get_patch_by_id("10").set_color("lightcoral")  # Reward only
        venn_g.get_patch_by_id("10").set_alpha(0.7)
    if venn_g.get_patch_by_id("01"):
        venn_g.get_patch_by_id("01").set_color("lightblue")  # Gaussian only
        venn_g.get_patch_by_id("01").set_alpha(0.7)
    if venn_g.get_patch_by_id("11"):
        venn_g.get_patch_by_id("11").set_color("mediumpurple")  # Both
        venn_g.get_patch_by_id("11").set_alpha(0.7)

    if venn_g.get_label_by_id("10"):
        venn_g.get_label_by_id("10").set_text(f"{g_reward_only_mean:.1f}% ± {g_reward_only_sem:.1f}%")
        venn_g.get_label_by_id("10").set_fontsize(12)
        venn_g.get_label_by_id("10").set_fontweight("bold")
    if venn_g.get_label_by_id("01"):
        venn_g.get_label_by_id("01").set_text(f"{g_only_mean:.1f}% ± {g_only_sem:.1f}%")
        venn_g.get_label_by_id("01").set_fontsize(12)
        venn_g.get_label_by_id("01").set_fontweight("bold")
    if venn_g.get_label_by_id("11"):
        venn_g.get_label_by_id("11").set_text(f"{g_both_mean:.1f}% ± {g_both_sem:.1f}%")
        venn_g.get_label_by_id("11").set_fontsize(12)
        venn_g.get_label_by_id("11").set_fontweight("bold")

    ax1.set_title("Reward vs Gaussian", fontsize=14, fontweight="bold")

    # Sigmoid venn
    venn_s = venn2(
        subsets=(s_reward_only_mean, s_only_mean, s_both_mean),
        set_labels=("Reward", "Sigmoid"),
        ax=ax2
    )

    if venn_s.get_patch_by_id("10"):
        venn_s.get_patch_by_id("10").set_color("lightcoral")  # Reward only
        venn_s.get_patch_by_id("10").set_alpha(0.7)
    if venn_s.get_patch_by_id("01"):
        venn_s.get_patch_by_id("01").set_color("lightblue")  # Sigmoid only
        venn_s.get_patch_by_id("01").set_alpha(0.7)
    if venn_s.get_patch_by_id("11"):
        venn_s.get_patch_by_id("11").set_color("mediumpurple")  # Both
        venn_s.get_patch_by_id("11").set_alpha(0.7)

    if venn_s.get_label_by_id("10"):
        venn_s.get_label_by_id("10").set_text(f"{s_reward_only_mean:.1f}% ± {s_reward_only_sem:.1f}%")
        venn_s.get_label_by_id("10").set_fontsize(12)
        venn_s.get_label_by_id("10").set_fontweight("bold")
    if venn_s.get_label_by_id("01"):
        venn_s.get_label_by_id("01").set_text(f"{s_only_mean:.1f}% ± {s_only_sem:.1f}%")
        venn_s.get_label_by_id("01").set_fontsize(12)
        venn_s.get_label_by_id("01").set_fontweight("bold")
    if venn_s.get_label_by_id("11"):
        venn_s.get_label_by_id("11").set_text(f"{s_both_mean:.1f}% ± {s_both_sem:.1f}%")
        venn_s.get_label_by_id("11").set_fontsize(12)
        venn_s.get_label_by_id("11").set_fontweight("bold")

    ax2.set_title("Reward vs Sigmoid", fontsize=14, fontweight="bold")

    fig.suptitle('Psychophysics: Gaussian and Sigmoid Overlap with Reward', fontsize=16, fontweight='bold')

    legend_elements = [
        patches.Patch(color='lightblue', alpha=0.7, label='Fit only'),
        patches.Patch(color='lightcoral', alpha=0.7, label='Reward only'),
        patches.Patch(color='mediumpurple', alpha=0.7, label='Both')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.03))

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    out_path = join(save_dir, "Gaussian_Sigmoid_vs_Reward_Venns.pdf")
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.show()

    print(f"Saved venn figure: {out_path}")


# ---------------------------------------------------------------------
# connected bar comparison: gaussian vs sigmoid
# ---------------------------------------------------------------------

def plot_gaussian_vs_sigmoid_connected_bars(data_dict, save_dir="."):
    """
    Compare Gaussian_Hit(reward)_All vs Sigmoid_Hit(reward)_All directly.

    Categories:
    - Fit only
    - Reward only
    - Both

    Bars:
    - Gaussian comparison
    - Sigmoid comparison

    Connected lines connect the same mouse.
    """
    g_only, g_reward_only, g_both = extract_overlap_arrays(data_dict, "Gaussian_Hit(reward)_All")
    s_only, s_reward_only, s_both = extract_overlap_arrays(data_dict, "Sigmoid_Hit(reward)_All")

    if g_only is None or s_only is None:
        print("Missing Gaussian or Sigmoid data. Cannot plot bars.")
        return

    categories = ["Fit only", "Reward only", "Both"]

    gaussian_arrays = [g_only, g_reward_only, g_both]
    sigmoid_arrays = [s_only, s_reward_only, s_both]

    gaussian_means = [mean_sem(x)[0] for x in gaussian_arrays]
    gaussian_sems = [mean_sem(x)[1] for x in gaussian_arrays]
    sigmoid_means = [mean_sem(x)[0] for x in sigmoid_arrays]
    sigmoid_sems = [mean_sem(x)[1] for x in sigmoid_arrays]

    stats_rows = []
    for cat, g_vals, s_vals in zip(categories, gaussian_arrays, sigmoid_arrays):
        res = choose_paired_test(g_vals, s_vals)
        res["category"] = cat
        stats_rows.append(res)

    fig, ax = plt.subplots(figsize=(12, 8))

    bar_width = 0.35
    x_groups = np.array([0, 1.2, 2.4])

    x_gauss = x_groups - bar_width / 2
    x_sig = x_groups + bar_width / 2

    ax.bar(x_gauss, gaussian_means, bar_width,
           yerr=gaussian_sems, label='Gaussian comparison',
           color='steelblue', edgecolor='black', alpha=0.85, capsize=4)

    ax.bar(x_sig, sigmoid_means, bar_width,
           yerr=sigmoid_sems, label='Sigmoid comparison',
           color='indianred', edgecolor='black', alpha=0.85, capsize=4)

    # connect same mice
    for group_idx, (g_vals, s_vals) in enumerate(zip(gaussian_arrays, sigmoid_arrays)):
        n_mice = min(len(g_vals), len(s_vals))
        for mouse_idx in range(n_mice):
            ax.plot([x_gauss[group_idx], x_sig[group_idx]],
                    [g_vals[mouse_idx], s_vals[mouse_idx]],
                    color='gray', alpha=0.55, linewidth=1, linestyle='--', marker='o', zorder=3)

    ax.set_xticks(x_groups)
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylabel('Percentage of total neurons', fontsize=13)
    ax.set_title('Psychophysics: Gaussian vs Sigmoid Overlap Profile with Reward',
                 fontsize=15, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(loc='upper right')

    top_vals = [
        max(gm + gs, sm + ss)
        for gm, gs, sm, ss in zip(gaussian_means, gaussian_sems, sigmoid_means, sigmoid_sems)
    ]
    ymax = max(top_vals) * 1.25 if max(top_vals) > 0 else 1.0
    ax.set_ylim(0, ymax)

    for i, res in enumerate(stats_rows):
        y = top_vals[i] * 1.03
        add_sig_bracket(ax, x_gauss[i], x_sig[i], y, res["p_value"])

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    out_path = join(save_dir, "Gaussian_vs_Sigmoid_Connected_Bars.pdf")
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.show()
    print(f"Saved bar figure: {out_path}")

    df_stats = pd.DataFrame(stats_rows)
    stats_path = join(save_dir, "Gaussian_vs_Sigmoid_stats_summary.xlsx")
    df_stats.to_excel(stats_path, index=False)
    print(f"Saved stats: {stats_path}")

    print("\n=== Gaussian vs Sigmoid statistics ===")
    for row in stats_rows:
        print(f"{row['category']}: {row['test']} | {row['stat_name']}={row['stat_value']:.3f}, p={row['p_value']:.4f}")

# ---------------------------------------------------------------------
# bar plot: total responsive to fit vs reward
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# bar plot: total responsive gaussian vs sigmoid
# ---------------------------------------------------------------------

def plot_total_responsive_gaussian_vs_sigmoid_bars(data_dict, save_dir="."):
    """
    Compare Gaussian vs Sigmoid for total responsive populations.

    Categories:
    - Fit responsive = fit only + both
    - Reward responsive = reward only + both

    Bars:
    - Gaussian
    - Sigmoid

    Connected lines connect the same mouse.
    """

    g_only, g_reward_only, g_both = extract_overlap_arrays(data_dict, "Gaussian_Hit(reward)_All")
    s_only, s_reward_only, s_both = extract_overlap_arrays(data_dict, "Sigmoid_Hit(reward)_All")

    if g_only is None or s_only is None:
        print("Missing Gaussian or Sigmoid data. Cannot plot total responsive bars.")
        return

    # Convert to arrays
    g_only = np.asarray(g_only, dtype=float)
    g_reward_only = np.asarray(g_reward_only, dtype=float)
    g_both = np.asarray(g_both, dtype=float)

    s_only = np.asarray(s_only, dtype=float)
    s_reward_only = np.asarray(s_reward_only, dtype=float)
    s_both = np.asarray(s_both, dtype=float)

    # Total responsive populations
    gaussian_fit_responsive = g_only + g_both
    sigmoid_fit_responsive = s_only + s_both

    gaussian_reward_responsive = g_reward_only + g_both
    sigmoid_reward_responsive = s_reward_only + s_both

    categories = ["Fit responsive", "Reward responsive"]

    gaussian_arrays = [
        gaussian_fit_responsive,
        gaussian_reward_responsive
    ]

    sigmoid_arrays = [
        sigmoid_fit_responsive,
        sigmoid_reward_responsive
    ]

    gaussian_means = [mean_sem(x)[0] for x in gaussian_arrays]
    gaussian_sems = [mean_sem(x)[1] for x in gaussian_arrays]

    sigmoid_means = [mean_sem(x)[0] for x in sigmoid_arrays]
    sigmoid_sems = [mean_sem(x)[1] for x in sigmoid_arrays]

    # Paired stats: Gaussian vs Sigmoid within each category
    stats_rows = []
    for cat, g_vals, s_vals in zip(categories, gaussian_arrays, sigmoid_arrays):
        res = choose_paired_test(g_vals, s_vals)
        res["category"] = cat
        stats_rows.append(res)

    fig, ax = plt.subplots(figsize=(10, 7))

    bar_width = 0.35
    x_groups = np.array([0, 1.2])

    x_gauss = x_groups - bar_width / 2
    x_sig = x_groups + bar_width / 2

    ax.bar(
        x_gauss,
        gaussian_means,
        bar_width,
        yerr=gaussian_sems,
        label="Gaussian",
        color="steelblue",
        edgecolor="black",
        alpha=0.85,
        capsize=4
    )

    ax.bar(
        x_sig,
        sigmoid_means,
        bar_width,
        yerr=sigmoid_sems,
        label="Sigmoid",
        color="indianred",
        edgecolor="black",
        alpha=0.85,
        capsize=4
    )

    # Connect same mice: Gaussian vs Sigmoid within each category
    for group_idx, (g_vals, s_vals) in enumerate(zip(gaussian_arrays, sigmoid_arrays)):
        n_mice = min(len(g_vals), len(s_vals))

        for mouse_idx in range(n_mice):
            ax.plot(
                [x_gauss[group_idx], x_sig[group_idx]],
                [g_vals[mouse_idx], s_vals[mouse_idx]],
                color="gray",
                alpha=0.55,
                linewidth=1,
                linestyle="--",
                marker="o",
                zorder=3
            )

    ax.set_xticks(x_groups)
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylabel("Percentage of total neurons", fontsize=13)
    ax.set_title(
        "Psychophysics: Total Responsive Neurons, Gaussian vs Sigmoid",
        fontsize=15,
        fontweight="bold"
    )

    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right")

    top_vals = [
        max(gm + gs, sm + ss)
        for gm, gs, sm, ss in zip(
            gaussian_means,
            gaussian_sems,
            sigmoid_means,
            sigmoid_sems
        )
    ]

    ymax = max(top_vals) * 1.25 if max(top_vals) > 0 else 1.0
    ax.set_ylim(0, ymax)

    for i, res in enumerate(stats_rows):
        y = top_vals[i] * 1.03
        add_sig_bracket(ax, x_gauss[i], x_sig[i], y, res["p_value"])

    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    out_path = join(save_dir, "Total_Responsive_Gaussian_vs_Sigmoid_Bars.pdf")
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.show()

    print(f"Saved total responsive Gaussian vs Sigmoid bar figure: {out_path}")

    # Save stats
    df_stats = pd.DataFrame(stats_rows)
    stats_path = join(save_dir, "Total_Responsive_Gaussian_vs_Sigmoid_stats_summary.xlsx")
    df_stats.to_excel(stats_path, index=False)

    print(f"Saved stats: {stats_path}")

    print("\n=== Total responsive Gaussian vs Sigmoid statistics ===")
    for row in stats_rows:
        print(
            f"{row['category']}: {row['test']} | "
            f"{row['stat_name']}={row['stat_value']:.3f}, "
            f"p={row['p_value']:.4f}"
        )

def extract_reward_direction_percentages(data_dict, fit_name):
    """
    fit_name: "Gaussian" or "Sigmoid"

    Returns percentages within the Gaussian/Sigmoid population.
    Increase and decrease are non-exclusive because reward and lick are combined.
    """
    all_entries = data_dict[f"{fit_name}_Hit(reward)_All"]
    inc_entries = data_dict[f"{fit_name}_Hit(reward)_Increase"]
    dec_entries = data_dict[f"{fit_name}_Hit(reward)_Decrease"]

    rows = []

    for mouse_idx, (all_d, inc_d, dec_d) in enumerate(
        zip(all_entries, inc_entries, dec_entries)
    ):
        # All values are percentages of the same total-neuron population.
        fit_total = all_d["set1_only"] + all_d["both"]
        responsive_fit = all_d["both"]

        if fit_total > 0:
            pct_any = responsive_fit / fit_total * 100
            pct_increase = inc_d["both"] / fit_total * 100
            pct_decrease = dec_d["both"] / fit_total * 100
        else:
            pct_any = np.nan
            pct_increase = np.nan
            pct_decrease = np.nan

        # Direction among reward/lick-responsive fit neurons.
        if responsive_fit > 0:
            pct_increase_among_responsive = (
                inc_d["both"] / responsive_fit * 100
            )
            pct_decrease_among_responsive = (
                dec_d["both"] / responsive_fit * 100
            )
        else:
            pct_increase_among_responsive = np.nan
            pct_decrease_among_responsive = np.nan

        rows.append({
            "mouse_index": mouse_idx,
            "fit": fit_name,
            "pct_any_reward_lick_response": pct_any,
            "pct_increase_of_all_fit_neurons": pct_increase,
            "pct_decrease_of_all_fit_neurons": pct_decrease,
            "pct_increase_among_responsive": pct_increase_among_responsive,
            "pct_decrease_among_responsive": pct_decrease_among_responsive,
        })

    return pd.DataFrame(rows)
# ---------------------------------------------------------------------
# optional summary table
# ---------------------------------------------------------------------

def create_summary_table(data_dict, save_dir="."):
    g_only, g_reward_only, g_both = extract_overlap_arrays(data_dict, "Gaussian_Hit(reward)_All")
    s_only, s_reward_only, s_both = extract_overlap_arrays(data_dict, "Sigmoid_Hit(reward)_All")

    if g_only is None or s_only is None:
        print("Missing Gaussian or Sigmoid data. Cannot create summary table.")
        return None

    summary = {
        "Metric": ["Fit only", "Reward only", "Both"],
        "Gaussian_Mean": [mean_sem(g_only)[0], mean_sem(g_reward_only)[0], mean_sem(g_both)[0]],
        "Gaussian_SEM": [mean_sem(g_only)[1], mean_sem(g_reward_only)[1], mean_sem(g_both)[1]],
        "Sigmoid_Mean": [mean_sem(s_only)[0], mean_sem(s_reward_only)[0], mean_sem(s_both)[0]],
        "Sigmoid_SEM": [mean_sem(s_only)[1], mean_sem(s_reward_only)[1], mean_sem(s_both)[1]],
    }

    df = pd.DataFrame(summary)
    df["Difference_Sigmoid_minus_Gaussian"] = df["Sigmoid_Mean"] - df["Gaussian_Mean"]

    os.makedirs(save_dir, exist_ok=True)
    csv_path = join(save_dir, "Gaussian_vs_Sigmoid_summary_table.csv")
    df.to_csv(csv_path, index=False)

    print("\n=== Summary table ===")
    print(df)
    print(f"\nSaved summary table to: {csv_path}")
    return df


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------

if __name__ == "__main__":
    psycho_pkl_files = [
        DATA_ROOT + r"\mouse09\sess06\ML\data\processed_data\auROC_freq_psycho_gaussian_sigmpoid\comparison_fractions_240827.pkl",
        DATA_ROOT + r"\mouse10\sess09\ML\data\processed_data\auROC_freq_psycho_gaussian_sigmpoid\comparison_fractions_241003.pkl",
        DATA_ROOT + r"\mouse05\sess04\ML\data\processed_data\auROC_freq_psycho_gaussian_sigmpoid\comparison_fractions_240610.pkl",
        DATA_ROOT + r"\mouse01\sess02\ML\data\processed_data\auROC_freq_psycho_gaussian_sigmpoid\comparison_fractions_240529.pkl",
        DATA_ROOT + r"\mouse14\sess12\ML\data\processed_data\auROC_freq_psycho_gaussian_sigmpoid\comparison_fractions_250227.pkl",
    ]

    save_dir = RESULTS_ROOT + r"\gaussian_vs_sigmoid_reward_comparison_psycho_v3"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    data = load_comparison_data(psycho_pkl_files)

    print(f"\nAvailable keys: {list(data.keys())}")
    print(f"Number of mice in Gaussian_Hit(reward)_All: {len(data.get('Gaussian_Hit(reward)_All', []))}")
    print(f"Number of mice in Sigmoid_Hit(reward)_All: {len(data.get('Sigmoid_Hit(reward)_All', []))}")

    gaussian_direction = extract_reward_direction_percentages(data, "Gaussian")
    sigmoid_direction = extract_reward_direction_percentages(data, "Sigmoid")

    direction_df = pd.concat(
        [gaussian_direction, sigmoid_direction],
        ignore_index=True
    )

    direction_df.to_csv(
        join(save_dir, "Gaussian_Sigmoid_reward_lick_directions.csv"),
        index=False
    )

    print(direction_df)
    plot_gaussian_sigmoid_venns(data, save_dir=save_dir)
    plot_gaussian_vs_sigmoid_connected_bars(data, save_dir=save_dir)
    plot_total_responsive_gaussian_vs_sigmoid_bars(data, save_dir=save_dir)
    create_summary_table(data, save_dir=save_dir)

    print(f"\nAll outputs saved to: {save_dir}")