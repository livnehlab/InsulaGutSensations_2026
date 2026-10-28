#!/usr/bin/env python3
"""
Detection: across-mice comparison of reward-responsive (Hit) vs. light-responsive neurons,
from the 2-set overlap fractions saved by cell_classification_auROC_plots.py
(comparison_fractions_<day>.pkl): bar plots with and without the "both" group, and
separate / combined Venn diagrams of the mean fractions.
"""

import os
import pickle
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from matplotlib_venn import venn2
import matplotlib.patches as patches


# ---------------------------------------------------------------------------
# Data locations (set these before running)
# ---------------------------------------------------------------------------
DATA_ROOT    = r'path\to\2p_data'  # folder with the imaging sessions (<mouse>/<session>/ML/...)
RESULTS_ROOT = r'path\to\results'  # folder where the across-mice results are saved

def mean_sem(values):
    """
    Returns (mean, sem) for a list of numeric values.
    If empty, returns (0, 0).
    """
    n = len(values)
    if n == 0:
        return (0, 0)
    mu = np.mean(values)
    sd = np.std(values, ddof=1)  # sample std
    sem = sd / np.sqrt(n)
    return mu, sem


def load_fraction_data_2set(list_of_pkl):
    """
    Loads 2-set fraction data from multiple pickle files (one per mouse).
    """
    all_2set = {}

    for pkl_file in list_of_pkl:
        print(f"Loading fraction data from: {pkl_file}")
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)

            # Load 2-set data
            if "2set" in data:
                for comp_name, fraction_dict in data["2set"].items():
                    if comp_name not in all_2set:
                        all_2set[comp_name] = []
                    all_2set[comp_name].append(fraction_dict)
        except FileNotFoundError:
            print(f"Warning: File not found: {pkl_file}")
        except Exception as e:
            print(f"Error loading {pkl_file}: {e}")

    return all_2set


def plot_reward_vs_light_comparison(all_2set, save_dir="."):
    """
    Plots a comparison of Hit(Reward) vs LightOn using 2-set fraction data.
    
    Creates a diverging bar chart showing:
    - Hit(Reward) increase/decrease
    - LightOn increase/decrease
    - Overlap (both) increase/decrease
    """
    
    # Look for the Hit(reward)_LightOn comparison
    inc_key = "Hit(reward)_LightOn_Increase"
    dec_key = "Hit(reward)_LightOn_Decrease"
    all_key = "Hit(reward)_LightOn_All"

    if inc_key not in all_2set or dec_key not in all_2set:
        print(f"Data for {inc_key} or {dec_key} not found in all_2set.")
        print(f"Available keys: {list(all_2set.keys())}")
        return
    
    inc_data = all_2set[inc_key]
    dec_data = all_2set[dec_key]
    all_data = all_2set[all_key]
    
    print(f"Processing {len(inc_data)} mice for increase data")
    print(f"Processing {len(dec_data)} mice for decrease data")
    
    # Extract data for each group
    reward_only_inc, light_only_inc, both_inc = [], [], []
    for d in inc_data:
        reward_only_inc.append(d.get("set1_only", 0))  # Hit(reward) only
        light_only_inc.append(d.get("set2_only", 0))   # LightOn only  
        both_inc.append(d.get("both", 0))              # Both Hit(reward) and LightOn
    
    reward_only_dec, light_only_dec, both_dec = [], [], []
    for d in dec_data:
        reward_only_dec.append(d.get("set1_only", 0))  # Hit(reward) only
        light_only_dec.append(d.get("set2_only", 0))   # LightOn only
        both_dec.append(d.get("both", 0))              # Both Hit(reward) and LightOn
    
    # Compute total reward responsive and light responsive
    reward_total_inc = [r + b for r, b in zip(reward_only_inc, both_inc)]
    light_total_inc = [l + b for l, b in zip(light_only_inc, both_inc)]
    reward_total_dec = [r + b for r, b in zip(reward_only_dec, both_dec)]
    light_total_dec = [l + b for l, b in zip(light_only_dec, both_dec)]
    
    # Compute means and SEMs
    reward_inc_mean, reward_inc_sem = mean_sem(reward_total_inc)
    light_inc_mean, light_inc_sem = mean_sem(light_total_inc)
    reward_dec_mean, reward_dec_sem = mean_sem(reward_total_dec)
    light_dec_mean, light_dec_sem = mean_sem(light_total_dec)
    
    print(f"Reward Responsive Increase: {reward_inc_mean:.2f} ± {reward_inc_sem:.2f}")
    print(f"Light Responsive Increase: {light_inc_mean:.2f} ± {light_inc_sem:.2f}")
    print(f"Reward Responsive Decrease: {reward_dec_mean:.2f} ± {reward_dec_sem:.2f}")
    print(f"Light Responsive Decrease: {light_dec_mean:.2f} ± {light_dec_sem:.2f}")
    
    # Create the plot
    x = np.arange(2)  # Two groups
    group_labels = ["Reward Responsive", "Light Responsive"]
    bar_width = 0.6
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot Increase bars (drawn upward)
    bars_inc = ax.bar(x, [reward_inc_mean, light_inc_mean],
                      width=bar_width, color='salmon', edgecolor='black',
                      label='Increase')
    ax.errorbar(x, [reward_inc_mean, light_inc_mean],
                yerr=[reward_inc_sem, light_inc_sem],
                fmt='none', ecolor='black', capsize=3)
    
    # Plot Decrease bars (drawn downward by plotting -mean)
    bars_dec = ax.bar(x, [-reward_dec_mean, -light_dec_mean],
                      width=bar_width, color='skyblue', edgecolor='black',
                      label='Decrease')
    ax.errorbar(x, [-reward_dec_mean, -light_dec_mean],
                yerr=[reward_dec_sem, light_dec_sem],
                fmt='none', ecolor='black', capsize=3)
    
    # Overlay individual data lines for Increase
    for i in range(len(reward_total_inc)):
        ax.plot(x, [reward_total_inc[i], light_total_inc[i]],
                color='salmon', marker='o', linestyle='--', linewidth=1,
                markerfacecolor='salmon', markeredgecolor='black', markeredgewidth=1,
                alpha=0.7, zorder=3)
    
    # Overlay individual data lines for Decrease
    for i in range(len(reward_total_dec)):
        ax.plot(x, [-reward_total_dec[i], -light_total_dec[i]],
                color='skyblue', marker='o', linestyle='--', linewidth=1,
                markerfacecolor='skyblue', markeredgecolor='black', markeredgewidth=1,
                alpha=0.7, zorder=3)
    
    # Add a horizontal line at zero
    ax.axhline(0, color='black', linewidth=0.8)
    
    # Format the plot
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, rotation=45, ha='right')
    
    # Set y-ticks
    yticks = np.arange(-50, 51, 10)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{abs(y)}%" for y in yticks])
    ax.set_ylabel("Percentage of total neurons")
    ax.set_title("Hit(Reward) vs LightOn Comparison")
    ax.legend()
    
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(join(save_dir, "Hit_Reward_vs_LightOn_Comparison.pdf"))
    plt.show()


def plot_reward_vs_light_comparison_with_both(all_2set, save_dir="."):
    """
    Plots a comparison of Hit(Reward) vs LightOn using 2-set fraction data.
    
    Creates a diverging bar chart showing:
    - Hit(Reward) only increase/decrease
    - LightOn only increase/decrease
    - Both (overlap) increase/decrease
    """
    
    # Look for the Hit(reward)_LightOn comparison
    inc_key = "Hit(reward)_LightOn_Increase"
    dec_key = "Hit(reward)_LightOn_Decrease"
    
    if inc_key not in all_2set or dec_key not in all_2set:
        print(f"Data for {inc_key} or {dec_key} not found in all_2set.")
        print(f"Available keys: {list(all_2set.keys())}")
        return
    
    inc_data = all_2set[inc_key]
    dec_data = all_2set[dec_key]
    
    print(f"Processing {len(inc_data)} mice for increase data")
    print(f"Processing {len(dec_data)} mice for decrease data")
    
    # Extract data for each group
    reward_only_inc, light_only_inc, both_inc = [], [], []
    for d in inc_data:
        reward_only_inc.append(d.get("set1_only", 0))  # Hit(reward) only
        light_only_inc.append(d.get("set2_only", 0))   # LightOn only  
        both_inc.append(d.get("both", 0))              # Both Hit(reward) and LightOn
    
    reward_only_dec, light_only_dec, both_dec = [], [], []
    for d in dec_data:
        reward_only_dec.append(d.get("set1_only", 0))  # Hit(reward) only
        light_only_dec.append(d.get("set2_only", 0))   # LightOn only
        both_dec.append(d.get("both", 0))              # Both Hit(reward) and LightOn
    
    # Compute means and SEMs for all three categories
    reward_inc_mean, reward_inc_sem = mean_sem(reward_only_inc)
    light_inc_mean, light_inc_sem = mean_sem(light_only_inc)
    both_inc_mean, both_inc_sem = mean_sem(both_inc)
    
    reward_dec_mean, reward_dec_sem = mean_sem(reward_only_dec)
    light_dec_mean, light_dec_sem = mean_sem(light_only_dec)
    both_dec_mean, both_dec_sem = mean_sem(both_dec)
    
    print(f"Reward Only Increase: {reward_inc_mean:.2f} ± {reward_inc_sem:.2f}")
    print(f"Light Only Increase: {light_inc_mean:.2f} ± {light_inc_sem:.2f}")
    print(f"Both Increase: {both_inc_mean:.2f} ± {both_inc_sem:.2f}")
    print(f"Reward Only Decrease: {reward_dec_mean:.2f} ± {reward_dec_sem:.2f}")
    print(f"Light Only Decrease: {light_dec_mean:.2f} ± {light_dec_sem:.2f}")
    print(f"Both Decrease: {both_dec_mean:.2f} ± {both_dec_sem:.2f}")
    
    # Create the plot with three groups
    x = np.arange(3)  # Three groups now
    group_labels = ["Reward Only", "Light Only", "Both"]
    bar_width = 0.6
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot Increase bars (drawn upward)
    bars_inc = ax.bar(x, [reward_inc_mean, light_inc_mean, both_inc_mean],
                      width=bar_width, color='salmon', edgecolor='black',
                      label='Increase')
    ax.errorbar(x, [reward_inc_mean, light_inc_mean, both_inc_mean],
                yerr=[reward_inc_sem, light_inc_sem, both_inc_sem],
                fmt='none', ecolor='black', capsize=3)
    
    # Plot Decrease bars (drawn downward by plotting -mean)
    bars_dec = ax.bar(x, [-reward_dec_mean, -light_dec_mean, -both_dec_mean],
                      width=bar_width, color='skyblue', edgecolor='black',
                      label='Decrease')
    ax.errorbar(x, [-reward_dec_mean, -light_dec_mean, -both_dec_mean],
                yerr=[reward_dec_sem, light_dec_sem, both_dec_sem],
                fmt='none', ecolor='black', capsize=3)
    
    # Overlay individual data lines for Increase
    for i in range(len(reward_only_inc)):
        ax.plot(x, [reward_only_inc[i], light_only_inc[i], both_inc[i]],
                color='salmon', marker='o', linestyle='--', linewidth=1,
                markerfacecolor='salmon', markeredgecolor='black', markeredgewidth=1,
                alpha=0.7, zorder=3)
    
    # Overlay individual data lines for Decrease
    for i in range(len(reward_only_dec)):
        ax.plot(x, [-reward_only_dec[i], -light_only_dec[i], -both_dec[i]],
                color='skyblue', marker='o', linestyle='--', linewidth=1,
                markerfacecolor='skyblue', markeredgecolor='black', markeredgewidth=1,
                alpha=0.7, zorder=3)
    
    # Add a horizontal line at zero
    ax.axhline(0, color='black', linewidth=0.8)
    
    # Format the plot
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, rotation=45, ha='right')
    
    # Set y-ticks
    yticks = np.arange(-50, 51, 10)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{abs(y)}%" for y in yticks])
    ax.set_ylabel("Percentage of total neurons")
    ax.set_title("Hit(Reward) vs LightOn Comparison (with Both category)")
    ax.legend()
    
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(join(save_dir, "Hit_Reward_vs_LightOn_Comparison_with_Both.pdf"))
    plt.show()


def plot_reward_vs_light_venn_diagram(all_2set, save_dir="."):
    """
    Creates Venn diagrams showing the overlap between Hit(Reward) and LightOn responsive cells.
    
    Creates separate Venn diagrams for:
    - Cells with increased activity
    - Cells with decreased activity
    """
    
    # Look for the Hit(reward)_LightOn comparison
    inc_key = "Hit(reward)_LightOn_Increase"
    dec_key = "Hit(reward)_LightOn_Decrease"
    
    if inc_key not in all_2set or dec_key not in all_2set:
        print(f"Data for {inc_key} or {dec_key} not found in all_2set.")
        print(f"Available keys: {list(all_2set.keys())}")
        return
    
    inc_data = all_2set[inc_key]
    dec_data = all_2set[dec_key]
    
    print(f"Processing {len(inc_data)} mice for increase data")
    print(f"Processing {len(dec_data)} mice for decrease data")
    
    # Extract data for each group and calculate averages across mice
    reward_only_inc, light_only_inc, both_inc = [], [], []
    for d in inc_data:
        reward_only_inc.append(d.get("set1_only", 0))
        light_only_inc.append(d.get("set2_only", 0))
        both_inc.append(d.get("both", 0))
    
    reward_only_dec, light_only_dec, both_dec = [], [], []
    for d in dec_data:
        reward_only_dec.append(d.get("set1_only", 0))
        light_only_dec.append(d.get("set2_only", 0))
        both_dec.append(d.get("both", 0))
    
    # Calculate means and SEMs for Venn diagram
    reward_only_inc_mean, reward_only_inc_sem = mean_sem(reward_only_inc)
    light_only_inc_mean, light_only_inc_sem = mean_sem(light_only_inc)
    both_inc_mean, both_inc_sem = mean_sem(both_inc)
    
    reward_only_dec_mean, reward_only_dec_sem = mean_sem(reward_only_dec)
    light_only_dec_mean, light_only_dec_sem = mean_sem(light_only_dec)
    both_dec_mean, both_dec_sem = mean_sem(both_dec)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Venn diagram for Increase
    venn_inc = venn2(subsets=(reward_only_inc_mean, light_only_inc_mean, both_inc_mean),
                     set_labels=('Reward\nResponsive', 'Light\nResponsive'),
                     ax=ax1)
    
    # Customize colors for increase (INVERTED)
    if venn_inc.get_patch_by_id('10'):  # Reward only
        venn_inc.get_patch_by_id('10').set_color('lightblue')  # Changed from lightcoral
        venn_inc.get_patch_by_id('10').set_alpha(0.7)
    if venn_inc.get_patch_by_id('01'):  # Light only
        venn_inc.get_patch_by_id('01').set_color('lightcoral')  # Changed from lightblue
        venn_inc.get_patch_by_id('01').set_alpha(0.7)
    if venn_inc.get_patch_by_id('11'):  # Both
        venn_inc.get_patch_by_id('11').set_color('mediumpurple')  # Changed from lightpurple
        venn_inc.get_patch_by_id('11').set_alpha(0.7)
    
    # Add percentage labels with SEM for increase
    if venn_inc.get_label_by_id('10'):
        venn_inc.get_label_by_id('10').set_text(f'{reward_only_inc_mean:.1f}±{reward_only_inc_sem:.1f}%')
    if venn_inc.get_label_by_id('01'):
        venn_inc.get_label_by_id('01').set_text(f'{light_only_inc_mean:.1f}±{light_only_inc_sem:.1f}%')
    if venn_inc.get_label_by_id('11'):
        venn_inc.get_label_by_id('11').set_text(f'{both_inc_mean:.1f}±{both_inc_sem:.1f}%')
    
    ax1.set_title('Cells with Increased Activity', fontsize=14, fontweight='bold')
    
    # Venn diagram for Decrease
    venn_dec = venn2(subsets=(reward_only_dec_mean, light_only_dec_mean, both_dec_mean),
                     set_labels=('Reward\nResponsive', 'Light\nResponsive'),
                     ax=ax2)
    
    # Customize colors for decrease (INVERTED)
    if venn_dec.get_patch_by_id('10'):  # Reward only
        venn_dec.get_patch_by_id('10').set_color('lightblue')  # Changed from lightcoral
        venn_dec.get_patch_by_id('10').set_alpha(0.7)
    if venn_dec.get_patch_by_id('01'):  # Light only
        venn_dec.get_patch_by_id('01').set_color('lightcoral')  # Changed from lightblue
        venn_dec.get_patch_by_id('01').set_alpha(0.7)
    if venn_dec.get_patch_by_id('11'):  # Both
        venn_dec.get_patch_by_id('11').set_color('mediumpurple')  # Changed from lightpurple
        venn_dec.get_patch_by_id('11').set_alpha(0.7)
    
    # Add percentage labels with SEM for decrease
    if venn_dec.get_label_by_id('10'):
        venn_dec.get_label_by_id('10').set_text(f'{reward_only_dec_mean:.1f}±{reward_only_dec_sem:.1f}%')
    if venn_dec.get_label_by_id('01'):
        venn_dec.get_label_by_id('01').set_text(f'{light_only_dec_mean:.1f}±{light_only_dec_sem:.1f}%')
    if venn_dec.get_label_by_id('11'):
        venn_dec.get_label_by_id('11').set_text(f'{both_dec_mean:.1f}±{both_dec_sem:.1f}%')
    
    ax2.set_title('Cells with Decreased Activity', fontsize=14, fontweight='bold')
    
    # Add overall title
    fig.suptitle('Hit(Reward) vs LightOn Responsive Cells', fontsize=16, fontweight='bold')
    
    # Add legend (UPDATED COLORS)
    legend_elements = [
        patches.Patch(color='lightblue', alpha=0.7, label='Reward Only'),  # Changed from lightcoral
        patches.Patch(color='lightcoral', alpha=0.7, label='Light Only'),  # Changed from lightblue
        patches.Patch(color='mediumpurple', alpha=0.7, label='Both')  # Changed from mediumpurple
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.05))
    
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(join(save_dir, "Hit_Reward_vs_LightOn_Venn_Diagram.pdf"), bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print("\n=== Venn Diagram Summary ===")
    print("INCREASE:")
    print(f"  Reward Only: {reward_only_inc_mean:.1f} ± {reward_only_inc_sem:.1f}%")
    print(f"  Light Only: {light_only_inc_mean:.1f} ± {light_only_inc_sem:.1f}%")
    print(f"  Both: {both_inc_mean:.1f} ± {both_inc_sem:.1f}%")
    print(f"  Total Reward Responsive: {reward_only_inc_mean + both_inc_mean:.1f}%")
    print(f"  Total Light Responsive: {light_only_inc_mean + both_inc_mean:.1f}%")
    
    print("\nDECREASE:")
    print(f"  Reward Only: {reward_only_dec_mean:.1f} ± {reward_only_dec_sem:.1f}%")
    print(f"  Light Only: {light_only_dec_mean:.1f} ± {light_only_dec_sem:.1f}%")
    print(f"  Both: {both_dec_mean:.1f} ± {both_dec_sem:.1f}%")
    print(f"  Total Reward Responsive: {reward_only_dec_mean + both_dec_mean:.1f}%")
    print(f"  Total Light Responsive: {light_only_dec_mean + both_dec_mean:.1f}%")


def plot_reward_vs_light_venn_combined(all_2set, save_dir="."):
    """
    Draw a Venn diagram of Hit(Reward) vs LightOn responsive cells using the
    collapsed 'All' fractions. Works with either a single dict or a list of dicts.
    Expects keys 'set1_only', 'set2_only', 'both' where:
      - set1_only = Reward only
      - set2_only = Light only
      - both      = Reward ∩ Light
    """
    import os
    from os.path import join
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib_venn import venn2
    import matplotlib.patches as patches

    all_key = "Hit(reward)_LightOn_All"
    if all_key not in all_2set:
        # fallback: try to compute from increase/decrease if present (optional)
        raise KeyError(f"'{all_key}' not found in all_2set. Available: {list(all_2set.keys())}")

    all_data = all_2set[all_key]

    # Normalize to a list of dicts
    if isinstance(all_data, dict):
        per_items = [all_data]
    elif isinstance(all_data, (list, tuple)):
        per_items = list(all_data)
        if len(per_items) == 0:
            raise ValueError(f"{all_key} is an empty list")
    else:
        raise TypeError(f"Unsupported type for {all_key}: {type(all_data)}")

    # Validate structure
    required = {"set1_only", "set2_only", "both"}
    for idx, d in enumerate(per_items):
        if not isinstance(d, dict) or not required.issubset(d.keys()):
            raise ValueError(f"Item {idx} under {all_key} must be a dict with keys {required}, got {d}")

    # Aggregate: mean across mice if list, otherwise the single value
    reward_only_vals = [float(d["set1_only"]) for d in per_items]
    light_only_vals  = [float(d["set2_only"]) for d in per_items]
    both_vals        = [float(d["both"])      for d in per_items]

    # Helpers
    def _mean(x):
        return float(np.mean(x)) if len(x) > 0 else 0.0

    def _sem(x):
        x = np.asarray(x, dtype=float)
        n = x.size
        if n <= 1:
            return 0.0
        return float(np.std(x, ddof=1) / np.sqrt(n))

    # Means and SEMs (percent units, as in the inputs)
    reward_only_mean, reward_only_sem = _mean(reward_only_vals), _sem(reward_only_vals)
    light_only_mean,  light_only_sem  = _mean(light_only_vals),  _sem(light_only_vals)
    both_mean,        both_sem        = _mean(both_vals),        _sem(both_vals)

    # Also compute overlap-out-of-total per mouse, then mean ± SEM of that ratio
    per_mouse_total = np.array(reward_only_vals) + np.array(light_only_vals) + np.array(both_vals)
    valid = per_mouse_total > 0
    overlap_pct_each = np.zeros_like(per_mouse_total, dtype=float)
    overlap_pct_each[valid] = (np.array(both_vals)[valid] / per_mouse_total[valid]) * 100.0
    overlap_pct_mean, overlap_pct_sem = _mean(overlap_pct_each[valid]), _sem(overlap_pct_each[valid])
    reward_only_mean = float(np.mean(reward_only_vals))
    light_only_mean  = float(np.mean(light_only_vals))
    both_mean        = float(np.mean(both_vals))

    total_reward_responsive = reward_only_mean + both_mean
    total_light_responsive  = light_only_mean + both_mean
    total_responsive        = reward_only_mean + light_only_mean + both_mean

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    venn = venn2(
        subsets=(reward_only_mean, light_only_mean, both_mean),
        set_labels=('Reward\nResponsive', 'Light\nResponsive'),
        ax=ax
    )

    # Colors
    if venn.get_patch_by_id('10'):
        venn.get_patch_by_id('10').set_color('lightblue')
        venn.get_patch_by_id('10').set_alpha(0.7)
    if venn.get_patch_by_id('01'):
        venn.get_patch_by_id('01').set_color('lightcoral')
        venn.get_patch_by_id('01').set_alpha(0.7)
    if venn.get_patch_by_id('11'):
        venn.get_patch_by_id('11').set_color('mediumpurple')
        venn.get_patch_by_id('11').set_alpha(0.7)

    # Percent labels
    # Percent labels with SEM
    if venn.get_label_by_id('10'):
        venn.get_label_by_id('10').set_text(f'{reward_only_mean:.1f}±{reward_only_sem:.1f}%')
        venn.get_label_by_id('10').set_fontsize(12)
        venn.get_label_by_id('10').set_fontweight('bold')
    if venn.get_label_by_id('01'):
        venn.get_label_by_id('01').set_text(f'{light_only_mean:.1f}±{light_only_sem:.1f}%')
        venn.get_label_by_id('01').set_fontsize(12)
        venn.get_label_by_id('01').set_fontweight('bold')
    if venn.get_label_by_id('11'):
        venn.get_label_by_id('11').set_text(f'{both_mean:.1f}±{both_sem:.1f}%')
        venn.get_label_by_id('11').set_fontsize(12)
        venn.get_label_by_id('11').set_fontweight('bold')

    ax.set_title('Hit(Reward) vs LightOn Responsive Cells\n(Combined Increase + Decrease)',
                 fontsize=16, fontweight='bold', pad=20)

    legend_elements = [
        patches.Patch(color='lightblue', alpha=0.7, label='Reward Only'),
        patches.Patch(color='lightcoral', alpha=0.7, label='Light Only'),
        patches.Patch(color='mediumpurple', alpha=0.7, label='Both')
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.3, 1))

    # Summary box
    overlap_out_of_total = (both_mean / total_responsive * 100) if total_responsive > 0 else 0.0
    # Summary box with mean ± SEM
    total_reward_responsive = reward_only_mean + both_mean
    total_light_responsive  = light_only_mean  + both_mean
    total_responsive        = reward_only_mean + light_only_mean + both_mean

    summary_text = (
        f"Summary Statistics (mean ± SEM):\n"
        f"Reward only: {reward_only_mean:.1f}±{reward_only_sem:.1f}%\n"
        f"Light only: {light_only_mean:.1f}±{light_only_sem:.1f}%\n"
        f"Both: {both_mean:.1f}±{both_sem:.1f}%\n"
        f"Total Reward Responsive: {total_reward_responsive:.1f}%\n"
        f"Total Light Responsive: {total_light_responsive:.1f}%\n"
        f"Total Responsive: {total_responsive:.1f}%\n"
        f"Overlap (out of total responsive): {overlap_pct_mean:.1f}±{overlap_pct_sem:.1f}%"
    )
    ax.text(1.05, 0.5, summary_text, transform=ax.transAxes, fontsize=10,
            va='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    # Optional: side bar chart with error bars
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    ax_inset = inset_axes(ax, width="35%", height="45%", loc="lower left", borderpad=2)
    cats = ["Reward only", "Light only", "Both"]
    means = [reward_only_mean, light_only_mean, both_mean]
    sems  = [reward_only_sem,  light_only_sem,  both_sem]
    x = np.arange(len(cats))
    ax_inset.bar(x, means, yerr=sems, capsize=4)
    ax_inset.set_xticks(x)
    ax_inset.set_xticklabels(["R-only", "L-only", "Both"], rotation=0)
    ax_inset.set_ylabel("%")
    ax_inset.set_title("Mean ± SEM", fontsize=10)
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(join(save_dir, "Hit_Reward_vs_LightOn_Venn_Combined.pdf"), bbox_inches='tight')
    plt.show()

    # Console summary
    print("\n=== Combined Venn Diagram Summary ===")
    print(f"Reward Only: {reward_only_mean:.1f}%")
    print(f"Light Only: {light_only_mean:.1f}%")
    print(f"Both: {both_mean:.1f}%")
    print(f"Total Reward Responsive: {total_reward_responsive:.1f}%")
    print(f"Total Light Responsive: {total_light_responsive:.1f}%")
    print(f"Total Responsive: {total_responsive:.1f}%")
    print(f"Overlap Percentage: {overlap_out_of_total:.1f}% of all responsive cells")

    # Print the per-mouse data row-wise
    if len(per_items) > 1:
        print(f"\nIndividual mouse data (n={len(per_items)}):")
        print("Mouse | Reward Only | Light Only | Both | Total Reward | Total Light")
        print("-" * 70)
        for i, d in enumerate(per_items, start=1):
            ro = d["set1_only"]; lo = d["set2_only"]; bt = d["both"]
            print(f"{i:5d} | {ro:11.1f} | {lo:10.1f} | {bt:4.1f} | {ro+bt:12.1f} | {lo+bt:11.1f}")
if __name__ == "__main__":
    # One comparison_fractions file per mouse (detection sessions)
    pkl_files = [
        # file paths:
        DATA_ROOT + r"\mouse01\sess01\ML\data\processed_data\auROC_z_no_sub_history\comparison_fractions_240519.pkl",
        DATA_ROOT + r"\mouse05\sess03\ML\data\processed_data\auROC_z_no_sub_history\comparison_fractions_240530.pkl",
        # DATA_ROOT + r"\mouse16\sess14\ML\data\processed_data\auROC_z_no_sub_history\comparison_fractions_250326.pkl",
        DATA_ROOT + r"\mouse09\sess05\ML\data\processed_data\auROC_z_no_sub_history\comparison_fractions_240826.pkl",
        DATA_ROOT + r"\mouse14\sess11\ML\data\processed_data\auROC_z_no_sub_history\comparison_fractions_250224.pkl",
        DATA_ROOT + r"\mouse10\sess07\ML\data\processed_data\auROC_z_no_sub_history\comparison_fractions_240829.pkl",
    ]
    # pkl_files = [
    #     # file paths:
    #     DATA_ROOT + r"\mouse16\sess15\ML\data\processed_data\auROC_v2\comparison_fractions_250401.pkl",
    #     DATA_ROOT + r"\mouse09\sess10\ML\data\processed_data\auROC_v2\comparison_fractions_241030.pkl",
    #     DATA_ROOT + r"\mouse14\sess13\ML\data\processed_data\auROC_v2\comparison_fractions_250310.pkl",
    #     DATA_ROOT + r"\mouse10\sess08\ML\data\processed_data\auROC_v2\comparison_fractions_240926.pkl",
    # ]
    save_dir = RESULTS_ROOT + r"\detection_reward_vs_light_fractions_v2"
    os.makedirs(save_dir, exist_ok=True)
    
    if pkl_files:
        all_2set = load_fraction_data_2set(pkl_files)
        
        # Generate all four plots
        plot_reward_vs_light_comparison(all_2set, save_dir=save_dir)  # Bar plot without "Both"
        plot_reward_vs_light_comparison_with_both(all_2set, save_dir=save_dir)  # Bar plot with "Both"
        plot_reward_vs_light_venn_diagram(all_2set, save_dir=save_dir)  # Separate Venn diagrams
        plot_reward_vs_light_venn_combined(all_2set, save_dir=save_dir)  # Combined Venn diagram
    else:
        print("Please add your pickle file paths to the pkl_files list")