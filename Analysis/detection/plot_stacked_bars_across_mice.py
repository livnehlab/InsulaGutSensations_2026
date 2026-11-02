
"""
Across-mice summary of the overlap between response categories (detection task).

Loads the per-mouse fractions of responsive neurons (comparison_fractions_<day>.pkl,
saved by cell_classification_auROC_plots.py) and plots the mean +/- SEM across mice:
stacked bars for 2-set comparisons, and diverging (increase up / decrease down) bars
for the CR(light) / Hit(reward) / FA(light) 3-set comparison.
"""

import os
import pickle
from os.path import join

import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
import matplotlib
from matplotlib_venn import venn3, venn2

matplotlib.use('Qt5Agg')



# ---------------------------------------------------------------------------
# Data locations (set these before running)
# ---------------------------------------------------------------------------
DATA_ROOT    = r'path\to\2p_data'  # folder with the imaging sessions (<mouse>/<session>/ML/...)
RESULTS_ROOT = r'path\to\results'  # folder where the across-mice results are saved

###############################################################################
# 1) mean_sem helper
###############################################################################
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


###############################################################################
# 2) Loading the saved fraction data
###############################################################################
def load_fraction_data(list_of_pkl):
    """
    Expects each .pkl to contain a dict with keys "2set" and/or "3set",
    each mapping to a dict of:
       comparison_name -> { <fraction fields> }

    Example .pkl content for one mouse:
      {
        "2set": {
          "ComparisonA": {"set1_only": 15.2, "set2_only": 5.0, "both": 2.3},
          "ComparisonB": {"set1_only": 9.1,  "set2_only": 4.5, "both": 1.2},
          ...
        },
        "3set": {
          "ComparisonX": {"set1_only": 10.0, "set2_only": 3.0, "set3_only": 5.0,
                          "set1_set2": 2.0, "set1_set3": 1.0, "set2_set3": 1.5, "all_three": 0.7},
          ...
        }
      }

    These are merged across mice by appending them into a list:
      all_2set[comp_name] = [ { "set1_only":..., "set2_only":..., "both":... }, ... ]  # one per mouse
      all_3set[comp_name] = [ { "set1_only":..., "set2_only":..., "set3_only":..., ... }, ... ]  # one per mouse

    Returns (all_2set, all_3set).
    """
    all_2set = {}
    all_3set = {}

    for pkl_file in list_of_pkl:
        print(f"Loading fraction data from: {pkl_file}")
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)

        # If there's "2set" data, merge it
        if "2set" in data:
            for comp_name, fraction_dict in data["2set"].items():
                # fraction_dict is like {"set1_only": X, "set2_only": Y, "both": Z}
                if comp_name not in all_2set:
                    all_2set[comp_name] = []
                all_2set[comp_name].append(fraction_dict)

        # If there's "3set" data, merge it
        if "3set" in data:
            for comp_name, fraction_dict in data["3set"].items():
                # fraction_dict is like {"set1_only": X, "set2_only": Y, "set3_only": Z, ..., "all_three": W}
                if comp_name not in all_3set:
                    all_3set[comp_name] = []
                all_3set[comp_name].append(fraction_dict)

    return all_2set, all_3set


###############################################################################
# 3) Plotting: 2-Set stacked bar with error bars
###############################################################################
def plot_stacked_bar_2set_percent(all_2set):
    """
    all_2set is a dict:
      comp_name -> list of fraction dicts, each fraction dict has:
        {"set1_only":..., "set2_only":..., "both":...}

    Computes the mean & SEM across mice and plots a stacked bar chart:
      - bottom = set1_only
      - middle = both
      - top    = set2_only
    """
    # 1) Filter out empty comparisons
    comp_names = sorted([c for c in all_2set if len(all_2set[c]) > 0])
    if len(comp_names) == 0:
        print("No 2-set comparisons to plot.")
        return

    # 2) Prepare arrays to hold means & sem
    means_s1_only = []
    sems_s1_only  = []
    means_both    = []
    sems_both     = []
    means_s2_only = []
    sems_s2_only  = []

    # 3) For each comparison, gather per-mouse fraction dict
    for comp_name in comp_names:
        mouse_list = all_2set[comp_name]  # e.g. [ {set1_only: X, set2_only: Y, both:Z}, ...]
        vals_s1_only = [m["set1_only"] for m in mouse_list]
        vals_both    = [m["both"]      for m in mouse_list]
        vals_s2_only = [m["set2_only"] for m in mouse_list]

        mu1, se1 = mean_sem(vals_s1_only)
        mu2, se2 = mean_sem(vals_both)
        mu3, se3 = mean_sem(vals_s2_only)

        means_s1_only.append(mu1)
        sems_s1_only.append(se1)
        means_both.append(mu2)
        sems_both.append(se2)
        means_s2_only.append(mu3)
        sems_s2_only.append(se3)

    # 4) Plot
    x = np.arange(len(comp_names))

    # Pastel colors
    color_s1  = "#FFD1DC"  # pastel pink
    color_bth = "#C5E1A5"  # pastel green
    color_s2  = "#B39DDB"  # pastel purple

    fig, ax = plt.subplots(figsize=(8,5))

    # Bottom = set1_only
    bar_s1 = ax.bar(x, means_s1_only, color=color_s1, label="Group1 Only")
    top_s1 = np.array(means_s1_only)
    ax.errorbar(x, top_s1, yerr=sems_s1_only, fmt='none', ecolor='black', capsize=3)

    # Middle = both
    bottom_bth = top_s1
    bar_bth = ax.bar(x, means_both, bottom=bottom_bth, color=color_bth, label="Both")
    top_bth = bottom_bth + means_both
    ax.errorbar(x, top_bth, yerr=sems_both, fmt='none', ecolor='black', capsize=3)

    # Top = set2_only
    bottom_s2 = top_bth
    bar_s2 = ax.bar(x, means_s2_only, bottom=bottom_s2, color=color_s2, label="Group2 Only")
    top_s2 = bottom_s2 + means_s2_only
    ax.errorbar(x, top_s2, yerr=sems_s2_only, fmt='none', ecolor='black', capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels(comp_names, rotation=45, ha="right")
    ax.set_ylabel("Percentage of total neurons")
    ax.set_title("")
    ax.legend()
    plt.tight_layout()
    plt.show()


###############################################################################
# 4) Plotting: 3-Set stacked bar with error bars
###############################################################################
def plot_stacked_bar_3set_percent(all_3set):
    """
    all_3set is a dict:
      comp_name -> list of fraction dicts, each fraction dict has:
        {
          "set1_only", "set2_only", "set3_only",
          "set1_set2", "set1_set3", "set2_set3",
          "all_three"
        }

    Computes the mean + SEM across mice and plots a stacked bar with 7 segments.
    """
    comp_names = sorted([c for c in all_3set if len(all_3set[c]) > 0])
    if len(comp_names) == 0:
        print("No 3-set comparisons to plot.")
        return

    segment_order = [
        "set1_only", "set2_only", "set3_only",
        "set1_set2", "set1_set3", "set2_set3",
        "all_three"
    ]
    segment_labels = ["Group1 Only", "Group2 Only", "Group3 Only",
                      "Group1&Group2", "Group1&Group3", "Group2&Group3", "All Three"]
    pastel_colors = [
        "#FFD1DC",  # pastel pink
        "#C5E1A5",  # pastel green
        "#F8BBD0",  # pastel rose
        "#E6EE9C",  # pastel lime
        "#B39DDB",  # pastel purple
        "#FFCC80",  # pastel orange
        "#CFD8DC",  # pastel gray
    ]

    # means & sem per segment
    means = {seg: [] for seg in segment_order}
    sems  = {seg: [] for seg in segment_order}

    for comp_name in comp_names:
        mouse_list = all_3set[comp_name]
        # For each segment, gather values across mice
        for seg in segment_order:
            vals = [m[seg] for m in mouse_list]
            mu, se = mean_sem(vals)
            means[seg].append(mu)
            sems[seg].append(se)

    x = np.arange(len(comp_names))
    fig, ax = plt.subplots(figsize=(10,5))

    bottom = np.zeros(len(comp_names))
    for i, seg in enumerate(segment_order):
        seg_means = np.array(means[seg])
        seg_sems  = np.array(sems[seg])
        color     = pastel_colors[i]
        label     = segment_labels[i]

        bars = ax.bar(x, seg_means, bottom=bottom, color=color, label=label)
        tops = bottom + seg_means
        ax.errorbar(x, tops, yerr=seg_sems, fmt='none', ecolor='black', capsize=3)

        bottom = bottom + seg_means

    ax.set_xticks(x)
    ax.set_xticklabels(comp_names, rotation=45, ha="right")
    ax.set_ylabel("Percentage of total neurons")
    ax.set_title("")
    ax.legend()
    plt.tight_layout()
    plt.show()


###############################################################################
# 5) Plotting: Increase vs. Decrease Diverging Bar Plot
###############################################################################
def plot_increase_decrease_for_3set(all_3set, common_condition):
    """
    Given the 3-set data (a dict mapping comp_name to list of fraction dictionaries)
    and a common condition string (e.g., "CR(LightOn)_Hit(Reward)_FA(LightOn)"),
    this function looks for the keys:
         common_condition + "_Increase"
         common_condition + "_Decrease"

    It computes the mean and SEM (across mice) for each segment:
         "set1_only", "set1_set3", "all_three", "set2_only", "set1_set2"

    and plots a bar chart where for each segment the Increase value is drawn upward
    (positive) and the Decrease value is drawn downward (by plotting its mean as negative).

    Additionally, the individual data points for each mouse are plotted and the points
    from the same mouse are connected with a line.
    """
    # Construct the keys
    inc_key = f"{common_condition}_Increase"
    dec_key = f"{common_condition}_Decrease"

    if inc_key not in all_3set or dec_key not in all_3set:
        print(f"Data for {inc_key} or {dec_key} not found in all_3set.")
        return

    # Each key maps to a list of fraction dictionaries (one per mouse)
    inc_data = all_3set[inc_key]
    dec_data = all_3set[dec_key]

    # Segments and x-axis labels (in plotting order)
    segments = ["set1_only", "set1_set3", "all_three", "set1_set2", "set2_only"] #,
    labels = ["CR Only", "CR&FA",  "All Three", "CR & Reward", "Reward Only"] #,

    # Compute means and SEMs for each segment for both Increase and Decrease groups.
    inc_means, inc_sems = [], []
    dec_means, dec_sems = [], []

    for seg in segments:
        # For Increase:
        inc_vals = [d.get(seg, 0) for d in inc_data]
        m_inc, se_inc = mean_sem(inc_vals)
        inc_means.append(m_inc)
        inc_sems.append(se_inc)

        # For Decrease:
        dec_vals = [d.get(seg, 0) for d in dec_data]
        m_dec, se_dec = mean_sem(dec_vals)
        dec_means.append(m_dec)
        dec_sems.append(se_dec)

    # Set up the plot.
    x = np.arange(len(segments))
    bar_width = 0.6
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot Increase bars (drawn upward, as positive values)
    bars_inc = ax.bar(x, inc_means, width=bar_width, color='skyblue',
                      edgecolor='black', label='Increase')
    ax.errorbar(x, inc_means, yerr=inc_sems, fmt='none', ecolor='black', capsize=3)

    # Plot Decrease bars (drawn downward by plotting -mean).
    bars_dec = ax.bar(x, [-m for m in dec_means], width=bar_width, color='salmon',
                      edgecolor='black', label='Decrease')
    ax.errorbar(x, [-m for m in dec_means], yerr=dec_sems, fmt='none', ecolor='black', capsize=3)

    # Plot individual data points and connect them with lines.
    # Increase data (plotted as is)
    for d in inc_data:
        y_points = [d.get(seg, 0) for seg in segments]
        ax.plot(x, y_points, color='skyblue', marker='o', linestyle='--', linewidth=1,
                markerfacecolor='skyblue', markeredgecolor='black', markeredgewidth=1,
                alpha=0.7, zorder=3)
    # Decrease data (plotted as negative)
    for d in dec_data:
        y_points = [-d.get(seg, 0) for seg in segments]
        ax.plot(x, y_points, color='salmon', marker='o', linestyle='--', linewidth=1,
                markerfacecolor='salmon', markeredgecolor='black', markeredgewidth=1,
                alpha=0.7, zorder=3)

    # Add a horizontal line at zero.
    ax.axhline(0, color='black', linewidth=0.8)

    # Format the x-axis.
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    # Set y-ticks; labels show absolute percentages.
    ax.set_yticks(np.arange(-40, 41, 10))
    ax.set_yticklabels([f"{abs(y)}%" for y in np.arange(-40, 41, 10)])
    ax.set_ylabel("Percentage of total neurons")
    ax.set_title(f"{common_condition}")
    ax.legend()

    plt.tight_layout()
    plt.savefig(join(save_dir, f"{common_condition}_Increase_Decrease.pdf"))
    plt.show()


def plot_increase_decrease_two_groups(all_3set, common_condition):
    """
    Given the 3-set data (a dict mapping comp_name to list of fraction dictionaries)
    and a common condition string (e.g., "CR(LightOn)_Hit(Reward)_FA(LightOn)"),
    this function looks for the keys:
         common_condition + "_Increase"
         common_condition + "_Decrease"

    It aggregates each mouse's data into two groups:
      • Reward Responsive = set2_only + set1_set2 + set2_set3 + all_three
      • Light Responsive  = set3_only + set1_set3 + set2_set3 + all_three

    Then it computes the mean and SEM (across mice) for both Increase and Decrease values,
    and plots a diverging bar chart with two bars:
      - The Increase value is drawn upward (positive).
      - The Decrease value is drawn downward (by plotting -value).

    Individual data points for each mouse (for Increase and for Decrease) are also overlaid
    and connected with dashed lines (with black outlines on the markers).
    """
    # Construct the keys
    inc_key = f"{common_condition}_Increase"
    dec_key = f"{common_condition}_Decrease"

    if inc_key not in all_3set or dec_key not in all_3set:
        print(f"Data for {inc_key} or {dec_key} not found in all_3set.")
        return

    # Each key maps to a list of fraction dictionaries (one per mouse)
    inc_data = all_3set[inc_key]
    dec_data = all_3set[dec_key]

    # For each mouse, aggregate data for the two groups:
    # Reward Responsive: set2_only + set1_set2 + set2_set3 + all_three
    # Light Responsive:  set3_only + set1_set3 + set2_set3 + all_three
    reward_inc_list, light_inc_list = [], []
    for d in inc_data:
        reward_val = d.get("set2_only", 0) + d.get("set1_set2", 0) + d.get("set2_set3", 0) + d.get("all_three", 0)
        light_val = d.get("set3_only", 0) + d.get("set1_set3", 0) + d.get("set2_set3", 0) + d.get("all_three", 0)
        reward_inc_list.append(reward_val)
        light_inc_list.append(light_val)

    reward_dec_list, light_dec_list = [], []
    for d in dec_data:
        reward_val = d.get("set2_only", 0) + d.get("set1_set2", 0) + d.get("set2_set3", 0) + d.get("all_three", 0)
        light_val = d.get("set3_only", 0) + d.get("set1_set3", 0) + d.get("set2_set3", 0) + d.get("all_three", 0)
        reward_dec_list.append(reward_val)
        light_dec_list.append(light_val)

    # Compute means and SEMs for each aggregated group.
    reward_inc_mean, reward_inc_sem = mean_sem(reward_inc_list)
    light_inc_mean, light_inc_sem = mean_sem(light_inc_list)
    reward_dec_mean, reward_dec_sem = mean_sem(reward_dec_list)
    light_dec_mean, light_dec_sem = mean_sem(light_dec_list)

    # Set up the plot.
    x = np.arange(2)  # Two groups
    group_labels = ["Reward Responsive", "Light Responsive"]
    bar_width = 0.6
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot Increase bars (drawn upward).
    bars_inc = ax.bar(x, [reward_inc_mean, light_inc_mean],
                      width=bar_width, color='skyblue', edgecolor='black',
                      label='Increase')
    ax.errorbar(x, [reward_inc_mean, light_inc_mean],
                yerr=[reward_inc_sem, light_inc_sem],
                fmt='none', ecolor='black', capsize=3)

    # Plot Decrease bars (drawn downward by plotting -mean).
    bars_dec = ax.bar(x, [-reward_dec_mean, -light_dec_mean],
                      width=bar_width, color='salmon', edgecolor='black',
                      label='Decrease')
    ax.errorbar(x, [-reward_dec_mean, -light_dec_mean],
                yerr=[reward_dec_sem, light_dec_sem],
                fmt='none', ecolor='black', capsize=3)

    # Overlay individual data lines for Increase.
    # (Each mouse contributes a point for Reward and Light, which are connected by a dashed line.)
    for i in range(len(reward_inc_list)):
        ax.plot(x, [reward_inc_list[i], light_inc_list[i]],
                color='skyblue', marker='o', linestyle='--', linewidth=1,
                markerfacecolor='skyblue', markeredgecolor='black', markeredgewidth=1,
                alpha=0.7, zorder=3)

    # Overlay individual data lines for Decrease.
    for i in range(len(reward_dec_list)):
        ax.plot(x, [-reward_dec_list[i], -light_dec_list[i]],
                color='salmon', marker='o', linestyle='--', linewidth=1,
                markerfacecolor='salmon', markeredgecolor='black', markeredgewidth=1,
                alpha=0.7, zorder=3)

    # Add a horizontal line at zero.
    ax.axhline(0, color='black', linewidth=0.8)

    # Format the x-axis.
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, rotation=45, ha='right')

    # Optionally set y-ticks (here we show absolute percentages).
    yticks = np.arange(-50, 51, 10)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{abs(y)}%" for y in yticks])
    ax.set_ylabel("Percentage of total neurons")
    ax.set_title(f"{common_condition}")
    ax.legend()

    plt.tight_layout()
    plt.savefig(join(save_dir, f"{common_condition}_Reward_Light_Increase_Decrease.pdf"))
    plt.show()


def plot_venn_diagram_for_3set(all_3set, common_condition, save_dir="."):
    """
    For a given common condition, this function computes the mean fraction of cells
    (across mice) for each of the five segments in the 3-set analysis:
         "set1_only", "set1_set3", "all_three", "set1_set2", "set2_only"
    It then maps these values (rounded to one decimal place) onto a 3-set Venn diagram
    where the three sets are:
         CR (set1), FA (set3), Reward (set2)
    A complete 3-set Venn diagram has 7 regions; the two regions that are not in the
    data (FA only and FA & Reward only) are set to 0.

    The resulting figure shows two subplots side-by-side:
       - Left: Mean percentages for the Increase condition.
       - Right: Mean percentages for the Decrease condition (absolute values are used).

    The diagram uses the following mapping:
         - Only CR:      set1_only
         - Only FA:      0
         - CR & FA only: set1_set3
         - Only Reward:  set2_only
         - CR & Reward:  set1_set2
         - FA & Reward:  0
         - All three:    all_three
    """
    # Construct keys
    inc_key = f"{common_condition}_Increase"
    dec_key = f"{common_condition}_Decrease"

    if inc_key not in all_3set or dec_key not in all_3set:
        print(f"Data for {inc_key} or {dec_key} not found in all_3set.")
        return

    # Each key maps to a list of dictionaries (one per mouse)
    inc_data = all_3set[inc_key]
    dec_data = all_3set[dec_key]

    # Define the segments (keys in each mouse's fraction dictionary)
    segments = ["set1_only", "set1_set3", "all_three", "set1_set2", "set2_only"]

    # Compute mean across mice for each segment for Increase and Decrease.
    inc_means = {}
    dec_means = {}
    for seg in segments:
        inc_vals = [d.get(seg, 0) for d in inc_data]
        dec_vals = [d.get(seg, 0) for d in dec_data]
        inc_means[seg] = np.mean(inc_vals)
        dec_means[seg] = np.mean(dec_vals)

    # Map the 5 values to 7 regions of a Venn diagram for Increase.
    subsets_inc = (
        round(inc_means["set1_only"], 1),  # Only CR
        0,  # Only FA
        round(inc_means["set1_set3"], 1),  # CR & FA only
        round(inc_means["set2_only"], 1),  # Only Reward
        round(inc_means["set1_set2"], 1),  # CR & Reward only
        0,  # FA & Reward
        round(inc_means["all_three"], 1)  # All three
    )

    # For Decrease, we use absolute values and round them.
    subsets_dec = (
        round(abs(dec_means["set1_only"]), 1),
        0,
        round(abs(dec_means["set1_set3"]), 1),
        round(abs(dec_means["set2_only"]), 1),
        round(abs(dec_means["set1_set2"]), 1),
        0,
        round(abs(dec_means["all_three"]), 1)
    )

    # Create the subplots.
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot Increase Venn diagram.
    venn3(subsets=subsets_inc, set_labels=("CR", "FA", "Reward"), ax=axes[0])
    axes[0].set_title(f"{common_condition} - Increase")

    # Plot Decrease Venn diagram.
    venn3(subsets=subsets_dec, set_labels=("CR", "FA", "Reward"), ax=axes[1])
    axes[1].set_title(f"{common_condition} - Decrease")

    plt.tight_layout()
    plt.savefig(join(save_dir, f"{common_condition}_VennDiagram.pdf"))
    plt.show()


def plot_venn_diagram_light_vs_reward(all_3set, common_condition, save_dir="."):
    """
    For a given common condition, this function computes the mean fraction of cells
    (across mice) for each segment of the 3-set analysis:
         "set1_only", "set1_set3", "set1_set2", "set2_only", "all_three"
    It then combines the CR and FA sets into a single "Light" set, mapping:
         - Only Light:   set1_only + set1_set3  (cells that respond to light only)
         - Only Reward:  set2_only              (cells that respond to reward only)
         - Intersection: set1_set2 + all_three   (cells that respond to both)

    Two Venn diagrams are created side by side:
       - Left: Increase condition (values as is)
       - Right: Decrease condition (absolute values are used)

    All percentages are rounded to one decimal place.
    """
    # Construct keys.
    inc_key = f"{common_condition}_Increase"
    dec_key = f"{common_condition}_Decrease"

    if inc_key not in all_3set or dec_key not in all_3set:
        print(f"Data for {inc_key} or {dec_key} not found in all_3set.")
        return

    # Each key maps to a list of dictionaries (one per mouse)
    inc_data = all_3set[inc_key]
    dec_data = all_3set[dec_key]

    # Expected segments
    segments = ["set1_only", "set1_set3", "set1_set2", "set2_only", "all_three"]

    # Compute the means across mice for Increase and Decrease.
    inc_means = {}
    dec_means = {}
    for seg in segments:
        inc_vals = [d.get(seg, 0) for d in inc_data]
        dec_vals = [d.get(seg, 0) for d in dec_data]
        inc_means[seg] = np.mean(inc_vals)
        dec_means[seg] = np.mean(dec_vals)

    # Combine CR and FA into "Light" for Increase.
    light_only_inc = round(inc_means["set1_only"] + inc_means["set1_set3"], 1)
    reward_only_inc = round(inc_means["set2_only"], 1)
    intersection_inc = round(inc_means["set1_set2"] + inc_means["all_three"], 1)

    # For Decrease (use absolute values).
    light_only_dec = round(abs(dec_means["set1_only"] + dec_means["set1_set3"]), 1)
    reward_only_dec = round(abs(dec_means["set2_only"]), 1)
    intersection_dec = round(abs(dec_means["set1_set2"] + dec_means["all_three"]), 1)

    # Create two subplots for Increase and Decrease.
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Increase Venn diagram (2 sets: Light and Reward).
    # venn2 expects subsets as a tuple: (only A, only B, A∩B) , color=["cadetblue", "rosybrown"]
    venn2(subsets=(light_only_inc, reward_only_inc, intersection_inc),
          set_labels=("Light", "Reward"), ax=axes[0],set_colors=["lightsteelblue", "rosybrown"])
    axes[0].set_title(f"{common_condition} - Increase")

    # Decrease Venn diagram.
    venn2(subsets=(light_only_dec, reward_only_dec, intersection_dec),
          set_labels=("Light", "Reward"), ax=axes[1], set_colors=["lightsteelblue", "rosybrown"])
    axes[1].set_title(f"{common_condition} - Decrease")

    plt.tight_layout()
    plt.savefig(join(save_dir, f"{common_condition}_Light_vs_Reward_VennDiagram.pdf"))
    plt.show()

    # sum the values for increase and decrease to one venn diagram
    # Create a combined Venn diagram for both Increase and Decrease.
    combined_light_inc = light_only_inc + light_only_dec
    combined_reward_inc = reward_only_inc + reward_only_dec
    combined_intersection_inc = intersection_inc + intersection_dec
    # round to one decimal place
    combined_light_inc = round(combined_light_inc, 1)
    combined_reward_inc = round(combined_reward_inc, 1)
    combined_intersection_inc = round(combined_intersection_inc, 1)
    fig, ax = plt.subplots(figsize=(6, 6))
    venn2(subsets=(combined_light_inc, combined_reward_inc, combined_intersection_inc),
            set_labels=("Light", "Reward"), set_colors=["lightsteelblue", "rosybrown"], ax=ax)
    ax.set_title(f"{common_condition} - Combined Increase & Decrease")
    plt.tight_layout()
    plt.savefig(join(save_dir, f"{common_condition}_Combined_Light_vs_Reward_VennDiagram.pdf"))
    plt.show()


###############################################################################
# 6) Main
###############################################################################
if __name__ == "__main__":
    # One .pkl file per mouse (saved by cell_classification_auROC_plots.py)
    pkl_files = [
        DATA_ROOT + r"\mouse09\sess05\ML\data\processed_data\auROC_z_no_sub_history\comparison_fractions_240826.pkl",
        DATA_ROOT + r"\mouse10\sess07\ML\data\processed_data\auROC_z_no_sub_history\comparison_fractions_240829.pkl",
        DATA_ROOT + r"\mouse05\sess03\ML\data\processed_data\auROC_z_no_sub_history\comparison_fractions_240530.pkl",
        DATA_ROOT + r"\mouse01\sess01\ML\data\processed_data\auROC_z_no_sub_history\comparison_fractions_240519.pkl",
        DATA_ROOT + r"\mouse14\sess11\ML\data\processed_data\auROC_z_no_sub_history\comparison_fractions_250224.pkl",
        # DATA_ROOT + r"\mouse16\sess14\ML\data\processed_data\auROC_z_no_sub_history\comparison_fractions_250326.pkl"
    ]
    save_dir = RESULTS_ROOT + r"\new_fractions_v2"
    os.makedirs(save_dir, exist_ok=True)

    # 1) Load fraction data from all .pkl files
    all_2set, all_3set = load_fraction_data(pkl_files)

    # 2) Plot 2-set
    plot_stacked_bar_2set_percent(all_2set)
    #
    # # 3) Plot 3-set
    # plot_stacked_bar_3set_percent(all_3set)

    # Specify the common condition (without the _Increase or _Decrease suffix).
    common_condition = "CR(LightOn)_Hit(Reward)_FA(LightOn)"

    # Plot the diverging bar chart for this common condition.
    plot_increase_decrease_for_3set(all_3set, common_condition)

    common_condition = "CR(LightOn)_Hit(Reward)_FA(LightOn)"

    # Create the two-group diverging bar plot.
    plot_increase_decrease_two_groups(all_3set, common_condition)
    # plot_venn_diagram_for_3set(all_3set, common_condition, save_dir)
    plot_venn_diagram_light_vs_reward(all_3set, common_condition, save_dir)