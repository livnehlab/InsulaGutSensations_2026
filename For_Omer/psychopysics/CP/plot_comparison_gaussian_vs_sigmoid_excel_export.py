"""Gaussian versus Sigmoid CP: original plots plus Excel source data.

Run this file in PyCharm with access to Z:. Five mouse paths are copied from
the active Gaussian/Sigmoid configuration in plot_comparison_results_across_mice.py.
Gaussian is the first condition, Sigmoid the second. Mouse pairs are sorted by ID.

Outputs: Documents/CP_Gaussian_Sigmoid/run_<timestamp>/
  Gaussian_vs_Sigmoid_source_data.xlsx and eight original comparison PDFs.
Each threshold sheet has all individual mouse percentages plus plotted mean/SEM.
Statistics contains the original auto-selected, two-sided paired tests.
Sources records the actual input paths and SHA256. Inputs are never overwritten.
Sample SEM and all original statistical helper calculations are preserved.
Requires numpy, scipy, pandas, matplotlib and openpyxl.
"""
import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from scipy.stats import wilcoxon
from scipy.stats import ttest_rel
plt.rcParams["font.family"] = "Arial"
plt.rcParams["pdf.fonttype"] = 42
from scipy.stats import shapiro, ttest_rel, wilcoxon
from scipy.stats import shapiro, ttest_rel, wilcoxon
from scipy.stats import t as student_t
from scipy.stats import shapiro, levene, ttest_rel, ttest_ind, mannwhitneyu, wilcoxon, t as t_dist, iqr, rankdata, norm
from scipy.stats import t as student_t
def _mad_outliers(x, thresh=3.5):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if mad == 0:
        return 0
    z = 0.6745 * (x - med) / mad
    return int(np.sum(np.abs(z) > thresh))

def _ttest_rel_alt(a, b, alternative="two-sided"):
    # Uses SciPy's alternative arg if available, otherwise converts two-sided p
    try:
        return float(ttest_rel(a, b, alternative=alternative).pvalue)
    except TypeError:
        stat, p_two = ttest_rel(a, b)  # older SciPy, two-sided only
        df = len(a) - 1
        if alternative == "two-sided":
            return float(p_two)
        if alternative == "greater":   # H1: mean(b) > mean(a)
            return float(student_t.sf(stat, df))
        if alternative == "less":      # H1: mean(b) < mean(a)
            return float(student_t.cdf(stat, df))
        return float(p_two)

def _auto_paired_test(a, b, alternative="two-sided"):
    """
    Auto choose paired t vs Wilcoxon on paired diffs, with one- or two-tailed control.
    Returns (p_value, label, info_dict).
    """
    a = np.asarray(a, float); b = np.asarray(b, float)
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]; b = b[mask]
    n = a.size
    info = {"n": int(n), "shapiro_p": np.nan, "outliers": 0, "all_zero_diff": False, "alternative": alternative}
    if n < 2:
        return np.nan, "Wilcoxon", info

    d = b - a
    if np.allclose(d, 0.0, rtol=0, atol=0):
        info["all_zero_diff"] = True
        return 1.0, "no difference", info

    if 3 <= n <= 5000:
        try:
            info["shapiro_p"] = float(shapiro(d).pvalue)
        except Exception:
            info["shapiro_p"] = np.nan
    info["outliers"] = _mad_outliers(d)

    normal_enough = np.isfinite(info["shapiro_p"]) and info["shapiro_p"] >= 0.05
    no_extreme = info["outliers"] == 0

    if normal_enough and no_extreme:
        p = _ttest_rel_alt(a, b, alternative=alternative)
        return p, "paired t", info

    try:
        p = float(wilcoxon(a, b, zero_method="wilcox", alternative=alternative).pvalue)
    except ValueError:
        p = 1.0
    return p, "Wilcoxon", info

def _p_stars(p):
    return "****" if p < 1e-4 else "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "ns"

def _bracket(ax, x1, x2, y_base, span, p, stars):
    h = 0.06 * span
    y = y_base + 0.03 * span
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c="black")
    ax.text((x1 + x2) / 2.0, y + h + 0.01 * span, f"{stars}\n p = {p:.3g}",
            ha="center", va="bottom")
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(top=max(ymax, y + h + 0.1 * span))
def _p_stars(p: float) -> str:
    return "****" if p < 1e-4 else "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "ns"

def _mad_outliers(x, thresh=3.5):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if mad == 0:
        return 0
    z = 0.6745 * (x - med) / mad
    return int(np.sum(np.abs(z) > thresh))

def _bracket(ax, x1, x2, y_base, span, p, stars):
    """Draw a bracket between x1 and x2 just above y_base."""
    h = 0.06 * span
    y = y_base + 0.03 * span
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c="black")
    ax.text((x1 + x2) / 2.0, y + h + 0.01 * span, f"{stars}\n p = {p:.3g}",
            ha="center", va="bottom")
    # make sure it fits
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(top=max(ymax, y + h + 0.1 * span))
def load_comparison_results(mice_data_paths):
    """
    Load comparison_results.pkl files from multiple mice.
    
    Parameters:
    -----------
    mice_data_paths : dict
        Dictionary with mouse names as keys and paths to their data folders as values
        e.g., {'OI124': 'Z:/2p_data/OI124/240610/ML/data/processed_data/auROC_freq_history_control_CR-CR_rep1/test_all_reps'}
    
    Returns:
    --------
    dict : Dictionary with mouse names as keys and their comparison results as values
    """
    results = {}
    
    for mouse_name, data_path in mice_data_paths.items():
        comparison_file = os.path.join(data_path, 'comparison_results.pkl')
        
        if os.path.exists(comparison_file):
            try:
                with open(comparison_file, 'rb') as f:
                    results[mouse_name] = pickle.load(f)
                print(f"Loaded data for {mouse_name}")
            except Exception as e:
                print(f"Error loading data for {mouse_name}: {e}")
        else:
            print(f"Comparison results file not found for {mouse_name} at {comparison_file}")
    
    return results

def load_comparison_results_two_days(mice_data_paths_day1, mice_data_paths_day2, day1_name="Detection", day2_name="Psychophysics"):
    """
    Load comparison_results.pkl files from two different days for multiple mice.
    
    Parameters:
    -----------
    mice_data_paths_day1 : dict
        Dictionary with mouse names as keys and paths to their day1 data folders
    mice_data_paths_day2 : dict
        Dictionary with mouse names as keys and paths to their day2 data folders
    day1_name : str
        Name for day1 (e.g., "Detection")
    day2_name : str
        Name for day2 (e.g., "Psychophysics")
    
    Returns:
    --------
    dict : Dictionary with structure {day1_name: results_day1, day2_name: results_day2}
    """
    results = {}
    
    # Load day1 data
    print(f"Loading {day1_name} day data...")
    results[day1_name] = load_comparison_results(mice_data_paths_day1)
    
    # Load day2 data
    print(f"Loading {day2_name} day data...")
    results[day2_name] = load_comparison_results(mice_data_paths_day2)
    
    return results

def create_day_comparison_plots(comparison_results_two_days, save_loc=None):
    """
    Create comparison plots between two days for each percentile threshold.
    
    Parameters:
    -----------
    comparison_results_two_days : dict
        Dictionary with day names as keys and comparison results as values
    save_loc : str, optional
        Path to save the plots
    """
    if not comparison_results_two_days or len(comparison_results_two_days) != 2:
        print("Need exactly two days of data for comparison")
        return
    
    day_names = list(comparison_results_two_days.keys())
    day1_name, day2_name = day_names[0], day_names[1]
    
    # Get common mice between both days
    mice_day1 = set(comparison_results_two_days[day1_name].keys())
    mice_day2 = set(comparison_results_two_days[day2_name].keys())
    common_mice = sorted(list(mice_day1.intersection(mice_day2)))

    # combine the percentage of the two sides (5 +95, 1 + 99, etc.)


    
    if not common_mice:
        print("No common mice found between the two days")
        return
    
    print(f"Creating comparison plots for {len(common_mice)} mice: {common_mice}")
    
    # Define percentile pairs to compare
    percentile_pairs = [
        ('99.5th-0.5th', 'pct_exc_above_99_5th', 'pct_inh_below_0_5th'),
        ('99th-1st', 'pct_exc_above_99th', 'pct_inh_below_1st'),
        ('97.5th-2.5th', 'pct_exc_above_97_5th', 'pct_inh_below_2_5th'),
        ('95th-5th', 'pct_exc_above_95th', 'pct_inh_below_5th')
    ]
    
    # Create one figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, (percentile_name, exc_key, inh_key) in enumerate(percentile_pairs):
        ax = axes[idx]
        
        # Extract data for both days
        exc_day1 = []
        inh_day1 = []
        exc_day2 = []
        inh_day2 = []
        combined_day1 = []
        combined_day2 = []
        
        for mouse in common_mice:
            # Day 1 data
            pct_data_day1 = comparison_results_two_days[day1_name][mouse]['percentages_passing_thresholds']
            exc_day1.append(pct_data_day1[exc_key])
            inh_day1.append(pct_data_day1[inh_key])
            combined_day1.append(pct_data_day1[exc_key] + pct_data_day1[inh_key])
            
            # Day 2 data
            pct_data_day2 = comparison_results_two_days[day2_name][mouse]['percentages_passing_thresholds']
            exc_day2.append(pct_data_day2[exc_key])
            inh_day2.append(pct_data_day2[inh_key])
            combined_day2.append(pct_data_day2[exc_key] + pct_data_day2[inh_key])
        
        # Convert to numpy arrays
        exc_day1 = np.array(exc_day1)
        inh_day1 = np.array(inh_day1)
        exc_day2 = np.array(exc_day2)
        inh_day2 = np.array(inh_day2)
        combined_day1 = np.array(combined_day1)
        combined_day2 = np.array(combined_day2)
        
        # Calculate means and SEMs
        mean_exc_day1 = np.mean(exc_day1)
        sem_exc_day1 = np.std(exc_day1) / np.sqrt(len(exc_day1))
        mean_inh_day1 = np.mean(inh_day1)
        sem_inh_day1 = np.std(inh_day1) / np.sqrt(len(inh_day1))
        mean_combined_day1 = np.mean(combined_day1)
        sem_combined_day1 = np.std(combined_day1) / np.sqrt(len(combined_day1))
        
        mean_exc_day2 = np.mean(exc_day2)
        sem_exc_day2 = np.std(exc_day2) / np.sqrt(len(exc_day2))
        mean_inh_day2 = np.mean(inh_day2)
        sem_inh_day2 = np.std(inh_day2) / np.sqrt(len(inh_day2))
        mean_combined_day2 = np.mean(combined_day2)
        sem_combined_day2 = np.std(combined_day2) / np.sqrt(len(combined_day2))
        
        # Set up x positions - now with 4 positions: exc_day1, exc_day2, inh_day1, inh_day2
        x_pos = np.arange(4)
        width = 0.6
        
        # Create bars - excitatory neurons first (positions 0, 1), then inhibitory (positions 2, 3)
        bars_exc_day1 = ax.bar(x_pos[0], mean_exc_day1, width, yerr=sem_exc_day1, capsize=5, 
                              label=f'Positive {day1_name}', color='red', alpha=0.7)
        bars_exc_day2 = ax.bar(x_pos[1], mean_exc_day2, width, yerr=sem_exc_day2, capsize=5, 
                              label=f'Positive {day2_name}', color='darkred', alpha=0.7)
        bars_inh_day1 = ax.bar(x_pos[2], mean_inh_day1, width, yerr=sem_inh_day1, capsize=5, 
                              label=f'Negative {day1_name}', color='blue', alpha=0.7)
        bars_inh_day2 = ax.bar(x_pos[3], mean_inh_day2, width, yerr=sem_inh_day2, capsize=5, 
                              label=f'Negative {day2_name}', color='darkblue', alpha=0.7)
        
        # Add individual data points with connecting lines
        for i, mouse in enumerate(common_mice):
            # Excitatory neurons - connecting lines between days
            ax.plot([x_pos[0], x_pos[1]], [exc_day1[i], exc_day2[i]], 
                   color='darkred', alpha=0.4, linewidth=1)
            ax.scatter([x_pos[0], x_pos[1]], [exc_day1[i], exc_day2[i]], 
                      color='darkred', alpha=0.6, s=30, zorder=5)
            
            # Inhibitory neurons - connecting lines between days
            ax.plot([x_pos[2], x_pos[3]], [inh_day1[i], inh_day2[i]], 
                   color='darkblue', alpha=0.4, linewidth=1)
            ax.scatter([x_pos[2], x_pos[3]], [inh_day1[i], inh_day2[i]], 
                      color='darkblue', alpha=0.6, s=30, zorder=5)
        
        # Customize plot
        ax.set_xlabel('Neuron Type and Day')
        ax.set_ylabel('Percentage of neurons')
        ax.set_title(f'{percentile_name} Percentile Thresholds\n{day1_name} vs {day2_name}')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'Pos\n{day1_name}', f'Pos\n{day2_name}',
                           f'Neg\n{day1_name}', f'Neg\n{day2_name}'], rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars - fix the bar access
        bar_containers = [bars_exc_day1, bars_exc_day2, bars_inh_day1, bars_inh_day2]
        means = [mean_exc_day1, mean_exc_day2, mean_inh_day1, mean_inh_day2]
        sems = [sem_exc_day1, sem_exc_day2, sem_inh_day1, sem_inh_day2]
        
        for bar_container, mean_val, sem_val in zip(bar_containers, means, sems):
            bar = bar_container[0]  # Get the first (and only) bar from the container
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + sem_val + 0.5,
                    f'{mean_val:.1f}±{sem_val:.1f}%', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()

    # creat another plot for the combined
    fig_combined, ax_combined = plt.subplots(1, 1, figsize=(8, 6))
    combined_means = [mean_combined_day1, mean_combined_day2]
    combined_sems = [sem_combined_day1, sem_combined_day2]
    combined_x_pos = np.arange(2)
    combined_width = 0.6
    combined_bars = ax_combined.bar(combined_x_pos, combined_means, combined_width,
                                    yerr=combined_sems, capsize=5,
                                    color=['gray', 'black'], alpha=0.7)
    # add mean and sem values on top of bars
    for i, (mean_val, sem_val) in enumerate(zip(combined_means, combined_sems)):
        ax_combined.text(combined_x_pos[i], mean_val + sem_val + 0.5,
                         f'{mean_val:.1f}±{sem_val:.1f}%', ha='center', va='bottom', fontsize=12)

    ax_combined.set_xlabel('Combined Neurons')
    ax_combined.set_ylabel('Percentage of neurons')
    ax_combined.set_title(f'Combined Neurons Passing Percentile Thresholds\n{day1_name} vs {day2_name}')
    ax_combined.set_xticks(combined_x_pos)
    ax_combined.set_xticklabels([f'Combined\n{day1_name}', f'Combined\n{day2_name}'])
    ax_combined.grid(True, alpha=0.3)
    # Add value labels on combined bars
    for bar in combined_bars:
        height = bar.get_height()
        ax_combined.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                         f'{height:.1f}%', ha='center', va='bottom', fontsize=12)
        ax_combined.legend(['Combined Neurons'])
    plt.tight_layout()
    if save_loc is None:
        save_loc = Path(os.getcwd(), 'comparison_plots')
    if not os.path.exists(save_loc):
        os.makedirs(save_loc)
    # Save the combined plot
    fig_combined.savefig(os.path.join(save_loc, f'combined_neurons_passing_thresholds_{day1_name}_vs_{day2_name}.pdf'),
                        dpi=300, bbox_inches='tight')
    print(f"Combined neurons passing thresholds plot saved to {save_loc}")



    
    if save_loc:
        plt.savefig(os.path.join(save_loc, f'day_comparison_{day1_name}_vs_{day2_name}.pdf'),
                   dpi=300, bbox_inches='tight')
        print(f"Day comparison plots saved to {save_loc}")
    
    plt.close()
def _mad_outliers(x, thresh=3.5):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if mad == 0:
        return 0
    z = 0.6745 * (x - med) / mad
    return int(np.sum(np.abs(z) > thresh))

def _ttest_rel_alt(a, b, alternative="two-sided"):
    try:
        stat, p = ttest_rel(a, b, alternative=alternative)
        return float(p)
    except TypeError:
        stat, p_two = ttest_rel(a, b)
        df = len(a) - 1
        if alternative == "two-sided":
            return float(p_two)
        elif alternative == "greater":  # H1: mean(b) > mean(a)
            return float(student_t.sf(stat, df))
        elif alternative == "less":     # H1: mean(b) < mean(a)
            return float(student_t.cdf(stat, df))
        else:
            return float(p_two)

def _ttest_ind_alt(a, b, equal_var, alternative="two-sided"):
    try:
        stat, p = ttest_ind(a, b, equal_var=equal_var, alternative=alternative)
        return float(p)
    except TypeError:
        stat, p_two = ttest_ind(a, b, equal_var=equal_var)
        # Need df only for a proper one-sided conversion. Use Welch df if equal_var=False is messy,
        # so fall back to two-sided if ancient SciPy and one-sided requested.
        if alternative == "two-sided":
            return float(p_two)
        return float(p_two)  # safe fallback


def _ci_mean_diff(a, b, paired=True, equal_var=True, alpha=0.05):
    a = np.asarray(a); b = np.asarray(b)
    if paired:
        d = b - a
        n = d.size
        md = float(np.mean(d))
        sd = float(np.std(d, ddof=1))
        se = sd / np.sqrt(n)
        df = n - 1
        tcrit = float(t_dist.ppf(1 - alpha / 2, df))
        ci_low, ci_high = md - tcrit * se, md + tcrit * se
        return {"parameter": "mean difference", "level": 1 - alpha, "low": ci_low, "high": ci_high, "df": df}
    else:
        n1, n2 = a.size, b.size
        m1, m2 = float(np.mean(a)), float(np.mean(b))
        v1, v2 = float(np.var(a, ddof=1)), float(np.var(b, ddof=1))
        md = m2 - m1
        if equal_var:
            sp2 = ((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)
            se = np.sqrt(sp2 * (1 / n1 + 1 / n2))
            df = n1 + n2 - 2
        else:
            se = np.sqrt(v1 / n1 + v2 / n2)
            df = (v1 / n1 + v2 / n2) ** 2 / ((v1 ** 2) / (n1 ** 2 * (n1 - 1)) + (v2 ** 2) / (n2 ** 2 * (n2 - 1)))
        tcrit = float(t_dist.ppf(1 - alpha / 2, df))
        ci_low, ci_high = md - tcrit * se, md + tcrit * se
        return {"parameter": "mean difference", "level": 1 - alpha, "low": ci_low, "high": ci_high, "df": float(df)}

def _cohens_d_paired(a, b):
    d = np.asarray(b) - np.asarray(a)
    md = np.mean(d)
    sd = np.std(d, ddof=1)
    return float(md / sd) if sd > 0 else np.nan  # Cohen's dz

def _cohens_d_ind(a, b, equal_var=True):
    a = np.asarray(a); b = np.asarray(b)
    m1, m2 = float(np.mean(a)), float(np.mean(b))
    v1, v2 = float(np.var(a, ddof=1)), float(np.var(b, ddof=1))
    n1, n2 = a.size, b.size
    if equal_var:
        sp = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
        d = (m2 - m1) / sp if sp > 0 else np.nan
        df = n1 + n2 - 2
    else:
        s_av = np.sqrt((v1 + v2) / 2.0)
        d = (m2 - m1) / s_av if s_av > 0 else np.nan
        # For Welch, df via Satterthwaite
        df = (v1 / n1 + v2 / n2) ** 2 / ((v1 ** 2) / (n1 ** 2 * (n1 - 1)) + (v2 ** 2) / (n2 ** 2 * (n2 - 1)))
    # Hedges g correction for small samples
    if np.isfinite(df) and df > 1:
        J = 1 - 3 / (4 * df - 1)
        g = d * J
    else:
        g = d
    return float(g)

def _wilcoxon_z_from_W(W, n, continuity=True):
    # Null mean and variance for Wilcoxon signed-rank (no ties assumption)
    mu = n * (n + 1) / 4.0
    sigma = np.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    cc = 0.5 if continuity else 0.0
    # Two-sided uses absolute deviation for z magnitude
    z = (W - mu - np.sign(W - mu) * cc) / sigma if sigma > 0 else np.nan
    return float(z)

def _mwu_effect_sizes(U, n1, n2):
    # r using normal approximation
    mu = n1 * n2 / 2.0
    sigma = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
    z = (U - mu) / sigma if sigma > 0 else np.nan
    r = float(z / np.sqrt(n1 + n2))
    # Rank-biserial correlation
    r_rb = 1.0 - 2.0 * U / (n1 * n2)
    return float(z), float(r), float(r_rb)

def _descriptives_t(a, b, paired):
    if paired:
        d = b - a
        return {
            "n": int(a.size),
            "mean_a": float(np.mean(a)), "sd_a": float(np.std(a, ddof=1)),
            "mean_b": float(np.mean(b)), "sd_b": float(np.std(b, ddof=1)),
            "mean_diff": float(np.mean(d)), "sd_diff": float(np.std(d, ddof=1))
        }
    else:
        return {
            "n_a": int(a.size), "mean_a": float(np.mean(a)), "sd_a": float(np.std(a, ddof=1)),
            "n_b": int(b.size), "mean_b": float(np.mean(b)), "sd_b": float(np.std(b, ddof=1))
        }

def _descriptives_np(a, b, paired):
    if paired:
        d = b - a
        return {
            "n": int(a.size),
            "median_a": float(np.median(a)), "IQR_a": float(iqr(a, nan_policy="omit")),
            "median_b": float(np.median(b)), "IQR_b": float(iqr(b, nan_policy="omit")),
            "median_diff": float(np.median(d)), "IQR_diff": float(iqr(d, nan_policy="omit"))
        }
    else:
        return {
            "n_a": int(a.size), "median_a": float(np.median(a)), "IQR_a": float(iqr(a, nan_policy="omit")),
            "n_b": int(b.size), "median_b": float(np.median(b)), "IQR_b": float(iqr(b, nan_policy="omit"))
        }

def choose_test(a, b, paired=True, mode="auto", alternative="two-sided"):
    """
    Decide the appropriate test and return (p_value, label, info dict).
    paired=True -> paired t vs Wilcoxon using checks on differences.
    paired=False -> indep t vs Welch vs Mann-Whitney using Shapiro + Levene.
    alternative: "two-sided" (default), "greater", or "less".
    info["report"] contains journal-ready details: statistic, df, effect size, CI, and descriptives.
    """
    a = np.asarray(a); b = np.asarray(b)
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]; b = b[mask]
    n = a.size

    info = {"paired": paired, "n": int(n),
            "shapiro_p": np.nan, "shapiro_p_a": np.nan, "shapiro_p_b": np.nan,
            "levene_p": np.nan, "outliers": 0, "outliers_a": 0, "outliers_b": 0,
            "all_zero_diff": False, "alternative": alternative}

    # Default empty report
    info["report"] = {
        "test": None,
        "statistic_name": None,
        "statistic_value": np.nan,
        "df": None,
        "p": np.nan,
        "alternative": alternative,
        "n1": None, "n2": None,
        "effect_size": {"name": None, "value": np.nan},
        "ci": None,
        "descriptives": None
    }

    if n < 2:
        label = "Wilcoxon" if paired else "Mann-Whitney U"
        info["report"]["test"] = label
        return np.nan, label, info

    if paired:
        d = b - a
        if np.allclose(d, 0.0, rtol=0, atol=0):
            info["all_zero_diff"] = True
            info["report"]["test"] = "no difference"
            info["report"]["descriptives"] = _descriptives_t(a, b, paired=True)
            return 1.0, "no difference", info

        if 3 <= n <= 5000:
            try:
                info["shapiro_p"] = float(shapiro(d).pvalue)
            except Exception:
                pass
        info["outliers"] = _mad_outliers(d)

        normal_enough = np.isfinite(info["shapiro_p"]) and info["shapiro_p"] >= 0.05
        no_extreme = info["outliers"] == 0

        if mode == "ttest" or (mode == "auto" and normal_enough and no_extreme):
            # Paired t-test
            res = ttest_rel(a, b, alternative=alternative)
            tval, p = float(res.statistic), float(res.pvalue)
            df = n - 1
            label = "paired t"
            # Effect size dz and CI for mean difference
            dz = _cohens_d_paired(a, b)
            ci = _ci_mean_diff(a, b, paired=True, alpha=0.05)
            info["report"] = {
                "test": label,
                "statistic_name": "t",
                "statistic_value": tval,
                "df": df,
                "p": p,
                "alternative": alternative,
                "n1": n, "n2": n,
                "effect_size": {"name": "Cohen's dz", "value": dz},
                "ci": ci,
                "descriptives": _descriptives_t(a, b, paired=True)
            }
            return p, label, info

        # Wilcoxon signed-rank
        try:
            wr = wilcoxon(a, b, zero_method="wilcox", alternative=alternative, correction=False, mode="auto")
            W, p = float(wr.statistic), float(wr.pvalue)
        except ValueError:
            W, p = np.nan, 1.0
        z = _wilcoxon_z_from_W(W, n, continuity=True) if np.isfinite(W) else np.nan
        r = float(z / np.sqrt(n)) if np.isfinite(z) else np.nan
        label = "Wilcoxon"
        info["report"] = {
            "test": label,
            "statistic_name": "W",
            "statistic_value": W,
            "df": None,
            "p": p,
            "alternative": alternative,
            "n1": n, "n2": n,
            "effect_size": {"name": "r = z/√N", "value": r},
            "ci": None,  # CI for median shift not computed here
            "descriptives": _descriptives_np(a, b, paired=True)
        }
        return p, label, info

    # Independent mode
    a_clean = a[np.isfinite(a)]
    b_clean = b[np.isfinite(b)]
    if 3 <= a_clean.size <= 5000:
        try:
            info["shapiro_p_a"] = float(shapiro(a_clean).pvalue)
        except Exception:
            pass
    if 3 <= b_clean.size <= 5000:
        try:
            info["shapiro_p_b"] = float(shapiro(b_clean).pvalue)
        except Exception:
            pass
    info["outliers_a"] = _mad_outliers(a_clean)
    info["outliers_b"] = _mad_outliers(b_clean)
    try:
        info["levene_p"] = float(levene(a_clean, b_clean, center="median").pvalue)
    except Exception:
        pass

    normal_a = np.isfinite(info["shapiro_p_a"]) and info["shapiro_p_a"] >= 0.05
    normal_b = np.isfinite(info["shapiro_p_b"]) and info["shapiro_p_b"] >= 0.05
    no_extreme = (info["outliers_a"] == 0 and info["outliers_b"] == 0)
    equal_var = np.isfinite(info["levene_p"]) and info["levene_p"] >= 0.05

    if mode == "ttest" or (mode == "auto" and normal_a and normal_b and no_extreme):
        # Independent or Welch t
        res = ttest_ind(a_clean, b_clean, equal_var=bool(equal_var), alternative=alternative)
        tval, p = float(res.statistic), float(res.pvalue)
        if equal_var:
            df = a_clean.size + b_clean.size - 2
            label = "indep t"
        else:
            # Welch df
            v1 = float(np.var(a_clean, ddof=1)); v2 = float(np.var(b_clean, ddof=1))
            n1, n2 = a_clean.size, b_clean.size
            df = (v1 / n1 + v2 / n2) ** 2 / ((v1 ** 2) / (n1 ** 2 * (n1 - 1)) + (v2 ** 2) / (n2 ** 2 * (n2 - 1)))
            label = "Welch t"
        g = _cohens_d_ind(a_clean, b_clean, equal_var=bool(equal_var))
        ci = _ci_mean_diff(a_clean, b_clean, paired=False, equal_var=bool(equal_var), alpha=0.05)
        info["report"] = {
            "test": label,
            "statistic_name": "t",
            "statistic_value": tval,
            "df": float(df),
            "p": p,
            "alternative": alternative,
            "n1": int(a_clean.size), "n2": int(b_clean.size),
            "effect_size": {"name": "Hedges g", "value": g},
            "ci": ci,
            "descriptives": _descriptives_t(a_clean, b_clean, paired=False)
        }
        return p, label, info

    # Mann–Whitney U
    mwu = mannwhitneyu(a_clean, b_clean, alternative=alternative)
    U, p = float(mwu.statistic), float(mwu.pvalue)
    z, r, r_rb = _mwu_effect_sizes(U, a_clean.size, b_clean.size)
    label = "Mann-Whitney U"
    info["report"] = {
        "test": label,
        "statistic_name": "U",
        "statistic_value": U,
        "df": None,
        "p": p,
        "alternative": alternative,
        "n1": int(a_clean.size), "n2": int(b_clean.size),
        "effect_size": {"name": "rank-biserial r", "value": r_rb},
        "ci": None,  # CI for r not computed here
        "descriptives": _descriptives_np(a_clean, b_clean, paired=False)
    }
    return p, label, info

def create_individual_percentile_comparison(comparison_results_two_days, percentile_name, exc_key, inh_key,
                                            save_loc=None, day1_name="Detection", day2_name="Psychophysics",
                                            alternative="two-sided"):
    if not comparison_results_two_days or len(comparison_results_two_days) != 2:
        print("Need exactly two days of data for comparison")
        return

    day_names = list(comparison_results_two_days.keys())
    day1_name, day2_name = day_names[0], day_names[1]

    # common mice
    mice_day1 = set(comparison_results_two_days[day1_name].keys())
    mice_day2 = set(comparison_results_two_days[day2_name].keys())
    common_mice = sorted(list(mice_day1.intersection(mice_day2)))
    if not common_mice:
        print("No common mice found between the two days")
        return

    # gather data
    exc_day1, inh_day1, exc_day2, inh_day2 = [], [], [], []
    combined_day1, combined_day2 = [], []
    for mouse in common_mice:
        p1 = comparison_results_two_days[day1_name][mouse]['percentages_passing_thresholds']
        p2 = comparison_results_two_days[day2_name][mouse]['percentages_passing_thresholds']
        exc_day1.append(p1[exc_key]);
        inh_day1.append(p1[inh_key])
        exc_day2.append(p2[exc_key]);
        inh_day2.append(p2[inh_key])
        combined_day1.append(p1[exc_key] + p1[inh_key])
        combined_day2.append(p2[exc_key] + p2[inh_key])

    exc_day1 = np.asarray(exc_day1, float)
    inh_day1 = np.asarray(inh_day1, float)
    exc_day2 = np.asarray(exc_day2, float)
    inh_day2 = np.asarray(inh_day2, float)
    combined_day1 = np.asarray(combined_day1, float)
    combined_day2 = np.asarray(combined_day2, float)

    # means and SEMs
    def _mean_sem(x):
        x = np.asarray(x, float)
        return np.nanmean(x), np.nanstd(x, ddof=1) / np.sqrt(np.sum(np.isfinite(x)))

    mean_exc_day1, sem_exc_day1 = _mean_sem(exc_day1)
    mean_exc_day2, sem_exc_day2 = _mean_sem(exc_day2)
    mean_inh_day1, sem_inh_day1 = _mean_sem(inh_day1)
    mean_inh_day2, sem_inh_day2 = _mean_sem(inh_day2)
    mean_combined_day1, sem_combined_day1 = _mean_sem(combined_day1)
    mean_combined_day2, sem_combined_day2 = _mean_sem(combined_day2)

    # Export the exact arrays and summaries used by the plots below.
    if save_loc:
        export_plot_values(save_loc, percentile_name, common_mice,
                           day1_name, day2_name, exc_day1, inh_day1,
                           exc_day2, inh_day2, combined_day1, combined_day2,
                           _mean_sem)


    # ---------- Figure 1: Exc and Inh with auto test and brackets ----------
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    x_pos = np.arange(4)
    width = 0.6

    bars_exc_day1 = ax.bar(x_pos[0], mean_exc_day1, width, yerr=sem_exc_day1, capsize=5,
                           label=f'Positive {day1_name}', color='red', alpha=0.7)
    bars_exc_day2 = ax.bar(x_pos[1], mean_exc_day2, width, yerr=sem_exc_day2, capsize=5,
                           label=f'Positive {day2_name}', color='darkred', alpha=0.7)
    bars_inh_day1 = ax.bar(x_pos[2], mean_inh_day1, width, yerr=sem_inh_day1, capsize=5,
                           label=f'Negative {day1_name}', color='blue', alpha=0.7)
    bars_inh_day2 = ax.bar(x_pos[3], mean_inh_day2, width, yerr=sem_inh_day2, capsize=5,
                           label=f'Negative {day2_name}', color='darkblue', alpha=0.7)

    # individual points + connections
    for i, mouse in enumerate(common_mice):
        ax.plot([x_pos[0], x_pos[1]], [exc_day1[i], exc_day2[i]],
                color='darkred', alpha=0.4, linewidth=1.5)
        ax.scatter([x_pos[0], x_pos[1]], [exc_day1[i], exc_day2[i]],
                   color='darkred', alpha=0.6, s=40, zorder=5)
        ax.plot([x_pos[2], x_pos[3]], [inh_day1[i], inh_day2[i]],
                color='darkblue', alpha=0.4, linewidth=1.5)
        ax.scatter([x_pos[2], x_pos[3]], [inh_day1[i], inh_day2[i]],
                   color='darkblue', alpha=0.6, s=40, zorder=5)

    ax.set_ylabel('Percentage of neurons')
    ax.set_title(f'{percentile_name} Percentile Thresholds\n{day1_name} vs {day2_name} (n={len(common_mice)} mice)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'Positive\n{day1_name}', f'Positive\n{day2_name}',
                        f'Negative\n{day1_name}', f'Negative\n{day2_name}'])
    ax.grid(True, alpha=0.3)

    # value labels
    bar_containers = [bars_exc_day1, bars_exc_day2, bars_inh_day1, bars_inh_day2]
    means = [mean_exc_day1, mean_exc_day2, mean_inh_day1, mean_inh_day2]
    sems = [sem_exc_day1, sem_exc_day2, sem_inh_day1, sem_inh_day2]
    for bar_container, mean_val, sem_val in zip(bar_containers, means, sems):
        bar = bar_container[0]
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., h + sem_val + 0.5,
                f'{mean_val:.1f}±{sem_val:.1f}%', ha='center', va='bottom', fontsize=12)

    # auto tests and brackets with chosen tail
    p_exc, test_exc, info_exc = choose_test(exc_day1, exc_day2, paired=True, mode="auto")
    p_inh, test_inh, info_inh = choose_test(inh_day1, inh_day2, paired=True, mode="auto")
    print(f"\nPositive neurons: {test_exc} p={p_exc:.4f}, info: {info_exc}")
    print(f"Negative neurons: {test_inh} p={p_inh:.4f}, info: {info_inh}")

    ymin, ymax = ax.get_ylim()
    span = ymax - ymin

    y_exc_base = np.nanmax([
        mean_exc_day1 + sem_exc_day1, mean_exc_day2 + sem_exc_day2,
        np.nanmax(exc_day1) if exc_day1.size else 0,
        np.nanmax(exc_day2) if exc_day2.size else 0
    ])
    y_inh_base = np.nanmax([
        mean_inh_day1 + sem_inh_day1, mean_inh_day2 + sem_inh_day2,
        np.nanmax(inh_day1) if inh_day1.size else 0,
        np.nanmax(inh_day2) if inh_day2.size else 0
    ])

    _bracket(ax, x_pos[0], x_pos[1], y_exc_base, span, p_exc, _p_stars(p_exc))
    _bracket(ax, x_pos[2], x_pos[3], y_inh_base, span, p_inh, _p_stars(p_inh))

    plt.tight_layout()
    if save_loc:
        safe_percentile_name = percentile_name.replace(' ', '_').replace('-', '_')
        fig.savefig(os.path.join(save_loc, f'comparison_{safe_percentile_name}_{day1_name}_vs_{day2_name}.pdf'),
                    dpi=300, bbox_inches='tight')
        print(f"{percentile_name} comparison plot saved to {save_loc}")
    plt.close()

    # ---------- Figure 2: Combined with auto test and bracket ----------
    fig_combined, ax_combined = plt.subplots(1, 1, figsize=(8, 6))
    combined_means = [mean_combined_day1, mean_combined_day2]
    combined_sems = [sem_combined_day1, sem_combined_day2]

    group_x = np.arange(1)
    bar_width = 0.2
    gap = 0.1

    bars_combined_day1 = ax_combined.bar(
        group_x - (bar_width / 2 + gap / 2), mean_combined_day1, bar_width,
        yerr=sem_combined_day1, capsize=5,
        label=f'Combined {day1_name}', color='gray', alpha=0.7
    )
    bars_combined_day2 = ax_combined.bar(
        group_x + (bar_width / 2 + gap / 2), mean_combined_day2, bar_width,
        yerr=sem_combined_day2, capsize=5,
        label=f'Combined {day2_name}', color='black', alpha=0.7
    )

    for i, mouse in enumerate(common_mice):
        ax_combined.plot(
            [group_x - (bar_width / 2 + gap / 2), group_x + (bar_width / 2 + gap / 2)],
            [combined_day1[i], combined_day2[i]],
            color='gray', alpha=0.4, linewidth=1.5
        )
        ax_combined.scatter(
            [group_x - (bar_width / 2 + gap / 2), group_x + (bar_width / 2 + gap / 2)],
            [combined_day1[i], combined_day2[i]],
            color='black', alpha=0.6, s=40, zorder=5
        )
    # value labels
    combined_bars = [bars_combined_day1, bars_combined_day2]
    combined_x_pos = [bc[0].get_x() + bc[0].get_width() / 2. for bc in combined_bars]
    for i, (mean_val, sem_val) in enumerate(zip(combined_means, combined_sems)):
        ax_combined.text(combined_x_pos[i], mean_val + sem_val + 0.5,
                         f'{mean_val:.1f}±{sem_val:.1f}%', ha='center', va='bottom', fontsize=12)

    p_comb, test_comb, info_comb = choose_test(combined_day1, combined_day2, paired=True, mode="auto")
    print(f"\nCombined neurons: {test_comb} p={p_comb:.4f}, info: {info_comb}")

    ymin2, ymax2 = ax_combined.get_ylim()
    span2 = ymax2 - ymin2
    y_base_comb = np.nanmax([
        mean_combined_day1 + sem_combined_day1, mean_combined_day2 + sem_combined_day2,
        np.nanmax(combined_day1) if combined_day1.size else 0,
        np.nanmax(combined_day2) if combined_day2.size else 0
    ])
    x_left = (group_x - (bar_width / 2 + gap / 2))[0]
    x_right = (group_x + (bar_width / 2 + gap / 2))[0]
    _bracket(ax_combined, x_left, x_right, y_base_comb, span2, p_comb, _p_stars(p_comb))

    ax_combined.set_xlabel('Combined Neurons')
    ax_combined.set_ylabel('Percentage of neurons')
    ax_combined.set_title(
        f'Combined Neurons Passing {percentile_name} Percentile Thresholds\n'
        f'{day1_name} vs {day2_name}'
    )
    ax_combined.set_xticks(group_x)
    ax_combined.set_xticklabels([f'Combined\n{day1_name} vs {day2_name}'])
    ax_combined.grid(True, alpha=0.3)
    ax_combined.legend()
    plt.tight_layout()

    if save_loc:
        safe_percentile_name = percentile_name.replace(' ', '_').replace('-', '_')
        fig_combined.savefig(
            os.path.join(save_loc, f'combined_neurons_passing_{safe_percentile_name}_{day1_name}_vs_{day2_name}.pdf'),
            dpi=300, bbox_inches='tight'
        )
        print(f"Combined neurons passing thresholds plot saved to {save_loc}")
    plt.close()

    # console stats
    print(f"\n=== {percentile_name} Statistics (n={len(common_mice)} mice) ===")
    print(f"  Excitatory {day1_name}: {mean_exc_day1:.2f}% ± {sem_exc_day1:.2f}% (SEM)")
    print(f"  Excitatory {day2_name}: {mean_exc_day2:.2f}% ± {sem_exc_day2:.2f}% (SEM)  "
          f"[{test_exc}, alt={alternative}, Shapiro p={np.nan if np.isnan(info_exc['shapiro_p']) else info_exc['shapiro_p']:.3g}, "
          f"outliers={info_exc['outliers']}]  p={p_exc:.3g}")
    print(f"  Inhibitory {day1_name}: {mean_inh_day1:.2f}% ± {sem_inh_day1:.2f}% (SEM)")
    print(f"  Inhibitory {day2_name}: {mean_inh_day2:.2f}% ± {sem_inh_day2:.2f}% (SEM)  "
          f"[{test_inh}, alt={alternative}, Shapiro p={np.nan if np.isnan(info_inh['shapiro_p']) else info_inh['shapiro_p']:.3g}, "
          f"outliers={info_inh['outliers']}]  p={p_inh:.3g}")
    print(f"  Combined {day1_name}: {mean_combined_day1:.2f}% ± {sem_combined_day1:.2f}% (SEM)")
    print(f"  Combined {day2_name}: {mean_combined_day2:.2f}% ± {sem_combined_day2:.2f}% (SEM)  "
          f"[{test_comb}, alt={alternative}, Shapiro p={np.nan if np.isnan(info_comb['shapiro_p']) else info_comb['shapiro_p']:.3g}, "
          f"outliers={info_comb['outliers']}]  p={p_comb:.3g}")


def create_bar_plots_across_mice(comparison_results, save_loc=None):
    """
    Create bar plots showing neurons passing thresholds across mice.
    
    Parameters:
    -----------
    comparison_results : dict
        Dictionary with mouse names as keys and their comparison results as values
    save_loc : str, optional
        Path to save the plots
    """
    os.makedirs(save_loc, exist_ok=True)
    if not comparison_results:
        print("No comparison results to plot")
        return
    
    # Extract data for plotting
    mice = list(comparison_results.keys())
    
    # Get percentages for each mouse
    pct_exc_99_5th = []
    pct_exc_99th = []
    pct_exc_97_5th = []
    pct_exc_95th = []
    pct_inh_0_5th = []
    pct_inh_1st = []
    pct_inh_2_5th = []
    pct_inh_5th = []
    
    for mouse in mice:
        pct_data = comparison_results[mouse]['percentages_passing_thresholds']
        pct_exc_99_5th.append(pct_data['pct_exc_above_99_5th'])
        pct_exc_99th.append(pct_data['pct_exc_above_99th'])
        pct_exc_97_5th.append(pct_data['pct_exc_above_97_5th'])
        pct_exc_95th.append(pct_data['pct_exc_above_95th'])
        pct_inh_0_5th.append(pct_data['pct_inh_below_0_5th'])
        pct_inh_1st.append(pct_data['pct_inh_below_1st'])
        pct_inh_2_5th.append(pct_data['pct_inh_below_2_5th'])
        pct_inh_5th.append(pct_data['pct_inh_below_5th'])
    
    # Set up the plotting - now with 4 subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
    
    # Plot 1: 99.5th-0.5th percentiles
    x = np.arange(len(mice))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, pct_exc_99_5th, width, label='Positive(≥99.5th)',
                    color='darkred', alpha=0.7)
    bars2 = ax1.bar(x + width/2, pct_inh_0_5th, width, label='Negative(≤0.5th)',
                    color='darkblue', alpha=0.7)
    
    ax1.set_xlabel('Mouse')
    ax1.set_ylabel('Percentage of neurons')
    ax1.set_title('Neurons passing 99.5th-0.5th percentile thresholds')
    ax1.set_xticks(x)
    ax1.set_xticklabels(mice, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
    
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
    
    # Plot 2: 99th-1st percentiles
    bars3 = ax2.bar(x - width/2, pct_exc_99th, width, label='Positive (≥99th)',
                    color='red', alpha=0.7)
    bars4 = ax2.bar(x + width/2, pct_inh_1st, width, label='Negative (≤1st)',
                    color='blue', alpha=0.7)
    
    ax2.set_xlabel('Mouse')
    ax2.set_ylabel('Percentage of neurons')
    ax2.set_title('Neurons passing 99th-1st percentile thresholds')
    ax2.set_xticks(x)
    ax2.set_xticklabels(mice, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars3:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
    
    for bar in bars4:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
    
    # Plot 3: 97.5th-2.5th percentiles
    bars5 = ax3.bar(x - width/2, pct_exc_97_5th, width, label='Positive (≥97.5th)',
                    color='orange', alpha=0.7)
    bars6 = ax3.bar(x + width/2, pct_inh_2_5th, width, label='Negative (≤2.5th)',
                    color='cyan', alpha=0.7)
    
    ax3.set_xlabel('Mouse')
    ax3.set_ylabel('Percentage of neurons')
    ax3.set_title('Neurons passing 97.5th-2.5th percentile thresholds')
    ax3.set_xticks(x)
    ax3.set_xticklabels(mice, rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars5:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
    
    for bar in bars6:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
    
    # Plot 4: 95th-5th percentiles
    bars7 = ax4.bar(x - width/2, pct_exc_95th, width, label='Positive (≥95th)',
                    color='lightcoral', alpha=0.7)
    bars8 = ax4.bar(x + width/2, pct_inh_5th, width, label='Negative (≤5th)',
                    color='lightblue', alpha=0.7)
    
    ax4.set_xlabel('Mouse')
    ax4.set_ylabel('Percentage of neurons')
    ax4.set_title('Neurons passing 95th-5th percentile thresholds')
    ax4.set_xticks(x)
    ax4.set_xticklabels(mice, rotation=45)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars7:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
    
    for bar in bars8:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    if save_loc:
        plt.savefig(os.path.join(save_loc, 'neurons_passing_thresholds_across_mice.pdf'),
                   dpi=300, bbox_inches='tight')
        print(f"Individual mouse plots saved to {save_loc}")
    
    plt.close()

def create_mean_bar_plot_95_5(comparison_results, save_loc=None):
    """
    Create bar plot showing mean percentage across mice with SEM and individual data points for 95th-5th percentiles.
    
    Parameters:
    -----------
    comparison_results : dict
        Dictionary with mouse names as keys and their comparison results as values
    save_loc : str, optional
        Path to save the plot
    """
    if not comparison_results:
        print("No comparison results to plot")
        return
    
    # Extract data for plotting
    mice = list(comparison_results.keys())
    
    # Get percentages for each mouse
    pct_exc_95th = []
    pct_inh_5th = []
    
    for mouse in mice:
        pct_data = comparison_results[mouse]['percentages_passing_thresholds']
        pct_exc_95th.append(pct_data['pct_exc_above_95th'])
        pct_inh_5th.append(pct_data['pct_inh_below_5th'])
    
    # Convert to numpy arrays
    pct_exc_95th = np.array(pct_exc_95th)
    pct_inh_5th = np.array(pct_inh_5th)
    
    # Calculate means and SEMs
    mean_exc_95th = np.mean(pct_exc_95th)
    sem_exc_95th = np.std(pct_exc_95th) / np.sqrt(len(pct_exc_95th))
    mean_inh_5th = np.mean(pct_inh_5th)
    sem_inh_5th = np.std(pct_inh_5th) / np.sqrt(len(pct_inh_5th))
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    categories = ['Positive\n(≥95th)', 'Negative\n(≤5th)']
    means = [mean_exc_95th, mean_inh_5th]
    sems = [sem_exc_95th, sem_inh_5th]
    
    x_pos = np.arange(len(categories))
    width = 0.6
    
    # Create bars
    bars = ax.bar(x_pos, means, width, yerr=sems, 
                  capsize=5, color=['red', 'blue'], alpha=0.7)
    
    # Add individual data points
    for i, (exc_data, inh_data) in enumerate(zip(pct_exc_95th, pct_inh_5th)):
        # Add some jitter to x positions for individual points
        jitter_exc = np.random.normal(0, 0.05)
        jitter_inh = np.random.normal(0, 0.05)
        
        ax.scatter(0 + jitter_exc, exc_data, color='darkred', alpha=0.6, s=30, zorder=5)
        ax.scatter(1 + jitter_inh, inh_data, color='darkblue', alpha=0.6, s=30, zorder=5)
    
    ax.set_xlabel('Threshold Type')
    ax.set_ylabel('Percentage of neurons')
    ax.set_title('Mean neurons passing 95th-5th percentile thresholds\nacross mice (n={})'.format(len(mice)))
    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories)
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, mean_val, sem_val) in enumerate(zip(bars, means, sems)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + sem_val + 0.5,
                f'{mean_val:.1f}±{sem_val:.1f}%', ha='center', va='bottom', fontsize=12)
    
    plt.tight_layout()
    
    if save_loc:
        plt.savefig(os.path.join(save_loc, 'mean_neurons_passing_thresholds_95_5.pdf'),
                   dpi=300, bbox_inches='tight')
        print(f"95th-5th percentile plot saved to {save_loc}")
    
    plt.close()
    
    # Print statistics
    print(f"\n=== 95th-5th Percentile Statistics (n={len(mice)} mice) ===")
    print(f"  Positive (≥95th): {mean_exc_95th:.2f}% ± {sem_exc_95th:.2f}% (SEM)")
    print(f"  Negative (≤5th): {mean_inh_5th:.2f}% ± {sem_inh_5th:.2f}% (SEM)")

def create_mean_bar_plot_99_1(comparison_results, save_loc=None):
    """
    Create bar plot showing mean percentage across mice with SEM and individual data points for 99th-1st percentiles.
    
    Parameters:
    -----------
    comparison_results : dict
        Dictionary with mouse names as keys and their comparison results as values
    save_loc : str, optional
        Path to save the plot
    """
    if not comparison_results:
        print("No comparison results to plot")
        return
    
    # Extract data for plotting
    mice = list(comparison_results.keys())
    
    # Get percentages for each mouse
    pct_exc_99th = []
    pct_inh_1st = []
    
    for mouse in mice:
        pct_data = comparison_results[mouse]['percentages_passing_thresholds']
        pct_exc_99th.append(pct_data['pct_exc_above_99th'])
        pct_inh_1st.append(pct_data['pct_inh_below_1st'])
    
    # Convert to numpy arrays
    pct_exc_99th = np.array(pct_exc_99th)
    pct_inh_1st = np.array(pct_inh_1st)
    
    # Calculate means and SEMs
    mean_exc_99th = np.mean(pct_exc_99th)
    sem_exc_99th = np.std(pct_exc_99th) / np.sqrt(len(pct_exc_99th))
    mean_inh_1st = np.mean(pct_inh_1st)
    sem_inh_1st = np.std(pct_inh_1st) / np.sqrt(len(pct_inh_1st))
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    categories = ['Positive\n(≥99th)', 'Negative\n(≤1st)']
    means = [mean_exc_99th, mean_inh_1st]
    sems = [sem_exc_99th, sem_inh_1st]
    
    x_pos = np.arange(len(categories))
    width = 0.6
    
    # Create bars
    bars = ax.bar(x_pos, means, width, yerr=sems, 
                  capsize=5, color=['darkred', 'darkblue'], alpha=0.7)
    
    # Add individual data points
    for i, (exc_data, inh_data) in enumerate(zip(pct_exc_99th, pct_inh_1st)):
        # Add some jitter to x positions for individual points
        jitter_exc = np.random.normal(0, 0.05)
        jitter_inh = np.random.normal(0, 0.05)
        
        ax.scatter(0 + jitter_exc, exc_data, color='darkred', alpha=0.6, s=30, zorder=5)
        ax.scatter(1 + jitter_inh, inh_data, color='darkblue', alpha=0.6, s=30, zorder=5)
    
    ax.set_xlabel('Threshold Type')
    ax.set_ylabel('Percentage of neurons')
    ax.set_title('Mean neurons passing 99th-1st percentile thresholds\nacross mice (n={})'.format(len(mice)))
    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories)
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, mean_val, sem_val) in enumerate(zip(bars, means, sems)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + sem_val + 0.5,
                f'{mean_val:.1f}±{sem_val:.1f}%', ha='center', va='bottom', fontsize=12)
    
    plt.tight_layout()
    
    if save_loc:
        plt.savefig(os.path.join(save_loc, 'mean_neurons_passing_thresholds_99_1.pdf'),
                   dpi=300, bbox_inches='tight')
        print(f"99th-1st percentile plot saved to {save_loc}")
    
    plt.close()
    
    # Print statistics
    print(f"\n=== 99th-1st Percentile Statistics (n={len(mice)} mice) ===")
    print(f"  Positive (≥99th): {mean_exc_99th:.2f}% ± {sem_exc_99th:.2f}% (SEM)")
    print(f"  Negative (≤1st): {mean_inh_1st:.2f}% ± {sem_inh_1st:.2f}% (SEM)")

def create_mean_bar_plot_99_5_0_5(comparison_results, save_loc=None):
    """
    Create bar plot showing mean percentage across mice with SEM and individual data points for 99.5th-0.5th percentiles.
    
    Parameters:
    -----------
    comparison_results : dict
        Dictionary with mouse names as keys and their comparison results as values
    save_loc : str, optional
        Path to save the plot
    """
    if not comparison_results:
        print("No comparison results to plot")
        return
    
    # Extract data for plotting
    mice = list(comparison_results.keys())
    
    # Get percentages for each mouse
    pct_exc_99_5th = []
    pct_inh_0_5th = []
    
    for mouse in mice:
        pct_data = comparison_results[mouse]['percentages_passing_thresholds']
        pct_exc_99_5th.append(pct_data['pct_exc_above_99_5th'])
        pct_inh_0_5th.append(pct_data['pct_inh_below_0_5th'])
    
    # Convert to numpy arrays
    pct_exc_99_5th = np.array(pct_exc_99_5th)
    pct_inh_0_5th = np.array(pct_inh_0_5th)
    
    # Calculate means and SEMs
    mean_exc_99_5th = np.mean(pct_exc_99_5th)
    sem_exc_99_5th = np.std(pct_exc_99_5th) / np.sqrt(len(pct_exc_99_5th))
    mean_inh_0_5th = np.mean(pct_inh_0_5th)
    sem_inh_0_5th = np.std(pct_inh_0_5th) / np.sqrt(len(pct_inh_0_5th))
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    categories = ['Positive\n(≥99.5th)', 'Negative\n(≤0.5th)']
    means = [mean_exc_99_5th, mean_inh_0_5th]
    sems = [sem_exc_99_5th, sem_inh_0_5th]
    
    x_pos = np.arange(len(categories))
    width = 0.6
    
    # Create bars
    bars = ax.bar(x_pos, means, width, yerr=sems, 
                  capsize=5, color=['darkred', 'darkblue'], alpha=0.7)
    
    # Add individual data points
    for i, (exc_data, inh_data) in enumerate(zip(pct_exc_99_5th, pct_inh_0_5th)):
        # Add some jitter to x positions for individual points
        jitter_exc = np.random.normal(0, 0.05)
        jitter_inh = np.random.normal(0, 0.05)
        
        ax.scatter(0 + jitter_exc, exc_data, color='darkred', alpha=0.6, s=30, zorder=5)
        ax.scatter(1 + jitter_inh, inh_data, color='darkblue', alpha=0.6, s=30, zorder=5)
    
    ax.set_xlabel('Threshold Type')
    ax.set_ylabel('Percentage of neurons')
    ax.set_title('Mean neurons passing 99.5th-0.5th percentile thresholds\nacross mice (n={})'.format(len(mice)))
    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories)
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, mean_val, sem_val) in enumerate(zip(bars, means, sems)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + sem_val + 0.5,
                f'{mean_val:.1f}±{sem_val:.1f}%', ha='center', va='bottom', fontsize=12)
    
    plt.tight_layout()
    
    if save_loc:
        plt.savefig(os.path.join(save_loc, 'mean_neurons_passing_thresholds_99_5_0_5.pdf'),
                   dpi=300, bbox_inches='tight')
        print(f"99.5th-0.5th percentile plot saved to {save_loc}")
    
    plt.close()
    
    # Print statistics
    print(f"\n=== 99.5th-0.5th Percentile Statistics (n={len(mice)} mice) ===")
    print(f"  Positive (≥99.5th): {mean_exc_99_5th:.2f}% ± {sem_exc_99_5th:.2f}% (SEM)")
    print(f"  Negative (≤0.5th): {mean_inh_0_5th:.2f}% ± {sem_inh_0_5th:.2f}% (SEM)")

def create_mean_bar_plot_97_5_2_5(comparison_results, save_loc=None):
    """
    Create bar plot showing mean percentage across mice with SEM and individual data points for 97.5th-2.5th percentiles.
    
    Parameters:
    -----------
    comparison_results : dict
        Dictionary with mouse names as keys and their comparison results as values
    save_loc : str, optional
        Path to save the plot
    """
    if not comparison_results:
        print("No comparison results to plot")
        return
    
    # Extract data for plotting
    mice = list(comparison_results.keys())
    
    # Get percentages for each mouse
    pct_exc_97_5th = []
    pct_inh_2_5th = []
    
    for mouse in mice:
        pct_data = comparison_results[mouse]['percentages_passing_thresholds']
        pct_exc_97_5th.append(pct_data['pct_exc_above_97_5th'])
        pct_inh_2_5th.append(pct_data['pct_inh_below_2_5th'])
    
    # Convert to numpy arrays
    pct_exc_97_5th = np.array(pct_exc_97_5th)
    pct_inh_2_5th = np.array(pct_inh_2_5th)
    
    # Calculate means and SEMs
    mean_exc_97_5th = np.mean(pct_exc_97_5th)
    sem_exc_97_5th = np.std(pct_exc_97_5th) / np.sqrt(len(pct_exc_97_5th))
    mean_inh_2_5th = np.mean(pct_inh_2_5th)
    sem_inh_2_5th = np.std(pct_inh_2_5th) / np.sqrt(len(pct_inh_2_5th))
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    categories = ['Positive\n(≥97.5th)', 'Negative\n(≤2.5th)']
    means = [mean_exc_97_5th, mean_inh_2_5th]
    sems = [sem_exc_97_5th, sem_inh_2_5th]
    
    x_pos = np.arange(len(categories))
    width = 0.6
    
    # Create bars
    bars = ax.bar(x_pos, means, width, yerr=sems, 
                  capsize=5, color=['orange', 'cyan'], alpha=0.7)
    
    # Add individual data points
    for i, (exc_data, inh_data) in enumerate(zip(pct_exc_97_5th, pct_inh_2_5th)):
        # Add some jitter to x positions for individual points
        jitter_exc = np.random.normal(0, 0.05)
        jitter_inh = np.random.normal(0, 0.05)
        
        ax.scatter(0 + jitter_exc, exc_data, color='orange', alpha=0.6, s=30, zorder=5)
        ax.scatter(1 + jitter_inh, inh_data, color='cyan', alpha=0.6, s=30, zorder=5)
    
    ax.set_xlabel('Threshold Type')
    ax.set_ylabel('Percentage of neurons')
    ax.set_title('Mean neurons passing 97.5th-2.5th percentile thresholds\nacross mice (n={})'.format(len(mice)))
    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories)
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, mean_val, sem_val) in enumerate(zip(bars, means, sems)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + sem_val + 0.5,
                f'{mean_val:.1f}±{sem_val:.1f}%', ha='center', va='bottom', fontsize=12)
    
    plt.tight_layout()
    
    if save_loc:
        plt.savefig(os.path.join(save_loc, 'mean_neurons_passing_thresholds_97_5_2_5.pdf'),
                   dpi=300, bbox_inches='tight')
        print(f"97.5th-2.5th percentile plot saved to {save_loc}")
    
    plt.close()
    
    # Print statistics
    print(f"\n=== 97.5th-2.5th Percentile Statistics (n={len(mice)} mice) ===")
    print(f"  Positive(≥97.5th): {mean_exc_97_5th:.2f}% ± {sem_exc_97_5th:.2f}% (SEM)")
    print(f"  Negative (≤2.5th): {mean_inh_2_5th:.2f}% ± {sem_inh_2_5th:.2f}% (SEM)")

def create_summary_table(comparison_results, save_loc=None):
    """
    Create a summary table with all the data.
    
    Parameters:
    -----------
    comparison_results : dict
        Dictionary with mouse names as keys and their comparison results as values
    save_loc : str, optional
        Path to save the table
    """
    if not comparison_results:
        print("No comparison results to create table")
        return
    
    # Prepare data for table
    table_data = []
    
    for mouse_name, results in comparison_results.items():
        pct_data = results['percentages_passing_thresholds']
        count_data = results['neurons_passing_thresholds']
        
        row = {
            'Mouse': mouse_name,
            'Exc_99_5th_pct': pct_data['pct_exc_above_99_5th'],
            'Exc_99th_pct': pct_data['pct_exc_above_99th'],
            'Exc_97_5th_pct': pct_data['pct_exc_above_97_5th'],
            'Exc_95th_pct': pct_data['pct_exc_above_95th'],
            'Inh_0_5th_pct': pct_data['pct_inh_below_0_5th'],
            'Inh_1st_pct': pct_data['pct_inh_below_1st'],
            'Inh_2_5th_pct': pct_data['pct_inh_below_2_5th'],
            'Inh_5th_pct': pct_data['pct_inh_below_5th'],
            'Exc_99_5th_count': count_data['n_exc_above_99_5th'],
            'Exc_99th_count': count_data['n_exc_above_99th'],
            'Exc_97_5th_count': count_data['n_exc_above_97_5th'],
            'Exc_95th_count': count_data['n_exc_above_95th'],
            'Inh_0_5th_count': count_data['n_inh_below_0_5th'],
            'Inh_1st_count': count_data['n_inh_below_1st'],
            'Inh_2_5th_count': count_data['n_inh_below_2_5th'],
            'Inh_5th_count': count_data['n_inh_below_5th']
        }
        table_data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(table_data)
    
    # Print table
    print("\n=== Summary Table ===")
    print(df.to_string(index=False, float_format='%.2f'))
    
    # Save table
    if save_loc:
        csv_path = os.path.join(save_loc, 'comparison_results_summary.csv')
        df.to_csv(csv_path, index=False)
        print(f"\nSummary table saved to {csv_path}")
    
    return df



def export_plot_values(save_loc, percentile_name, mice, day1, day2,
                       exc1, inh1, exc2, inh2, combined1, combined2, mean_sem):
    """One sheet per threshold: percentages per mouse and plotted mean/SEM."""
    destination = Path(save_loc) / 'Gaussian_vs_Sigmoid_source_data.xlsx'
    table = pd.DataFrame({
        'Mouse': mice,
        f'{day1}_positive_pct': exc1, f'{day1}_negative_pct': inh1,
        f'{day1}_combined_pct': combined1,
        f'{day2}_positive_pct': exc2, f'{day2}_negative_pct': inh2,
        f'{day2}_combined_pct': combined2,
    })
    rows = []
    for day, arrays in [(day1, [exc1, inh1, combined1]), (day2, [exc2, inh2, combined2])]:
        for category, values in zip(['positive', 'negative', 'combined'], arrays):
            mean, sem = mean_sem(values)
            rows.append([day, category, mean, sem, int(np.isfinite(values).sum())])
    summary = pd.DataFrame(rows, columns=['Condition', 'Category', 'Mean_pct', 'SEM_pct', 'n_mice'])
    sheet = percentile_name
    kwargs = {'mode': 'a', 'if_sheet_exists': 'overlay'} if destination.exists() else {'mode': 'w'}
    with pd.ExcelWriter(destination, engine='openpyxl', **kwargs) as writer:
        # Clear an old threshold sheet once, then write both tables without replacing it.
        if sheet in writer.book.sheetnames:
            del writer.book[sheet]
        table.to_excel(writer, sheet_name=sheet, index=False, startrow=3)
        summary.to_excel(writer, sheet_name=sheet, index=False, startrow=len(table)+6)
        ws = writer.sheets[sheet]
        ws['A1'] = f'{day1} versus {day2} | thresholds {percentile_name}'
        ws['A2'] = 'Values are percentages (0–100); combined = positive + negative; points are paired mice.'
        ws['A3'] = 'Source: comparison_results.pkl in the configured folders. SEM = sample SD / sqrt(finite n).'
        ws.freeze_panes = 'B5'
        from openpyxl.styles import Font, PatternFill
        for row in [4, len(table)+7]:
            for cell in ws[row]:
                cell.font = Font(bold=True)
                cell.fill = PatternFill('solid', fgColor='E8ECEF')
        for column in ws.iter_cols():
            ws.column_dimensions[column[0].column_letter].width = 30
            for cell in column:
                if isinstance(cell.value, float):
                    cell.number_format = '0.000000'
    print(f'Exported {sheet}: {destination}')




# Exact active paths copied from the original script.
GAUSSIAN_PATHS = {
    'OI116': Path('Z:\\2p_data\\OI116\\240529\\ML\\data\\processed_data\\auROC_freq_history_control_CR-CR_rep1\\all_reps_with_sigmoid_gaussian\\subset_analysis\\gaussian'),
    'OI124': Path('Z:\\2p_data\\OI124\\240610\\ML\\data\\processed_data\\auROC_freq_history_control_CR-CR_rep1\\all_reps_with_sigmoid_gaussian\\subset_analysis\\gaussian'),
    'OI151': Path('Z:\\2p_data\\OI151\\240827\\ML\\data\\processed_data\\auROC_freq_history_control_CR-CR_rep1\\all_reps_with_sigmoid_gaussian\\subset_analysis\\gaussian'),
    'OI208': Path('Z:\\2p_data\\OI208\\250227\\ML\\data\\processed_data\\auROC_freq_history_control_CR-CR_rep1\\all_reps_with_sigmoid_gaussian\\subset_analysis\\gaussian'),
    'OI152': Path('Z:\\2p_data\\OI152\\241003\\ML\\data\\processed_data\\auROC_freq_history_control_CR-CR_rep1\\all_reps_with_sigmoid_gaussian\\subset_analysis\\gaussian'),
}
SIGMOID_PATHS = {
    'OI116': Path('Z:\\2p_data\\OI116\\240529\\ML\\data\\processed_data\\auROC_freq_history_control_CR-CR_rep1\\all_reps_with_sigmoid_gaussian\\subset_analysis\\sigmoid'),
    'OI124': Path('Z:\\2p_data\\OI124\\240610\\ML\\data\\processed_data\\auROC_freq_history_control_CR-CR_rep1\\all_reps_with_sigmoid_gaussian\\subset_analysis\\sigmoid'),
    'OI151': Path('Z:\\2p_data\\OI151\\240827\\ML\\data\\processed_data\\auROC_freq_history_control_CR-CR_rep1\\all_reps_with_sigmoid_gaussian\\subset_analysis\\sigmoid'),
    'OI208': Path('Z:\\2p_data\\OI208\\250227\\ML\\data\\processed_data\\auROC_freq_history_control_CR-CR_rep1\\all_reps_with_sigmoid_gaussian\\subset_analysis\\sigmoid'),
    'OI152': Path('Z:\\2p_data\\OI152\\241003\\ML\\data\\processed_data\\auROC_freq_history_control_CR-CR_rep1\\all_reps_with_sigmoid_gaussian\\subset_analysis\\sigmoid'),
}

from datetime import datetime
import hashlib
import json
import sys

OUTPUT_PARENT = Path.home() / 'Documents' / 'CP_Gaussian_Sigmoid'
THRESHOLDS = [
    ('99.5th-0.5th', 'pct_exc_above_99_5th', 'pct_inh_below_0_5th'),
    ('99th-1st', 'pct_exc_above_99th', 'pct_inh_below_1st'),
    ('97.5th-2.5th', 'pct_exc_above_97_5th', 'pct_inh_below_2_5th'),
    ('95th-5th', 'pct_exc_above_95th', 'pct_inh_below_5th'),
]

def run(gaussian_paths=None, sigmoid_paths=None, output_parent=None):
    from openpyxl import load_workbook
    from openpyxl.styles import Font, PatternFill
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(errors='backslashreplace')
    gpaths=GAUSSIAN_PATHS if gaussian_paths is None else gaussian_paths
    spaths=SIGMOID_PATHS if sigmoid_paths is None else sigmoid_paths
    if set(gpaths) != set(spaths) or len(gpaths)<2:
        raise ValueError('Gaussian and Sigmoid must contain the same mouse IDs (at least two).')
    groups={'Gaussian':gpaths,'Sigmoid':spaths}
    mice=sorted(gpaths)
    missing=[str(Path(p)/'comparison_results.pkl') for paths in groups.values() for p in paths.values()
             if not (Path(p)/'comparison_results.pkl').is_file()]
    if missing:
        raise FileNotFoundError('Missing source files; check the configured paths:\n'+'\n'.join(missing))
    results={name:{} for name in groups}
    sources=[]
    # Explicit loading: a read/schema error must not silently reduce the sample.
    for name,paths in groups.items():
        for mouse in mice:
            path=Path(paths[mouse])/'comparison_results.pkl'
            raw=path.read_bytes()
            obj=pickle.loads(raw)
            pct=obj['percentages_passing_thresholds']
            for threshold,pos,neg in THRESHOLDS:
                for key in [pos,neg]:
                    value=float(pct[key])
                    if not np.isfinite(value) or not 0 <= value <= 100:
                        raise ValueError(f'{mouse} {name}: {key} must be a finite percentage (0-100).')
            results[name][mouse]=obj
            sources.append({'Condition':name,'Mouse':mouse,'Path':str(path),
                            'SHA256':hashlib.sha256(raw).hexdigest()})
    parent=OUTPUT_PARENT if output_parent is None else Path(output_parent)
    out=parent/('run_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(parents=True,exist_ok=False)
    print(f'Output folder: {out}',flush=True)
    stats=[]
    for threshold,pos,neg in THRESHOLDS:
        create_individual_percentile_comparison(results,threshold,pos,neg,save_loc=str(out),alternative='two-sided')
        for category in ['positive','negative','combined']:
            arrays=[]
            for name in groups:
                vals=[]
                for mouse in mice:
                    p=results[name][mouse]['percentages_passing_thresholds']
                    vals.append(p[pos] if category=='positive' else p[neg] if category=='negative' else p[pos]+p[neg])
                arrays.append(np.asarray(vals,float))
            p,test,info=choose_test(*arrays,paired=True,mode='auto',alternative='two-sided')
            report=info['report']
            effect=report['effect_size']
            stats.append({'Threshold':threshold,'Category':category,'First condition':'Gaussian','Second condition':'Sigmoid',
                          'Test':test,'Alternative':'two-sided','N pairs':info['n'],'p value':p,
                          'Statistic name':report['statistic_name'],'Statistic':report['statistic_value'],
                          'df':report['df'],'Effect name':effect['name'],'Effect value':effect['value'],
                          'Shapiro p (paired differences)':info['shapiro_p'],'MAD outliers':info['outliers']})
        plt.close('all')
    target=out/'Gaussian_vs_Sigmoid_source_data.xlsx'
    with pd.ExcelWriter(target,engine='openpyxl',mode='a',if_sheet_exists='replace') as writer:
        pd.DataFrame(stats).to_excel(writer,sheet_name='Statistics',index=False)
        pd.DataFrame(sources).to_excel(writer,sheet_name='Sources',index=False)
        for name in ['Statistics','Sources']:
            ws=writer.sheets[name]
            ws.freeze_panes='A2'
            ws.auto_filter.ref=ws.dimensions
            for cell in ws[1]:
                cell.font=Font(bold=True)
                cell.fill=PatternFill('solid',fgColor='E8ECEF')
            for col in ws.columns:
                ws.column_dimensions[col[0].column_letter].width=26 if name=='Statistics' else 24
                for cell in col:
                    if isinstance(cell.value,float): cell.number_format='0.000000'
            if name=='Sources':
                ws.column_dimensions['C'].width=100
                ws.column_dimensions['D'].width=68
    # Read back every plotted individual value, mean and SEM from the saved file.
    book=load_workbook(target,data_only=True,read_only=True)
    for threshold,pos,neg in THRESHOLDS:
        ws=book[threshold]
        expected=[]
        for mouse in mice:
            row=[mouse]
            for name in groups:
                pct=results[name][mouse]['percentages_passing_thresholds']
                row.extend([pct[pos],pct[neg],pct[pos]+pct[neg]])
            expected.append(row)
        actual=list(ws.iter_rows(min_row=5,max_row=4+len(mice),min_col=1,max_col=7,values_only=True))
        assert [r[0] for r in actual]==mice
        np.testing.assert_allclose(np.array([r[1:] for r in actual],float),np.array([r[1:] for r in expected],float),rtol=1e-12,atol=1e-12)
        summaries=list(ws.iter_rows(min_row=len(mice)+8,max_row=len(mice)+13,min_col=1,max_col=5,values_only=True))
        for i,row in enumerate(summaries):
            vals=np.array([r[i+1] for r in expected],float)
            np.testing.assert_allclose(row[2:4],[np.mean(vals),np.std(vals,ddof=1)/np.sqrt(len(vals))],rtol=1e-12,atol=1e-12)
    book.close()
    for source in sources:
        if hashlib.sha256(Path(source['Path']).read_bytes()).hexdigest()!=source['SHA256']:
            raise RuntimeError('Input changed during analysis: '+source['Path'])
    (out/'verification.json').write_text(json.dumps({'excel_readback':'passed','paired_mouse_order':mice,
        'conditions':['Gaussian','Sigmoid'],'SEM':'sample SD (ddof=1) / sqrt(n)',
        'tests':'Original choose_test: paired=True, mode=auto, alternative=two-sided',
        'source_hashes_unchanged':True},indent=2),encoding='utf-8')
    print(f'Plots and verified Excel saved to: {out}',flush=True)
    return out

if __name__ == '__main__':
    run()
