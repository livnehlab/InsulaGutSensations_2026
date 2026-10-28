"""Across-mice neurometric summary (as in post_neurometric.py), run separately for
trial-history subsets (trials after a hit / not after a hit), with Excel source data.

Dependencies: numpy, pandas, scipy, matplotlib, statsmodels, openpyxl.

SUBSETS controls the runs: ('after_hit', 'not_after_hit') runs both, separately;
use ('after_hit',) or ('not_after_hit',) to run only one.

Inputs for each mouse/session:
  <DATA_ROOT>/<mouse>/<day>/ML/data/processed_data/trial_history_neurometric/<subset>/
    half_diff_values_sigmoid.npy
    half_max_values_gaussian.npy
    r_squared_values.npy
    excitatory_inhibitory_statistics.csv
The behavioral conditions_played.npy and responses.npy are read from ML/data2 for
mouse01/mouse05 and from ML/data for the other mice.

The plots and statistics are the same as in post_neurometric.py: frequency histograms,
percentage bars, excitation / inhibition, threshold comparisons, histograms / KDEs,
scatters and violins. Both the pooled and the per-mouse mean frequency distributions are
generated. Frequency histograms are normalized within each fit's in-range frequency
counts (not all analyzed neurons); the total-category percentage bars use all neurons.

Behavioral thresholds are computed from the full session after removing the 23 priming
trials (they are not refit per trial-history subset); the subsets apply to the neural
results only.

All configured mice are required (a missing input raises an error). Each run writes to
Documents/Neurometric_history/run_<timestamp>/<subset>/: all plot files, PNG previews,
source_data.xlsx, the excitation / inhibition statistics workbook, copies of the inputs
and verification.json.

source_data.xlsx tabs:
Percent_all_neurons: two-bar group means / SEMs.
Frequency_pooled: pooled frequency bin counts and percentages (no SEM).
Frequency_mean_SEM: equal-mouse mean / SEM plus each mouse's bin percentages.
Per_mouse_metrics: paired per-mouse percentages, thresholds, peaks and fractions.
Neuron_frequencies: saved per-neuron frequency values and bin membership.
Behavior_trials: trial inputs and priming-trial inclusion.
Plot_values: bar bounds, line / dot coordinates, error segments and fill paths
             captured from every saved figure, keyed by filename and axis.
Statistics: test results, with full diagnostic details in JSON.
Sources: input paths, numbered copies and SHA256 hashes.
"""
from pathlib import Path
from datetime import datetime
import hashlib
import io
import json
import traceback
import sys
from matplotlib.container import BarContainer
from matplotlib.collections import PathCollection, LineCollection, PolyCollection
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter


# ---------------------------------------------------------------------------
# Data locations (set these before running)
# ---------------------------------------------------------------------------
DATA_ROOT = Path(r'path\to\2p_data')  # folder with the imaging sessions (<mouse>/<session>/ML/...)

OUTPUT_PARENT = Path.home() / 'Documents' / 'Neurometric_history'
MOUSE_SESSIONS = [('mouse09','sess06'),('mouse10','sess09'),('mouse05','sess04'),('mouse01','sess02'),('mouse14','sess12')]
MOUSE_IDS = [mouse for mouse, day in MOUSE_SESSIONS]
SUBSETS = ('after_hit','not_after_hit')
_ACTIVE_CAPTURE = []
_ACTIVE_OUTPUT = None
ORIGINAL_SCRIPT = 'post_neurometric.py'
ORIGINAL_SHA256 = 'b9443eb222ed951eb6bfc73dbc27bef03571851eb3254afb6a40d96ba2c34c11'

# ===========================
# IMPORTS
# ===========================
import os
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import scipy.io
import matplotlib.pyplot as plt
import scipy.stats as st
from os.path import join
from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde
from scipy.stats import shapiro, levene, ttest_rel, ttest_ind, mannwhitneyu, wilcoxon, t as t_dist, iqr, rankdata, norm
from scipy.stats import t as student_t
# --- statistics imports ---
import os
import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.stats.anova import AnovaRM



# ===========================
# FUNCTION DEFINITIONS
# ===========================
def sigmoid(x, L, x0, k, b):
    """Sigmoid function for curve fitting."""
    y = L / (1 + np.exp(-k * (x - x0))) + b
    return y

# --- Helpers ---
ALPHA_NORMALITY = 0.05

def _choose_paired_test(x, y, alpha=ALPHA_NORMALITY):
    """Auto paired test: Shapiro on differences -> t-test if normal else Wilcoxon."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    d = x - y
    res = {
        "test": None, "stat_name": None, "stat_value": np.nan,
        "df": np.nan, "p_value": np.nan,
        "shapiro_W": np.nan, "shapiro_p": np.nan,
        "mean_diff": float(np.nanmean(d)) if d.size else np.nan,
        "std_diff": float(np.nanstd(d, ddof=1)) if d.size > 1 else np.nan,
        "n": int(d.size),
        "cohens_dz": _cohens_d_paired(x, y) if d.size > 1 else np.nan
    }
    use_normal = False
    if d.size >= 3:
        try:
            W, p = stats.shapiro(d)
            res["shapiro_W"], res["shapiro_p"] = float(W), float(p)
            use_normal = (p >= alpha)
        except Exception:
            use_normal = False
    if use_normal:
        t, p = stats.ttest_rel(x, y, nan_policy="omit")
        df = max(int(np.count_nonzero(np.isfinite(d)) - 1), 0)
        res.update({"test": "paired t-test", "stat_name": "t", "stat_value": float(t), "df": df, "p_value": float(p)})
    else:
        # continuity correction on
        try:
            W, p = stats.wilcoxon(x, y, zero_method="wilcox", correction=True, alternative="two-sided")
        except ValueError:
            W, p = 0.0, 1.0
        res.update({"test": "Wilcoxon signed-rank", "stat_name": "W", "stat_value": float(W), "df": np.nan, "p_value": float(p)})
    return res

def p_to_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "n.s."

def add_sig_bar(ax, x1, x2, y, text, h=0.02):
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.2, c="black")
    ax.text((x1+x2)/2, y+h+0.005, text, ha="center", va="bottom", fontsize=10)

def plot_psychometric(conditions_played, responses, mouse, color, output_dir, save=True):
    """
    Plots the psychometric (sigmoid) function based on the provided data and saves the plot.
    
    Parameters:
        conditions_played (list): List of condition codes.
        responses (list): List of responses corresponding to each condition.
        mouse (str): Identifier for the subject (used in the plot legend).
        color (str): Color for the plot lines.
        output_dir (str): Directory where the output files will be saved.
        save (bool): Whether to save the plot.
    
    Returns:
        float: Threshold for half-max detection rate.
    """
    # Map condition to actual stimulation frequency (Hz)
    frequency_map = {6: 0, 7: 15, 8: 10, 9: 7.5, 10: 5, 11: 3, 12: 1, 13: 0.5}
    response_dict = {freq: {'licked': 0, 'total': 0} for freq in frequency_map.values()}

    # Count responses for each frequency
    for condition, response in zip(conditions_played, responses):
        freq = frequency_map[condition]
        if response == 3 or response == 5:
            response_dict[freq]['licked'] += 1
        response_dict[freq]['total'] += 1

    # Calculate fractions and prepare data for fitting
    fractions_licked = {
        freq: res['licked'] / res['total'] if res['total'] > 0 else 0
        for freq, res in response_dict.items()
    }
    sorted_frequencies = sorted(fractions_licked.keys())
    sorted_fractions = [fractions_licked[freq] for freq in sorted_frequencies]
    x_data = np.array(sorted_frequencies)
    y_data = np.array(sorted_fractions)

    try:
        # Fit sigmoid curve
        p0 = [max(y_data), np.median(x_data), 1, min(y_data)]
        popt, pcov = curve_fit(sigmoid, x_data, y_data, p0, method='trf')
        
        # Generate smooth curve for plotting
        x_values = np.linspace(min(x_data), max(x_data), 100)
        y_values = sigmoid(x_values, *popt)
        y_fit = sigmoid(x_data, *popt)
        
        # Calculate goodness of fit
        residuals = y_data - y_fit
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        print(f"R^2: {r_squared:.2f}")
        
        n = len(x_data)
        k = len(popt)
        r_squared_adj = 1 - (1 - r_squared) * (n - 1) / (n - k - 1)
        print(f"Adjusted R^2: {r_squared_adj:.2f}")
        
        # Calculate threshold for half-max
        half_max = (np.max(y_values) - np.min(y_values)) / 2
        diff_value = np.min(y_values) + half_max
        threshold_for_half_max = x_values[np.where(1 - y_values >= diff_value)[0][0]]
        
        # Save R² values
        with open(join(output_dir, 'R^2.txt'), 'w') as f:
            f.write(f"R^2: {r_squared:.2f}\n")
            f.write(f"Adjusted R^2: {r_squared_adj:.2f}\n")

        # Plot fitted sigmoid curve
        plt.plot(x_values, (1 - y_values) * 100, label=mouse, color=color, linewidth=4)
        plt.axvline(x=popt[1], color=color, linestyle='--')
        plt.text(popt[1], 50, f'{popt[1]:.2f} Hz', color=color, ha='right', va='center')
        plt.axvline(x=threshold_for_half_max, color='black', linestyle='--')
        plt.text(threshold_for_half_max, 50, f'{threshold_for_half_max:.2f} Hz', 
                color='black', ha='left', va='center')

    except RuntimeError as e:
        print(f"Curve fitting failed due to: {e}")
        threshold_for_half_max = np.nan

    # Finalize plot
    plt.legend()
    plt.xlabel('Optogenetic Stimulation Frequency (Hz)')
    plt.ylabel('Detection Rate (%)')
    plt.xticks(x_data, labels=[f"{freq} Hz" for freq in x_data])
    plt.yticks(np.linspace(0, 100, 11))

    if save:
        _save_figure(join(output_dir, 'Psychophysics Plot.png'))
    plt.close()
    
    return threshold_for_half_max


def create_histogram_with_density(data, title, output_path, psyco_threshold):
    """Create histogram with KDE density overlay."""
    plt.figure(figsize=(10, 6))
    n, bins, patches = plt.hist(data, bins=20, density=True, color='gainsboro', 
                               alpha=0.7, label='Histogram')
    
    # Add KDE density line
    density = gaussian_kde(data)
    xs = np.linspace(0, 15, 100)
    density.covariance_factor = lambda: 0.1
    density._compute_covariance()
    plt.plot(xs, density(xs), color='black', label='Density (KDE)')
    plt.fill_between(xs, 0, density(xs), color='gray', alpha=0.2)
    
    # Add behavioral threshold line
    plt.axvline(x=psyco_threshold, color='red', linestyle='--')
    
    # Format plot
    plt.xticks([0, 0.5, 1, 3, 5, 7.5, 10, 15])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Density')
    plt.title(title)
    plt.legend()
    _save_figure(output_path)
    plt.close()


def create_bar_plot_with_individual_data(x_positions, avg_values, sem_values, individual_data, 
                                        labels, colors, ylabel, title, output_path):
    """Create bar plot with individual mouse data overlaid."""
    plt.figure(figsize=(6, 6))
    
    # Plot average bars with error bars
    for i, (pos, avg, sem, color, label) in enumerate(zip(x_positions, avg_values, sem_values, colors, labels)):
        plt.bar(pos, avg, width=0.4, color=color, alpha=0.5, label=label)
        plt.errorbar(pos, avg, yerr=sem, fmt='none', color='black', capsize=3)
    
    # Overlay individual mouse data with connecting lines
    num_mice = len(individual_data[0])
    for i in range(num_mice):
        mouse_values = [individual_data[j][i] for j in range(len(individual_data))]
        plt.plot(x_positions, mouse_values, marker='o', color='gray', 
                linestyle='--', markersize=6, alpha=0.3)
    
    plt.xticks(x_positions, labels)
    plt.ylabel(ylabel)
    plt.title(title)
    _save_figure(output_path)
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


# ===========================

def analyze_subset(subset_paths, behavior_paths, save_loc):
    mice = [m for m, d in MOUSE_SESSIONS]
    days = [d for m, d in MOUSE_SESSIONS]
    # Define frequency bins for histograms
    freq_centers = np.array([0, 0.5, 1, 3, 5, 7.5, 10, 15])  # Hz
    edges = np.concatenate(([freq_centers[0] - 0.25],
                            (freq_centers[:-1] + freq_centers[1:]) / 2,
                            [freq_centers[-1] + 2.5]))


    # ===========================
    # DATA CONTAINERS
    # ===========================
    # Overall data containers
    all_half_diff = []
    all_half_diff_gauss = []
    mouse_ids = []
    behavior_thresholds = []
    peak_values = []
    peak_values_gauss = []
    diff_per_neuron_per_mouse = []
    diff_per_neuron_gauss_per_mouse = []
    neurons_sigmoid_per_mouse = []
    neurons_gauss_per_mouse = []

    # Per-mouse percentage containers
    perc_above5_sigmoid_per_mouse = []
    perc_above5_gauss_per_mouse = []
    perc_below_thresh_sigmoid_per_mouse = []
    perc_below_thresh_gauss_per_mouse = []
    perc_above_thresh_sigmoid_per_mouse = []
    perc_above_thresh_gauss_per_mouse = []

    # Frequency bin containers for SEM calculations
    perc_sigmoid_bins_per_mouse = []
    perc_gauss_bins_per_mouse = []

    # Overall accumulators
    total_neurons_sigmoid = 0
    total_neurons_gauss = 0
    total_above5_sigmoid = 0
    total_above5_gauss = 0
    total_below_thresh_sigmoid = 0
    total_below_thresh_gauss = 0


    # ===========================
    # MAIN DATA PROCESSING LOOP
    # ===========================
    for mouse, day in zip(mice, days):
        print(f"Processing {mouse} - {day}")
        n_priming_trials = 23
        # Set up paths
        mouse_path = str(subset_paths[mouse])
        if mouse in ['mouse01', 'mouse05']:
            ROOT_PATH = str(behavior_paths[mouse])
        else:
            ROOT_PATH = str(behavior_paths[mouse])
    
        # Load half-difference values for sigmoid fit
        half_diff_file = os.path.join(mouse_path, 'half_diff_values_sigmoid.npy')
        if not os.path.exists(half_diff_file):
            print(f"File not found: {half_diff_file}")
            continue
        half_diff = np.load(half_diff_file)
    
        # Load half-difference values for gaussian fit
        half_diff_gauss_file = os.path.join(mouse_path, 'half_max_values_gaussian.npy')
        if not os.path.exists(half_diff_gauss_file):
            print(f"File not found: {half_diff_gauss_file}")
            continue
        half_diff_gauss = np.load(half_diff_gauss_file)
    
        # Load behavioral data
        conditions_played = np.load(join(ROOT_PATH, 'conditions_played.npy'))
        responses = np.load(join(ROOT_PATH, 'responses.npy'))

        # Remove priming trials from behavioral data
        conditions_played = conditions_played[n_priming_trials:]
        responses = responses[n_priming_trials:]

        # Aggregate overall data
        all_half_diff.extend(half_diff)
        all_half_diff_gauss.extend(half_diff_gauss)
        mouse_ids.append(mouse)
        neurons_sigmoid_per_mouse.append(len(half_diff))
        neurons_gauss_per_mouse.append(len(half_diff_gauss))

        # Calculate per-mouse frequency bin percentages
        hist_sig_mouse, _ = np.histogram(half_diff, bins=edges)
        hist_gauss_mouse, _ = np.histogram(half_diff_gauss, bins=edges)
        perc_sigmoid_bins_per_mouse.append(hist_sig_mouse / hist_sig_mouse.sum() * 100)
        perc_gauss_bins_per_mouse.append(hist_gauss_mouse / hist_gauss_mouse.sum() * 100)

        # Get behavioral detection threshold from psychometric fitting
        psyco = plot_psychometric(conditions_played, responses, mouse=mouse,
                                 color='blue', output_dir=_behavior_output(save_loc, mouse))
        behavior_thresholds.append(psyco)

        # calculate per mouse and neuron the difference between the behavioral threshold and the half-difference
        diff_per_neuron_per_mouse.extend(half_diff - psyco)
        diff_per_neuron_gauss_per_mouse.extend(half_diff_gauss - psyco)

        # Calculate histogram peaks (modes) for each fit
        counts, bin_edges = np.histogram(half_diff, bins=20)
        max_idx = np.argmax(counts)
        bin_center = (bin_edges[max_idx] + bin_edges[max_idx + 1]) / 2
        peak_values.append(bin_center)

        counts_gauss, bin_edges_gauss = np.histogram(half_diff_gauss, bins=20)
        max_idx_gauss = np.argmax(counts_gauss)
        bin_center_gauss = (bin_edges_gauss[max_idx_gauss] + bin_edges_gauss[max_idx_gauss + 1]) / 2
        peak_values_gauss.append(bin_center_gauss)

        # Calculate percentages for different thresholds
        perc_above5_sigmoid = (np.sum(half_diff >= 5) / len(half_diff)) * 100
        perc_above5_gauss = (np.sum(half_diff_gauss >= 5) / len(half_diff_gauss)) * 100
        perc_above5_sigmoid_per_mouse.append(perc_above5_sigmoid)
        perc_above5_gauss_per_mouse.append(perc_above5_gauss)

        perc_below_thresh_sigmoid = (np.sum(half_diff < psyco) / len(half_diff)) * 100
        perc_below_thresh_gauss = (np.sum(half_diff_gauss < psyco) / len(half_diff_gauss)) * 100
        perc_below_thresh_sigmoid_per_mouse.append(perc_below_thresh_sigmoid)
        perc_below_thresh_gauss_per_mouse.append(perc_below_thresh_gauss)

        perc_above_thresh_sigmoid = (np.sum(half_diff >= psyco) / len(half_diff)) * 100
        perc_above_thresh_gauss = (np.sum(half_diff_gauss >= psyco) / len(half_diff_gauss)) * 100
        perc_above_thresh_sigmoid_per_mouse.append(perc_above_thresh_sigmoid)
        perc_above_thresh_gauss_per_mouse.append(perc_above_thresh_gauss)

        # Update overall accumulators
        total_neurons_sigmoid += len(half_diff)
        total_neurons_gauss += len(half_diff_gauss)
        total_above5_sigmoid += np.sum(half_diff >= 5)
        total_above5_gauss += np.sum(half_diff_gauss >= 5)
        total_below_thresh_sigmoid += np.sum(half_diff < psyco)
        total_below_thresh_gauss += np.sum(half_diff_gauss < psyco)

        # Create individual mouse histogram plots
        overall_hist_path = os.path.join(save_loc, f'half_diff_histogram_density_{mouse}.pdf')
        create_histogram_with_density(half_diff, 'Half-Difference', overall_hist_path, psyco)
        print("number of neurons for sigmoid fit:", len(half_diff))
        print("Overall histogram and density plot saved to:", overall_hist_path)

        overall_hist_path_gauss = os.path.join(save_loc, f'half_diff_histogram_density_gaussian_{mouse}.pdf')
        create_histogram_with_density(half_diff_gauss, 'Half-Difference Gaussian', 
                                     overall_hist_path_gauss, psyco)
        print("number of neurons for gaussian fit:", len(half_diff_gauss))
        print("Overall histogram and density plot saved to:", overall_hist_path_gauss)


    # ===========================
    # CROSS-MOUSE ANALYSIS PLOTS
    # ===========================

    # 1. Percent-of-neurons-per-frequency: Sigmoid vs Gaussian (with SEM)
    perc_sigmoid_bins_per_mouse = np.stack(perc_sigmoid_bins_per_mouse)
    perc_gauss_bins_per_mouse = np.stack(perc_gauss_bins_per_mouse)

    mean_sig = perc_sigmoid_bins_per_mouse.mean(axis=0)
    sem_sig = st.sem(perc_sigmoid_bins_per_mouse, axis=0)
    mean_gauss = perc_gauss_bins_per_mouse.mean(axis=0)
    sem_gauss = st.sem(perc_gauss_bins_per_mouse, axis=0)

    x = np.arange(len(freq_centers))
    bar_w = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(x - bar_w/2, mean_sig, yerr=sem_sig, width=bar_w,
            color='gray', alpha=0.5, capsize=4, label='Sigmoid')
    plt.bar(x + bar_w/2, mean_gauss, yerr=sem_gauss, width=bar_w,
            color='tan', alpha=0.5, capsize=4, label='Gaussian')

    # Add individual mouse data as points
    for i in range(len(mouse_ids)):
        plt.scatter(x - bar_w/2, perc_sigmoid_bins_per_mouse[i], color='gray', alpha=0.3, s=20)
        plt.scatter(x + bar_w/2, perc_gauss_bins_per_mouse[i], color='tan', alpha=0.3, s=20)

    plt.xticks(x, [f'{f} Hz' for f in freq_centers])
    plt.ylabel('Percent of Neurons')
    plt.xlabel('Half-Difference Frequency Bin')
    plt.title('Percent of Neurons per Frequency (mean ± SEM)')
    plt.legend()
    plt.tight_layout()

    freq_percent_sem_path = os.path.join(save_loc, 'percent_neurons_per_frequency_sigmoid_vs_gaussian.pdf')
    _save_figure(freq_percent_sem_path)
    plt.close()
    print("Frequency-percent plot with SEM saved to:", freq_percent_sem_path)

    # 1.1. Percent of sigmoid vs gaussian across all mice (all frequencies together)
    # Calculate total percentage of sigmoid vs gaussian neurons for each mouse
    perc_sigmoid_per_mouse = []
    perc_gaussian_per_mouse = []
    all_mice_sigmoid_excitatory_fraction = []
    all_mice_sigmoid_inhibitory_fraction = []
    all_mice_gaussian_excitatory_fraction = []
    all_mice_gaussian_inhibitory_fraction = []

    # Calculate the percentage for each mouse (relative to total neurons analyzed)
    for i, mouse in enumerate(mouse_ids):
        # Get the number of neurons for this mouse
        if i < len(mice):
            # Load the data for this specific mouse to get counts
            mouse_path = str(subset_paths[mouse])
        
            # Load half-difference values for sigmoid and gaussian fits
            half_diff_file = os.path.join(mouse_path, 'half_diff_values_sigmoid.npy')
            half_diff_gauss_file = os.path.join(mouse_path, 'half_max_values_gaussian.npy')
            r_squared_file = os.path.join(mouse_path, 'r_squared_values.npy')
            # load csv stats file
            stats_file = os.path.join(mouse_path, 'excitatory_inhibitory_statistics.csv')
            stats_df = pd.read_csv(stats_file)
            # get the fraction of excitatory and inhibitory neurons for each category
            sigmoid_excitatory_fraction = stats_df[stats_df['Category'] == 'Sigmoid']['Excitatory_Fraction'].values[0]
            sigmoid_inhibitory_fraction = stats_df[stats_df['Category'] == 'Sigmoid']['Inhibitory_Fraction'].values[0]
            gaussian_excitatory_fraction = stats_df[stats_df['Category'] == 'Gaussian']['Excitatory_Fraction'].values[0]
            gaussian_inhibitory_fraction = stats_df[stats_df['Category'] == 'Gaussian']['Inhibitory_Fraction'].values[0]
            all_mice_sigmoid_excitatory_fraction.append(sigmoid_excitatory_fraction)
            all_mice_sigmoid_inhibitory_fraction.append(sigmoid_inhibitory_fraction)
            all_mice_gaussian_excitatory_fraction.append(gaussian_excitatory_fraction)
            all_mice_gaussian_inhibitory_fraction.append(gaussian_inhibitory_fraction)

            if (os.path.exists(half_diff_file) and os.path.exists(half_diff_gauss_file) 
                and os.path.exists(r_squared_file)):
                half_diff_sigmoid = np.load(half_diff_file)
                half_diff_gaussian = np.load(half_diff_gauss_file)
                r_squared_values = np.load(r_squared_file)
            
                # Total neurons analyzed is the length of r_squared_values array
                total_neurons_this_mouse = len(r_squared_values)
            
                perc_sigmoid_this_mouse = (len(half_diff_sigmoid) / total_neurons_this_mouse) * 100
                perc_gaussian_this_mouse = (len(half_diff_gaussian) / total_neurons_this_mouse) * 100
            
                perc_sigmoid_per_mouse.append(perc_sigmoid_this_mouse)
                perc_gaussian_per_mouse.append(perc_gaussian_this_mouse)
            
                print(f"{mouse}: Total neurons={total_neurons_this_mouse}, Sigmoid={len(half_diff_sigmoid)} ({perc_sigmoid_this_mouse:.1f}%), Gaussian={len(half_diff_gaussian)} ({perc_gaussian_this_mouse:.1f}%)")
            else:
                print(f"Warning: Missing files for {mouse}")
                continue

    # Create the total percentage plot with mean ± SEM, individual data points, and connecting lines
    plt.figure(figsize=(8, 6))

    # Calculate means and SEMs
    mean_sigmoid_total = np.mean(perc_sigmoid_per_mouse)
    mean_gaussian_total = np.mean(perc_gaussian_per_mouse)
    sem_sigmoid_total = st.sem(perc_sigmoid_per_mouse)
    sem_gaussian_total = st.sem(perc_gaussian_per_mouse)
    # calculate the total percentage of neurons that are either sigmoid or gaussian
    perc_all_per_mouse = [s + g for s, g in zip(perc_sigmoid_per_mouse, perc_gaussian_per_mouse)]
    mean_all_total = np.mean(perc_all_per_mouse)
    sem_all_total = st.sem(perc_all_per_mouse)
    print("Mean percentage of neurons that are either sigmoid or gaussian:", mean_all_total)
    print("SEM of percentage of neurons that are either sigmoid or gaussian:", sem_all_total)

    # Plot bars with error bars
    x_positions_total = [0, 1]
    colors_total = ['gray', 'tan']
    labels_total = ['Sigmoid', 'Gaussian']
    means_total = [mean_sigmoid_total, mean_gaussian_total]
    sems_total = [sem_sigmoid_total, sem_gaussian_total]

    # Create bars
    bars = plt.bar(x_positions_total, means_total, yerr=sems_total, 
                   color=colors_total, alpha=0.6, width=0.6, 
                   capsize=5, error_kw={'linewidth': 2})

    # Add individual data points with connecting lines
    for i in range(len(perc_sigmoid_per_mouse)):
        mouse_values = [perc_sigmoid_per_mouse[i], perc_gaussian_per_mouse[i]]
        plt.plot(x_positions_total, mouse_values, 'o-', color='black', 
                 alpha=0.4, linestyle='--', markersize=6, linewidth=1.5)

    # Add individual data points as scatter
    plt.scatter([0] * len(perc_sigmoid_per_mouse), perc_sigmoid_per_mouse, 
               color='darkgray', alpha=0.7, s=40, edgecolors='white', linewidth=1, zorder=3)
    plt.scatter([1] * len(perc_gaussian_per_mouse), perc_gaussian_per_mouse, 
               color='saddlebrown', alpha=0.7, s=40, edgecolors='white', linewidth=1, zorder=3)

    # Formatting
    plt.xticks(x_positions_total, labels_total)
    plt.ylabel('Percentage of Total Neurons (%)')
    plt.title('Percentage of Sigmoid vs Gaussian Neurons\n(Mean ± SEM across mice)')
    plt.ylim(0, max(max(means_total) + max(sems_total) + 10, 60))

    # Calculate and display the percentage of neurons that are neither sigmoid nor gaussian
    mean_neither = 100 - mean_sigmoid_total - mean_gaussian_total
    sem_neither = np.sqrt(sem_sigmoid_total**2 + sem_gaussian_total**2)  # Error propagation for difference

    # Add text annotations showing exact percentages
    for i, (pos, mean_val, sem_val) in enumerate(zip(x_positions_total, means_total, sems_total)):
        plt.text(pos, mean_val + sem_val + 2, f'{mean_val:.1f}% ± {sem_val:.1f}%', 
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Add text showing the percentage that are neither
    plt.text(0.5, max(means_total) + max(sems_total) + 6, 
             f'Neither: {mean_neither:.1f}% ± {sem_neither:.1f}%', 
             ha='center', va='bottom', fontsize=10, fontweight='bold', 
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.7))

    plt.tight_layout()

    # Save the plot
    total_percent_path = os.path.join(save_loc, 'total_percent_sigmoid_vs_gaussian_neurons.pdf')
    _save_figure(total_percent_path)
    plt.close()
    print("Total percentage plot saved to:", total_percent_path)
     # check significance between sigmoid and gaussian percentages
    p, test_label, info = choose_test(perc_sigmoid_per_mouse, perc_gaussian_per_mouse, paired=True, mode="wilcoxon")
    rep = info["report"]
    print(f"{rep['test']}: {rep['statistic_name']} = {rep['statistic_value']}, "
          f"df = {rep['df']}, p = {rep['p']}, "
          f"{rep['effect_size']['name']} = {rep['effect_size']['value']}")
    if rep["ci"]:
        print(f"95% CI for {rep['ci']['parameter']}: [{rep['ci']['low']:.3g}, {rep['ci']['high']:.3g}]")
    print(f"[stats] n={info['n']}  shapiro_p={info['shapiro_p']:.3g}  outliers={info['outliers']}  chosen={test_label}")


    # 1.2. Excitatory and inhibitory fraction of sigmoid vs gaussian neurons
    # --- summary stats ---
    mean_sigmoid_excitatory_fraction  = np.mean(all_mice_sigmoid_excitatory_fraction)
    mean_sigmoid_inhibitory_fraction  = np.mean(all_mice_sigmoid_inhibitory_fraction)
    mean_gaussian_excitatory_fraction = np.mean(all_mice_gaussian_excitatory_fraction)
    mean_gaussian_inhibitory_fraction = np.mean(all_mice_gaussian_inhibitory_fraction)

    sem_sigmoid_excitatory_fraction  = stats.sem(all_mice_sigmoid_excitatory_fraction, nan_policy="omit")
    sem_sigmoid_inhibitory_fraction  = stats.sem(all_mice_sigmoid_inhibitory_fraction, nan_policy="omit")
    sem_gaussian_excitatory_fraction = stats.sem(all_mice_gaussian_excitatory_fraction, nan_policy="omit")
    sem_gaussian_inhibitory_fraction = stats.sem(all_mice_gaussian_inhibitory_fraction, nan_policy="omit")

    means = np.array([
        [mean_sigmoid_excitatory_fraction,  mean_sigmoid_inhibitory_fraction],
        [mean_gaussian_excitatory_fraction, mean_gaussian_inhibitory_fraction]
    ])
    sems = np.array([
        [sem_sigmoid_excitatory_fraction,  sem_sigmoid_inhibitory_fraction],
        [sem_gaussian_excitatory_fraction, sem_gaussian_inhibitory_fraction]
    ])

    x_model = np.arange(2)  # 0: Sigmoid, 1: Gaussian
    width = 0.36

    plt.figure(figsize=(8, 6))
    bars_exc = plt.bar(
        x_model - width/2, means[:, 0], yerr=sems[:, 0],
        width=width, facecolor='lightgray', edgecolor='black', linewidth=1.5,
        capsize=5, error_kw={'linewidth': 1.5, 'capthick': 1.5}, label='Excitatory'
    )
    bars_inh = plt.bar(
        x_model + width/2, means[:, 1], yerr=sems[:, 1],
        width=width, facecolor='tan', edgecolor='black', linewidth=1.5,
        capsize=5, error_kw={'linewidth': 1.5, 'capthick': 1.5}, label='Inhibitory'
    )

    # Per-mouse lines
    for i in range(len(all_mice_sigmoid_excitatory_fraction)):
        xs_sig = [x_model[0] - width/2, x_model[0] + width/2]
        ys_sig = [all_mice_sigmoid_excitatory_fraction[i], all_mice_sigmoid_inhibitory_fraction[i]]
        plt.plot(xs_sig, ys_sig, linestyle='--', linewidth=1.2, color='gray', alpha=0.7, zorder=3)
        plt.scatter(xs_sig, ys_sig, s=40, facecolors='white', edgecolors='gray', linewidth=1.2, zorder=4)

    for i in range(len(all_mice_gaussian_excitatory_fraction)):
        xs_gau = [x_model[1] - width/2, x_model[1] + width/2]
        ys_gau = [all_mice_gaussian_excitatory_fraction[i], all_mice_gaussian_inhibitory_fraction[i]]
        plt.plot(xs_gau, ys_gau, linestyle='--', linewidth=1.2, color='gray', alpha=0.7, zorder=3)
        plt.scatter(xs_gau, ys_gau, s=40, facecolors='white', edgecolors='gray', linewidth=1.2, zorder=4)

    plt.xticks(x_model, ['Sigmoid', 'Gaussian'])
    plt.ylabel('Fraction of Neurons')
    plt.title('Excitatory and Inhibitory Fractions per Model')
    plt.ylim(0, 1)
    plt.legend(frameon=False)
    ax = plt.gca()

    # --- 1) Within-family paired tests: Excitation vs Inhibition ---
    res_sig = _choose_paired_test(all_mice_sigmoid_excitatory_fraction,
                                  all_mice_sigmoid_inhibitory_fraction)
    res_gau = _choose_paired_test(all_mice_gaussian_excitatory_fraction,
                                  all_mice_gaussian_inhibitory_fraction)

    # annotate on plot
    y_sig = max(means[0, 0] + sems[0, 0], means[0, 1] + sems[0, 1]) + 0.03
    y_gau = max(means[1, 0] + sems[1, 0], means[1, 1] + sems[1, 1]) + 0.03
    add_sig_bar(ax, x_model[0] - width/2, x_model[0] + width/2,
                y_sig, f"p={res_sig['p_value']:.3f} {p_to_stars(res_sig['p_value'])}")
    add_sig_bar(ax, x_model[1] - width/2, x_model[1] + width/2,
                y_gau, f"p={res_gau['p_value']:.3f} {p_to_stars(res_gau['p_value'])}")

    # --- 2) Two-way repeated-measures ANOVA: Family x Condition ---
    # Build long DataFrame
    records = []
    n = len(all_mice_sigmoid_excitatory_fraction)
    for i in range(n):
        records += [
            {"subject": i, "Family": "Sigmoid",  "Condition": "Excitatory",
             "Value": float(all_mice_sigmoid_excitatory_fraction[i])},
            {"subject": i, "Family": "Sigmoid",  "Condition": "Inhibitory",
             "Value": float(all_mice_sigmoid_inhibitory_fraction[i])},
            {"subject": i, "Family": "Gaussian", "Condition": "Excitatory",
             "Value": float(all_mice_gaussian_excitatory_fraction[i])},
            {"subject": i, "Family": "Gaussian", "Condition": "Inhibitory",
             "Value": float(all_mice_gaussian_inhibitory_fraction[i])},
        ]
    df_long = pd.DataFrame.from_records(records).dropna(subset=["Value"])

    # statsmodels AnovaRM
    anova = AnovaRM(df_long, depvar="Value", subject="subject",
                    within=["Family", "Condition"]).fit()
    anova_tbl = anova.anova_table.reset_index().rename(columns={"index": "Effect"})

    # Extract standard columns if present
    def _get_col(tbl, candidates):
        for c in candidates:
            if c in tbl.columns: return c
        raise KeyError("Expected column not found in AnovaRM table")
    col_F  = _get_col(anova_tbl, ["F Value", "F", "F Value Num"])
    col_p  = _get_col(anova_tbl, ["Pr > F", "Pr(>F)", "Pr > F "])
    col_df1 = _get_col(anova_tbl, ["Num DF", "df1"])
    col_df2 = _get_col(anova_tbl, ["Den DF", "df2"])

    # Prepare Excel rows
    rows = []
    rows.append({
        "Analysis": "Within-family (Sigmoid) Exc vs Inh",
        "Test": res_sig["test"], "Stat name": res_sig["stat_name"], "Stat": res_sig["stat_value"],
        "df": res_sig["df"], "p": res_sig["p_value"],
        "Shapiro W": res_sig["shapiro_W"], "Shapiro p": res_sig["shapiro_p"], "n": res_sig["n"]
    })
    rows.append({
        "Analysis": "Within-family (Gaussian) Exc vs Inh",
        "Test": res_gau["test"], "Stat name": res_gau["stat_name"], "Stat": res_gau["stat_value"],
        "df": res_gau["df"], "p": res_gau["p_value"],
        "Shapiro W": res_gau["shapiro_W"], "Shapiro p": res_gau["shapiro_p"], "n": res_gau["n"]
    })

    for eff in ["Family", "Condition", "Family:Condition"]:
        r = anova_tbl.loc[anova_tbl["Effect"] == eff].iloc[0]
        rows.append({
            "Analysis": f"RM-ANOVA {eff}",
            "Test": "AnovaRM", "Stat name": "F", "Stat": float(r[col_F]),
            "df": f"{int(r[col_df1])}, {int(r[col_df2])}", "p": float(r[col_p]),
            "Shapiro W": np.nan, "Shapiro p": np.nan, "n": df_long["subject"].nunique()
        })

    # Save Excel
    os.makedirs(save_loc, exist_ok=True)
    excel_path = os.path.join(save_loc, "excit_inhib_sigmoid_vs_gaussian_stats.xlsx")
    pd.DataFrame(rows).to_excel(excel_path, index=False)
    print("Saved stats to:", excel_path)

    # Finalize plot and save as before
    plt.tight_layout()
    excitatory_inhibitory_path = os.path.join(save_loc, 'excitatory_inhibitory_fraction_sigmoid_vs_gaussian.pdf')
    _save_figure(excitatory_inhibitory_path)
    plt.close()
    print("Excitatory and inhibitory fraction plot saved to:", excitatory_inhibitory_path)

    # Optional: console print of ANOVA
    print("\nRepeated-measures ANOVA (Family x Condition):")
    print(anova_tbl)
    print("\nWithin-family paired tests:")
    print("Sigmoid:", res_sig)
    print("Gaussian:", res_gau)


    # 2. Pooled frequency analysis (no SEM)
    sig_vals = np.asarray(all_half_diff)
    gauss_vals = np.asarray(all_half_diff_gauss)
    print("Total neurons pooled - Sigmoid:", len(sig_vals), "Gaussian:", len(gauss_vals))

    hist_sig_all, _ = np.histogram(sig_vals, bins=edges)
    hist_gauss_all, _ = np.histogram(gauss_vals, bins=edges)

    percent_sig_all = hist_sig_all / hist_sig_all.sum() * 100
    percent_gauss_all = hist_gauss_all / hist_gauss_all.sum() * 100

    x_freq = np.arange(len(freq_centers))
    bar_w_freq = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(x_freq - bar_w_freq/2, percent_sig_all, width=bar_w_freq,
            color='gray', alpha=0.5, label='Sigmoid')
    plt.bar(x_freq + bar_w_freq/2, percent_gauss_all, width=bar_w_freq,
            color='tan', alpha=0.5, label='Gaussian')

    total_sigmoid_neurons = len(sig_vals)
    total_gauss_neurons = len(gauss_vals)
    mean_sigmoid_neurons = float(np.mean(neurons_sigmoid_per_mouse)) if neurons_sigmoid_per_mouse else float("nan")
    mean_gauss_neurons = float(np.mean(neurons_gauss_per_mouse)) if neurons_gauss_per_mouse else float("nan")
    sem_sigmoid_neurons = float(stats.sem(neurons_sigmoid_per_mouse)) if len(neurons_sigmoid_per_mouse) > 1 else float("nan")
    sem_gauss_neurons = float(stats.sem(neurons_gauss_per_mouse)) if len(neurons_gauss_per_mouse) > 1 else float("nan")

    fig = plt.gcf()
    summary_text = (
        f"Sigmoid: n={total_sigmoid_neurons} neurons, "
        f"{mean_sigmoid_neurons:.1f} ± {sem_sigmoid_neurons:.1f} neurons per mouse; "
        f"Gaussian: n={total_gauss_neurons} neurons, "
        f"{mean_gauss_neurons:.1f} ± {sem_gauss_neurons:.1f} neurons per mouse; "
        f"from {len(mouse_ids)} mice"
    )

    plt.xticks(x_freq, [f'{f} Hz' for f in freq_centers])
    plt.ylabel('Percent of Neurons')
    plt.xlabel('Half-Difference Frequency Bin')
    plt.title('Percent of Neurons per Frequency (pooled across mice)')
    plt.legend()
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    fig.text(0.5, 0.01, summary_text, ha='center', va='bottom', fontsize=9)

    freq_percent_pooled_path = os.path.join(save_loc, 'percent_neurons_per_frequency_sigmoid_vs_gaussian_pooled.pdf')
    _save_figure(freq_percent_pooled_path)
    plt.close()
    print("Frequency-percent plot (pooled) saved to:", freq_percent_pooled_path)
    print(summary_text)

    # make a overlay histograms of the diff_per_neuron_per_mouse and diff_per_neuron_gauss_per_mouse
    plt.figure(figsize=(10, 6))
    plt.hist(diff_per_neuron_per_mouse, bins=20, density=True, color='gray', alpha=0.5, label='Sigmoid')
    plt.hist(diff_per_neuron_gauss_per_mouse, bins=20, density=True, color='tan', alpha=0.5, label='Gaussian')
    # add KDE density lines
    density = gaussian_kde(diff_per_neuron_per_mouse)
    xs = np.linspace(min(diff_per_neuron_per_mouse), max(diff_per_neuron_per_mouse), 100)
    density.covariance_factor = lambda: 0.1
    density._compute_covariance()
    plt.plot(xs, density(xs), color='gray', label='Density (KDE)')
    plt.fill_between(xs, 0, density(xs), color='gray', alpha=0.2)
    density = gaussian_kde(diff_per_neuron_gauss_per_mouse)
    xs = np.linspace(min(diff_per_neuron_gauss_per_mouse), max(diff_per_neuron_gauss_per_mouse), 100)
    density.covariance_factor = lambda: 0.1
    density._compute_covariance()
    plt.plot(xs, density(xs), color='tan', label='Density (KDE)')
    plt.fill_between(xs, 0, density(xs), color='tan', alpha=0.2)
    plt.xlabel('Difference between Half-Difference and Behavioral Threshold')
    plt.ylabel('Density')
    plt.title('Difference between Half-Difference and Behavioral Threshold')
    plt.legend()
    _save_figure(os.path.join(save_loc, 'diff_per_neuron_sigmoid_vs_gauss_histogram.pdf'))
    plt.close()
    print("Diff per neuron histogram saved to:", os.path.join(save_loc, 'diff_per_neuron_sigmoid_vs_gauss_histogram.pdf'))

    # 3. Scatter Plot: Behavioral threshold vs. Histogram peak per mouse
    plt.figure(figsize=(8, 6))
    plt.scatter(behavior_thresholds, peak_values, color='black')
    plt.scatter(behavior_thresholds, peak_values_gauss, color='red')
    plt.ylim(0, max(peak_values) + 1)
    plt.xlim(0, max(behavior_thresholds) + 1)
    plt.xlabel('Detection Threshold (Behavior)')
    plt.ylabel('Peak of Half-Difference Histogram')
    plt.title('Behavioral Threshold vs. Half-Difference Peak per Mouse')
    plt.legend(['Sigmoid', 'Gaussian'])
    scatter_plot_path = os.path.join(save_loc, 'behavior_vs_half_diff_peak_scatter.pdf')
    _save_figure(scatter_plot_path)
    plt.close()
    print("Scatter plot saved to:", scatter_plot_path)

    # make a bars plot of: mean values with individual data points connected with lines
    plt.figure(figsize=(8, 6))

    # Calculate means and standard errors
    mean_sigmoid = np.mean(peak_values)
    mean_gaussian = np.mean(peak_values_gauss)
    mean_behavioral = np.mean(behavior_thresholds)

    sem_sigmoid = st.sem(peak_values)
    sem_gaussian = st.sem(peak_values_gauss)
    sem_behavioral = st.sem(behavior_thresholds)

    # Calculate standard deviations to show diversity
    std_sigmoid = np.std(peak_values)
    std_gaussian = np.std(peak_values_gauss)
    std_behavioral = np.std(behavior_thresholds)

    # Create bar plot with 3 mean bars and error bars showing SEM
    x_bars = np.arange(3)
    means = [mean_behavioral, mean_sigmoid, mean_gaussian]
    sems = [sem_behavioral, sem_sigmoid, sem_gaussian]
    colors = ['black', 'gray', 'tan']
    labels = ['Behavioral Threshold', 'Sigmoid', 'Gaussian']

    plt.bar(x_bars, means, yerr=sems, color=colors, alpha=0.5, width=0.6,
            capsize=5, error_kw={'linewidth': 2, 'alpha': 0.8})

    # Add individual mouse data points and connect with dashed lines
    for i in range(len(mouse_ids)):
        values = [behavior_thresholds[i], peak_values[i], peak_values_gauss[i]]
        plt.plot(x_bars, values, 'o-', color='black', alpha=0.4, linestyle='--', markersize=4)

    # Add text annotations showing standard deviation (diversity measure)
    for i, (label, std_val) in enumerate(zip(labels, [std_behavioral, std_sigmoid, std_gaussian])):
        plt.text(x_bars[i], means[i] + sems[i] + 0.5, f'SD: {std_val:.2f}',
                 ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.ylabel('Peak of Half-Difference Histogram')
    # plt.title('Mean Peak Values with Individual Mouse Data')
    plt.xticks(x_bars, labels)
    _save_figure(os.path.join(save_loc, 'peak_half_diff_histogram_per_mouse_bars.pdf'))
    plt.close()
    print("Peak half-diff histogram per mouse saved to:", os.path.join(save_loc, 'peak_half_diff_histogram_per_mouse_bars.pdf'))

    # Additional plot to specifically show the spread/diversity differences with dual y-axis
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Separate the data: behavioral threshold vs. peak values
    behavioral_data = [behavior_thresholds]
    peak_data = [peak_values, peak_values_gauss]
    behavioral_pos = [1]
    peak_pos = [2, 3]

    # Plot behavioral threshold on left y-axis (ax1)
    parts1 = ax1.violinplot(behavioral_data, positions=behavioral_pos, widths=0.6, showmeans=False, showmedians=False)
    parts1['bodies'][0].set_facecolor('black')
    parts1['bodies'][0].set_alpha(0.6)

    # Remove blue elements for behavioral threshold
    for partname in ('cbars', 'cmins', 'cmaxes'):
        if partname in parts1:
            parts1[partname].set_color('none')
            parts1[partname].set_alpha(0)

    # Add behavioral threshold data points
    ax1.scatter([1] * len(behavior_thresholds), behavior_thresholds, alpha=0.7, s=30, color='black', edgecolors='white', linewidth=0.5)

    # Set up left y-axis
    ax1.set_ylabel('Behavioral Threshold', color='black', fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='black')

    # Create second y-axis for peak values
    ax2 = ax1.twinx()

    # Plot peak values on right y-axis (ax2)
    parts2 = ax2.violinplot(peak_data, positions=peak_pos, widths=0.6, showmeans=False, showmedians=False)
    colors_peaks = ['gray', 'tan']
    for pc, color in zip(parts2['bodies'], colors_peaks):
        pc.set_facecolor(color)
        pc.set_alpha(0.6)

    # Remove blue elements for peak values
    for partname in ('cbars', 'cmins', 'cmaxes'):
        if partname in parts2:
            parts2[partname].set_color('none')
            parts2[partname].set_alpha(0)

    # Add peak data points
    for i, (data, pos, color) in enumerate(zip(peak_data, peak_pos, colors_peaks)):
        ax2.scatter([pos] * len(data), data, alpha=0.7, s=30, color=color, edgecolors='white', linewidth=0.5)

    # Set up right y-axis
    ax2.set_ylabel('Peak of Half-Difference Histogram', color='gray', fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='gray')


    # Connect individual mice across all measures with gray dashed lines
    # Use a transform approach to properly connect across different axes
    for i in range(len(mouse_ids)):
        behav_val = behavior_thresholds[i]
        sigmoid_val = peak_values[i]
        gauss_val = peak_values_gauss[i]

        # Create a continuous line by plotting on a shared coordinate system
        # connect the points through display coordinates

        # Get the y-coordinates in display coordinates for proper connection
        # Transform behavioral threshold coordinate
        behav_display = ax1.transData.transform([(1, behav_val)])[0]
        sigmoid_display = ax2.transData.transform([(2, sigmoid_val)])[0]
        gauss_display = ax2.transData.transform([(3, gauss_val)])[0]

        # Convert back to data coordinates for a unified line
        # everything is plotted on ax1, with the right-axis values scaled to fit

        # Normalize the right-axis values to fit in left-axis range
        sigmoid_norm = sigmoid_val
        gauss_norm = gauss_val

        # Create the connecting line using ax1 coordinate system
        # Scale sigmoid and gaussian values to be visually connectable
        behav_range = max(behavior_thresholds) - min(behavior_thresholds)
        peak_range = max(max(peak_values), max(peak_values_gauss)) - min(min(peak_values), min(peak_values_gauss))

        if peak_range > 0 and behav_range > 0:
            # Scale peak values to behavioral threshold range for visual connection
            sigmoid_scaled = min(behavior_thresholds) + (sigmoid_val - min(min(peak_values), min(peak_values_gauss))) * behav_range / peak_range
            gauss_scaled = min(behavior_thresholds) + (gauss_val - min(min(peak_values), min(peak_values_gauss))) * behav_range / peak_range

            # Plot the full connecting line on ax1 with scaled values
            ax1.plot([1, 2, 3], [behav_val, sigmoid_scaled, gauss_scaled], 'o-',
                    color='gray', alpha=0.3, linestyle='--', markersize=2, linewidth=1)

    # Add CV annotations for each measure
    positions = [1, 2, 3]
    all_data = [behavior_thresholds, peak_values, peak_values_gauss]
    labels = ['Behavior', 'Sigmoid', 'Gaussian']

    # Add CV annotations
    for i, (data, pos, label) in enumerate(zip(all_data, positions, labels)):
        cv = np.std(data) / np.mean(data) * 100
        if i == 0:  # Behavioral threshold
            y_pos = max(behavior_thresholds) + 0.3
            ax1.text(pos, y_pos, f'CV: {cv:.1f}%', ha='center', va='bottom',
                    fontsize=9, fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        else:  # Peak values
            y_pos = max(data) + 0.3
            ax2.text(pos, y_pos, f'CV: {cv:.1f}%', ha='center', va='bottom',
                    fontsize=9, fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # Set x-axis with color-coded tick labels
    ax1.set_xticks(positions)
    ax1.set_xticklabels(labels)

    # Color the x-axis tick labels to match their corresponding y-axis
    tick_colors = ['black', 'gray', 'gray']  # Behavioral Threshold (left axis), Sigmoid & Gaussian (right axis)
    for tick, color in zip(ax1.get_xticklabels(), tick_colors):
        tick.set_color(color)
        tick.set_fontweight('bold')

    # Remove grid and spines
    ax1.grid(False)
    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)

    plt.tight_layout()
    _save_figure(os.path.join(save_loc, 'spread_comparison_violin_plot.pdf'))
    plt.close()
    print("Spread comparison violin plot saved to:", os.path.join(save_loc, 'spread_comparison_violin_plot.pdf'))


    # 4. Scatter plot: Percent of neurons (>= 5 Hz) vs. Detection Threshold
    plt.figure(figsize=(8, 6))
    plt.scatter(behavior_thresholds, perc_above5_sigmoid_per_mouse, color='black', label='Sigmoid >= 5 Hz')
    plt.scatter(behavior_thresholds, perc_above5_gauss_per_mouse, color='red', label='Gaussian >= 5 Hz')
    plt.xlabel('Detection Threshold (Behavior)')
    plt.ylabel('Percent of Neurons >= 5 Hz')
    plt.title('Percent of Neurons >= 5 Hz vs. Detection Threshold')
    plt.legend()
    scatter_above5_path = os.path.join(save_loc, 'percent_neurons_above5_vs_detection_threshold.pdf')
    _save_figure(scatter_above5_path)
    plt.close()
    print("Scatter plot (>= 5 Hz) saved to:", scatter_above5_path)


    # ===========================
    # BAR PLOTS FOR DIFFERENT THRESHOLDS
    # ===========================
    x_positions = [0, 1]  # 0 for Sigmoid, 1 for Gaussian

    # 5. Bar plot for percent of neurons >= 5 Hz
    avg_above5_sigmoid = np.mean(perc_above5_sigmoid_per_mouse)
    avg_above5_gauss = np.mean(perc_above5_gauss_per_mouse)
    sem_above5_sigmoid = st.sem(perc_above5_sigmoid_per_mouse)
    sem_above5_gauss = st.sem(perc_above5_gauss_per_mouse)

    create_bar_plot_with_individual_data(
        x_positions, [avg_above5_sigmoid, avg_above5_gauss],
        [sem_above5_sigmoid, sem_above5_gauss],
        [perc_above5_sigmoid_per_mouse, perc_above5_gauss_per_mouse],
        ['Sigmoid', 'Gaussian'], ['gray', 'tan'],
        'Percent of Neurons >= 5 Hz', 'Percent of Neurons >= 5 Hz (per Mouse)',
        os.path.join(save_loc, 'bar_percent_neurons_above5.pdf')
    )
    print("Bar plot (>= 5 Hz) saved")


    # 6. Bar plot for percent of neurons < Detection Threshold
    avg_below_thresh_sigmoid = np.mean(perc_below_thresh_sigmoid_per_mouse)
    avg_below_thresh_gauss = np.mean(perc_below_thresh_gauss_per_mouse)
    sem_below_thresh_sigmoid = st.sem(perc_below_thresh_sigmoid_per_mouse)
    sem_below_thresh_gauss = st.sem(perc_below_thresh_gauss_per_mouse)

    create_bar_plot_with_individual_data(
        x_positions, [avg_below_thresh_sigmoid, avg_below_thresh_gauss],
        [sem_below_thresh_sigmoid, sem_below_thresh_gauss],
        [perc_below_thresh_sigmoid_per_mouse, perc_below_thresh_gauss_per_mouse],
        ['Sigmoid', 'Gaussian'], ['gray', 'tan'],
        'Percent of Neurons < Detection Threshold', 'Percent of Neurons Below Detection Threshold (per Mouse)',
        os.path.join(save_loc, 'bar_percent_neurons_below_detection_threshold.pdf')
    )
    print("Bar plot (< detection threshold) saved")


    # 7. Bar plot for percent of neurons >= Detection Threshold
    avg_above_thresh_sigmoid = np.mean(perc_above_thresh_sigmoid_per_mouse)
    avg_above_thresh_gauss = np.mean(perc_above_thresh_gauss_per_mouse)
    sem_above_thresh_sigmoid = st.sem(perc_above_thresh_sigmoid_per_mouse)
    sem_above_thresh_gauss = st.sem(perc_above_thresh_gauss_per_mouse)

    create_bar_plot_with_individual_data(
        x_positions, [avg_above_thresh_sigmoid, avg_above_thresh_gauss],
        [sem_above_thresh_sigmoid, sem_above_thresh_gauss],
        [perc_above_thresh_sigmoid_per_mouse, perc_above_thresh_gauss_per_mouse],
        ['Sigmoid', 'Gaussian'], ['gray', 'tan'],
        'Percent of Neurons >= Detection Threshold', 'Percent of Neurons Above Detection Threshold (per Mouse)',
        os.path.join(save_loc, 'bar_percent_neurons_above_detection_threshold.pdf')
    )
    print("Bar plot (>= detection threshold) saved")

    print("All Analysis completed successfully!")

    return locals()

def _behavior_output(save_loc,mouse):
    path=Path(save_loc)/'behavior'/mouse
    path.mkdir(parents=True,exist_ok=True)
    return str(path)


def _require(condition,message):
    if not condition:
        raise ValueError(message)


def _save_figure(filename,*args,**kwargs):
    """Save the original figure and record its actual plotted artists."""
    fig=plt.gcf(); fig.canvas.draw()
    plot=Path(filename).relative_to(_ACTIVE_OUTPUT).as_posix()
    rows=[]
    for axis_no,ax in enumerate(fig.axes):
        for i,bar in enumerate(ax.patches):
            if hasattr(bar,'get_width') and hasattr(bar,'get_height'):
                rows.append([plot,axis_no,'bar',i,bar.get_label(),0,bar.get_x(),bar.get_y(),
                             bar.get_x()+bar.get_width(),bar.get_y()+bar.get_height(),'data'])
        for i,line in enumerate(ax.lines):
            coords=line.get_xydata()
            system='data' if line.get_transform()==ax.transData else 'artist (blended axes/data)'
            for j,(x,y) in enumerate(coords):
                rows.append([plot,axis_no,'line',i,line.get_label(),j,x,y,None,None,system])
        for i,collection in enumerate(ax.collections):
            label=collection.get_label()
            if isinstance(collection,PathCollection):
                for j,(x,y) in enumerate(collection.get_offsets()):
                    rows.append([plot,axis_no,'scatter',i,label,j,x,y,None,None,'data'])
            elif isinstance(collection,LineCollection):
                for j,segment in enumerate(collection.get_segments()):
                    for k,(x,y) in enumerate(segment):
                        rows.append([plot,axis_no,'error/segment',i,label,str(j)+':'+str(k),x,y,None,None,'data'])
            elif isinstance(collection,PolyCollection):
                for j,poly in enumerate(collection.get_paths()):
                    for k,(x,y) in enumerate(poly.vertices):
                        rows.append([plot,axis_no,'fill/violin',i,label,str(j)+':'+str(k),x,y,None,None,'data'])
    _ACTIVE_CAPTURE.extend(rows)
    fig.savefig(filename,*args,**kwargs)
    if Path(filename).suffix.lower()=='.pdf':
        fig.savefig(Path(filename).with_suffix('.png'),dpi=160)


def _paths(root,subset):
    neural={}; behavior={}
    for mouse,day in MOUSE_SESSIONS:
        base=Path(root)/mouse/day/'ML'
        neural[mouse]=base/'data/processed_data/trial_history_neurometric'/subset
        behavior[mouse]=base/('data2' if mouse in ['mouse01','mouse05'] else 'data')
    return neural,behavior


def _preflight(root,subsets):
    _require(len(MOUSE_IDS)>=2 and len(set(MOUSE_IDS))==len(MOUSE_IDS),'At least two unique mice are required.')
    _require(len(subsets)>0 and len(set(subsets))==len(subsets) and
             set(subsets).issubset({'after_hit','not_after_hit'}),'Invalid or duplicate SUBSETS.')
    missing=[]
    for subset in subsets:
        neural,behavior=_paths(root,subset)
        for mouse in MOUSE_IDS:
            for name in ['half_diff_values_sigmoid.npy','half_max_values_gaussian.npy','r_squared_values.npy',
                         'excitatory_inhibitory_statistics.csv']:
                if not (neural[mouse]/name).is_file(): missing.append(str(neural[mouse]/name))
            for name in ['conditions_played.npy','responses.npy']:
                if not (behavior[mouse]/name).is_file(): missing.append(str(behavior[mouse]/name))
    if missing:
        raise FileNotFoundError('Required original inputs are missing. No mice were dropped.\n'+'\n'.join(dict.fromkeys(missing)))


def _snapshot_and_validate(neural,behavior,out):
    manifest=[]; arrays={}; frames={}; behavior_arrays={}
    snap=out/'input_snapshots';snap.mkdir(parents=True,exist_ok=True)
    number=0
    for mouse,day in MOUSE_SESSIONS:
        arrays[mouse]={}; behavior_arrays[mouse]={}
        inputs=[(neural[mouse]/name,'neural') for name in [
            'half_diff_values_sigmoid.npy','half_max_values_gaussian.npy','r_squared_values.npy',
            'excitatory_inhibitory_statistics.csv']]
        inputs += [(behavior[mouse]/name,'behavior') for name in ['conditions_played.npy','responses.npy']]
        for path,kind in inputs:
            raw=path.read_bytes();number+=1
            relative='input_snapshots/f%04d%s'%(number,path.suffix)
            (out/relative).write_bytes(raw)
            digest=hashlib.sha256(raw).hexdigest()
            manifest.append([mouse,day,kind,path.name,str(path),relative,digest])
            if path.suffix=='.npy':
                value=np.load(io.BytesIO(raw),allow_pickle=False)
                _require(value.ndim==1,mouse+': expected a 1D array in '+path.name)
                (arrays if kind=='neural' else behavior_arrays)[mouse][path.name]=value
            else:
                frames[mouse]=pd.read_csv(io.BytesIO(raw))
        total=len(arrays[mouse]['r_squared_values.npy'])
        ns=0
        for name in ['half_diff_values_sigmoid.npy','half_max_values_gaussian.npy']:
            values=arrays[mouse][name];ns+=len(values)
            _require(len(values)>1 and np.all(np.isfinite(values)) and np.ptp(values)>0,
                     mouse+': original full KDE Analysis requires at least two finite, varying frequencies in '+name)
            _require(np.histogram(values,bins=np.array([-.25,.25,.75,2,4,6.25,8.75,12.5,17.5]))[0].sum()>0,
                     mouse+': no frequencies in original histogram bins.')
        _require(total>=ns,mouse+': fit category counts exceed total analyzed neurons.')
        frame=frames[mouse]
        for category in ['Sigmoid','Gaussian']:
            selected=frame.loc[frame['Category']==category]
            _require(len(selected)==1,mouse+': category missing or duplicated in statistics CSV.')
            vals=selected[['Excitatory_Fraction','Inhibitory_Fraction']].to_numpy(dtype=float)
            _require(np.all(np.isfinite(vals)) and np.all((vals>=0)&(vals<=1)),mouse+': invalid source fractions.')
        c=behavior_arrays[mouse]['conditions_played.npy'];r=behavior_arrays[mouse]['responses.npy']
        _require(len(c)==len(r) and len(c)>23,mouse+': behavior trial lengths do not agree or no post-priming trials.')
        _require(set(c[23:]).issubset({6,7,8,9,10,11,12,13}),mouse+': unrecognized behavioral condition code.')
    return manifest,arrays,frames,behavior_arrays


def _value(v):
    if isinstance(v,np.generic):v=v.item()
    if isinstance(v,float) and not np.isfinite(v):return None
    return v


def _table(wb,name,note,headers,rows):
    ws=wb.create_sheet(name)
    ws.append([name.replace('_',' ')]);ws.append([note]);ws.append([]);ws.append(headers)
    ws.row_dimensions[2].height=48
    ws['A2'].alignment=Alignment(wrap_text=True,vertical='top')
    ws.merge_cells(start_row=2,start_column=1,end_row=2,end_column=len(headers))
    ws['A1'].font=Font(name='Arial',size=14,bold=True)
    for row in rows:
        ws.append([_value(v) for v in row])
    for c in ws[4]:
        c.font=Font(name='Arial',size=10,bold=True,color='FFFFFF');c.fill=PatternFill('solid',fgColor='484355')
        c.alignment=Alignment(wrap_text=True,vertical='center')
    ws.row_dimensions[4].height=42
    for j,header in enumerate(headers,1):
        ws.column_dimensions[get_column_letter(j)].width=min(32,max(15,len(str(header))+2))
    ws.freeze_panes='B5';ws.auto_filter.ref='A4:%s%d'%(get_column_letter(len(headers)),ws.max_row)
    ws.sheet_view.showGridLines=False
    for row in ws.iter_rows(min_row=5):
        for c in row:
            if isinstance(c.value,float):c.number_format='0.000000'
    return ws


def _build_tables(state,arrays,frames,behavior_arrays,capture,manifest):
    s=state; ids=s['mouse_ids'];centers=s['freq_centers'];edges=s['edges']
    _require(ids==MOUSE_IDS,'Loaded mouse order differs from requested IDs.')
    summary=[['Sigmoid',s['mean_sigmoid_total'],s['sem_sigmoid_total'],len(ids)],
             ['Gaussian',s['mean_gaussian_total'],s['sem_gaussian_total'],len(ids)]]
    pooled=[[f,edges[i],edges[i+1],int(s['hist_sig_all'][i]),int(s['hist_gauss_all'][i]),
             s['percent_sig_all'][i],s['percent_gauss_all'][i]] for i,f in enumerate(centers)]
    frequency=[]
    for fit,mean,sem,individual in [('Sigmoid','mean_sig','sem_sig','perc_sigmoid_bins_per_mouse'),
                                   ('Gaussian','mean_gauss','sem_gauss','perc_gauss_bins_per_mouse')]:
        for i,f in enumerate(centers):
            frequency.append([fit,f,edges[i],edges[i+1],s[mean][i],s[sem][i],len(ids)]+list(s[individual][:,i]))
    mouse_rows=[];neuron_rows=[];behavior_rows=[]
    extra=['peak_values','peak_values_gauss','perc_above5_sigmoid_per_mouse','perc_above5_gauss_per_mouse',
           'perc_below_thresh_sigmoid_per_mouse','perc_below_thresh_gauss_per_mouse',
           'perc_above_thresh_sigmoid_per_mouse','perc_above_thresh_gauss_per_mouse',
           'all_mice_sigmoid_excitatory_fraction','all_mice_sigmoid_inhibitory_fraction',
           'all_mice_gaussian_excitatory_fraction','all_mice_gaussian_inhibitory_fraction']
    for i,mouse in enumerate(ids):
        a=arrays[mouse];total=len(a['r_squared_values.npy'])
        sig=a['half_diff_values_sigmoid.npy'];gau=a['half_max_values_gaussian.npy']
        np.testing.assert_allclose([100*len(sig)/total,100*len(gau)/total],
                                   [s['perc_sigmoid_per_mouse'][i],s['perc_gaussian_per_mouse'][i]],rtol=1e-12)
        mouse_rows.append([mouse,total,len(sig),len(gau),s['perc_sigmoid_per_mouse'][i],
                           s['perc_gaussian_per_mouse'][i],s['behavior_thresholds'][i]]+[s[k][i] for k in extra])
        for fit,values in [('Sigmoid',sig),('Gaussian',gau)]:
            bins=np.searchsorted(edges,values,side='right')-1
            bins[values==edges[-1]]=len(centers)-1
            for j,(value,b) in enumerate(zip(values,bins)):
                in_range=0<=b<len(centers)
                neuron_rows.append([mouse,fit,j,value,int(b)+1 if in_range else None,
                                    centers[b] if in_range else None,bool(in_range),value-s['behavior_thresholds'][i]])
        c=behavior_arrays[mouse]['conditions_played.npy'];r=behavior_arrays[mouse]['responses.npy']
        for j,(cond,response) in enumerate(zip(c,r)):
            behavior_rows.append([mouse,j,cond,response,j>=23])
    stat_rows=[]
    rep=s['rep']
    stat_rows.append(['Total Sigmoid vs Gaussian (%)',rep['test'],rep['statistic_name'],rep['statistic_value'],
                      rep['df'],s['p'],rep['effect_size']['name'],rep['effect_size']['value'],len(ids)])
    for entry in s['rows']:
        stat_rows.append([entry['Analysis'],entry['Test'],entry['Stat name'],entry['Stat'],entry['df'],entry['p'],None,None,entry['n']])
    notes='Original full-session behavioral reference, after removing 23 priming trials. Mouse IDs align all paired columns.'
    tables={
      'Percent_all_neurons':('Mean and sample SEM of per-mouse category percentages. Denominator: all analyzed neurons. Individual values and counts are in Per_mouse_metrics.',
          ['Fit','Mean (%)','SEM (%)','N mice'],summary),
      'Frequency_pooled':('Pooled-neuron histogram (no SEM). Each fit is normalized by the sum of its IN-RANGE bin counts. Bins left-inclusive/right-exclusive; final bin includes right edge.',
          ['Frequency (Hz)','Left edge (Hz)','Right edge (Hz)','Sigmoid count','Gaussian count','Sigmoid (%)','Gaussian (%)'],pooled),
      'Frequency_mean_SEM':('Each mouse is first normalized by its fit-specific in-range bin counts; then mean and sample SEM across mice. This differs from pooled-neuron weighting.',
          ['Fit','Frequency (Hz)','Left edge (Hz)','Right edge (Hz)','Mean (%)','SEM (%)','N mice']+ids,frequency),
      'Per_mouse_metrics':(notes,
          ['Mouse ID','Total neurons','Sigmoid neurons','Gaussian neurons','Sigmoid (%)','Gaussian (%)','Behavior threshold (Hz)',
           'Sigmoid histogram peak (Hz)','Gaussian histogram peak (Hz)','Sigmoid >=5Hz (%)','Gaussian >=5Hz (%)',
           'Sigmoid <behavior (%)','Gaussian <behavior (%)','Sigmoid >=behavior (%)','Gaussian >=behavior (%)',
           'Sigmoid excitatory fraction','Sigmoid inhibitory fraction','Gaussian excitatory fraction','Gaussian inhibitory fraction'],mouse_rows),
      'Neuron_frequencies':('Values copied from the original saved frequency arrays. Array index is 0-based and is NOT a biological neuron ID. Out-of-bin values remain here; histogram denominators exclude them.',
          ['Mouse ID','Fit','Saved array index','Frequency (Hz)','Bin (1-based)','Bin center (Hz)','In histogram range','Frequency minus behavior (Hz)'],neuron_rows),
      'Behavior_trials':(notes+' Response codes and condition codes are preserved as stored.',
          ['Mouse ID','Original trial index','Condition code','Response code','Used in behavior fit'],behavior_rows),
      'Plot_values':('Actual artists from every saved figure. Bars: (X,Y) lower-left and (X end,Y end) upper-right. Lines/dots/fills: ordered points. Blended axes/data coordinates are marked explicitly. Axis/series indices are 0-based.',
          ['Plot filename','Axis','Artist kind','Artist index','Label','Point/segment index','X','Y','X end','Y end','Coordinates'],capture),
      'Statistics':('Original tests and settings preserved. Total-category comparison uses paired Wilcoxon. Excitatory/inhibitory comparisons use original auto paired tests and repeated-measures ANOVA. Full details are also in verification.json.',
          ['Comparison','Test','Statistic name','Statistic','df','p value','Effect name','Effect value','N mice'],stat_rows),
      'Sources':('Exact input bytes are retained in numbered snapshots. Original file paths and SHA256 establish which saved source version was used.',
          ['Mouse ID','Session YYMMDD','Input kind','Filename','Original path','Snapshot path','SHA256'],manifest)
    }
    # Check captured bars of both candidate frequency figures against named exports.
    for filename,expected in [
      ('percent_neurons_per_frequency_sigmoid_vs_gaussian_pooled.pdf',np.r_[s['percent_sig_all'],s['percent_gauss_all']]),
      ('percent_neurons_per_frequency_sigmoid_vs_gaussian.pdf',np.r_[s['mean_sig'],s['mean_gauss']]),
      ('total_percent_sigmoid_vs_gaussian_neurons.pdf',[s['mean_sigmoid_total'],s['mean_gaussian_total']])]:
        actual=[r[9]-r[7] for r in capture if r[0]==filename and r[2]=='bar']
        np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-12)
    return tables


def _export_and_verify(out,tables):
    wb=Workbook();wb.remove(wb.active)
    for name,(note,headers,rows) in tables.items():
        _table(wb,name,note,headers,rows)
    path=out/'source_data.xlsx';wb.save(path);wb.close()
    saved=load_workbook(path,read_only=True,data_only=True)
    try:
        _require(saved.sheetnames==list(tables),'Workbook sheet mismatch.')
        for name,(_,headers,expected) in tables.items():
            actual=list(saved[name].iter_rows(min_row=5,max_col=len(headers),values_only=True))
            _require(len(actual)==len(expected),'Row count mismatch: '+name)
            for row,original in zip(actual,expected):
                for a,b in zip(row,original):
                    b=_value(b)
                    if isinstance(b,(float,int)) and not isinstance(b,bool):
                        _require(a is not None and np.isclose(a,b,rtol=1e-12,atol=1e-12),name+': numeric mismatch.')
                    else:_require(a==b,name+': identity/text mismatch '+str((a,b)))
    finally:saved.close()
    return path


def _clean(value):
    if isinstance(value,dict):return {str(k):_clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [_clean(v) for v in value]
    if isinstance(value,np.ndarray):return _clean(value.tolist())
    return _value(value)


def run(data_root=None,output_parent=None,subsets=None):
    global _ACTIVE_CAPTURE,_ACTIVE_OUTPUT
    # Preserve the current console encoding; avoid Windows print failures for
    # the original statistical report's Unicode sqrt / plus-minus characters.
    if hasattr(sys.stdout,'reconfigure'):
        sys.stdout.reconfigure(errors='backslashreplace')
    root=DATA_ROOT if data_root is None else Path(data_root)
    parent=OUTPUT_PARENT if output_parent is None else Path(output_parent)
    subsets=tuple(SUBSETS if subsets is None else subsets)
    _preflight(root,subsets)
    run_dir=parent/('run_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    _require(os.name!='nt' or len(str((run_dir/'not_after_hit/percent_neurons_per_frequency_sigmoid_vs_gaussian_pooled.png').resolve()))<240,
             'Output path is too long; use a shorter OUTPUT_PARENT.')
    for subset in subsets:
        out=run_dir/subset;out.mkdir(parents=True,exist_ok=False)
        neural,behavior=_paths(root,subset)
        try:
            manifest,arrays,frames,behavior_arrays=_snapshot_and_validate(neural,behavior,out)
            _ACTIVE_OUTPUT=out;_ACTIVE_CAPTURE=[]
            print('\nRUNNING FULL ORIGINAL ANALYSIS: '+subset)
            state=analyze_subset(neural,behavior,str(out))
            tables=_build_tables(state,arrays,frames,behavior_arrays,_ACTIVE_CAPTURE,manifest)
            book=_export_and_verify(out,tables)
            for row in manifest:
                _require(hashlib.sha256(Path(row[4]).read_bytes()).hexdigest()==row[6],'Source changed during Analysis: '+row[4])
                _require(hashlib.sha256((out/row[5]).read_bytes()).hexdigest()==row[6],'Snapshot changed: '+row[5])
            report={'passed':True,'subset':subset,'mice':MOUSE_IDS,'original_script':ORIGINAL_SCRIPT,
                    'original_sha256':ORIGINAL_SHA256,'exporter_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    'checks':['All requested mice present and keyed by ID','All source files snapshotted and hashed',
                              'Pooled and equal-mouse frequency bars match original plotted heights',
                              'Total-category percentages match original source counts and plot',
                              'All exported Excel cells match captured/calculated values','Original inputs unchanged'],
                    'behavior_reference':'Original full session, after first 23 trials; not subset-conditioned',
                    'plot_files':sorted(set(r[0] for r in _ACTIVE_CAPTURE)),
                    'total_percentage_test':state['info'],'exc_inh_sigmoid_test':state['res_sig'],
                    'exc_inh_gaussian_test':state['res_gau'],'anova':state['anova_tbl'].to_dict(orient='records'),
                    'inputs':manifest}
            (out/'verification.json').write_text(json.dumps(_clean(report),indent=2,allow_nan=False),encoding='utf-8')
            print('VERIFIED '+subset+': '+str(book.resolve()))
        except Exception:
            (out/'FAILED.txt').write_text(traceback.format_exc(),encoding='utf-8')
            print('Run did not complete. Diagnostics: '+str(out/'FAILED.txt'))
            raise
        finally:
            plt.close('all');_ACTIVE_CAPTURE=[];_ACTIVE_OUTPUT=None
    print('\nALL REQUESTED SUBSETS COMPLETE: '+str(run_dir.resolve()))
    return run_dir


if __name__=='__main__':
    run()
