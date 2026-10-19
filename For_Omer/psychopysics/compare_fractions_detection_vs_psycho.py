#!/usr/bin/env python3
"""
Comparison script for Detection vs Psychophysics analyses
Compares the results of plot_reward_vs_light_simple.py (detection) with plot_reward_vs_light_psycho.py (psychophysics)
Creates side-by-side plots and statistical comparisons for the same mice across both datasets.
"""

import os
import pickle
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from matplotlib_venn import venn2
import matplotlib.patches as patches
from scipy import stats
from scipy.stats import shapiro, levene, ttest_rel, ttest_ind, mannwhitneyu, wilcoxon, t as t_dist, iqr, rankdata, norm
from scipy.stats import t as student_t
import pandas as pd
from scipy import stats
plt.rcParams["font.family"] = "Arial"
plt.rcParams["pdf.fonttype"] = 42
# ---------- helpers for stats and annotation ----------

def _shapiro_safe(x):
    """Return (W, p) if n>=3 and not constant, else (None, None)."""
    x = np.asarray(x, float)
    if len(x) < 3 or np.allclose(x, x[0]):
        return None, None
    try:
        return stats.shapiro(x)
    except Exception:
        return None, None
def _mad_outliers(x, thresh=20):
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
def choose_paired_test(x, y):
    """
    Decide paired test based on Shapiro on paired differences.
    Returns dict with keys:
      method: 'ttest_rel' or 'wilcoxon'
      stat, p, n, normal
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    diff = x - y
    n = len(diff)
    W, p_norm = _shapiro_safe(diff)
    normal = (p_norm is not None) and (p_norm > 0.05)

    if normal and n >= 2 and np.std(diff, ddof=1) > 0:
        t, p = stats.ttest_rel(x, y)
        return dict(method='ttest_rel', stat=t, p=p, n=n, normal=True, normal_p=p_norm)
    else:
        # wilcoxon will drop zeros by default if zero_method='wilcox'
        try:
            w, p = stats.wilcoxon(x, y, zero_method='wilcox', alternative='two-sided')
        except ValueError:
            # Fallback when all diffs are zero
            w, p = 0.0, 1.0
        return dict(method='wilcoxon', stat=w, p=p, n=n, normal=False, normal_p=p_norm)

def p_to_stars(p):
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return 'n.s.'

def add_sig_bracket(ax, x1, x2, y, p, text_y_pad=0.02):
    """
    Draw a significance bracket between bars centered at x1 and x2, baseline at y.
    y should be above both bar tops. Uses axes y-range to compute padding.
    """
    ymax = ax.get_ylim()[1]
    ypad = (ax.get_ylim()[1] - ax.get_ylim()[0]) * text_y_pad
    ax.plot([x1, x1, x2, x2], [y, y+ypad, y+ypad, y], color='black', linewidth=1)
    ax.text((x1+x2)/2, y+ypad*1.1, f"{p_to_stars(p)}  p={p:.3g}", ha='center', va='bottom', fontsize=11)

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


def load_comparison_data(detection_pkl_files, psycho_pkl_files):
    """
    Loads 2-set fraction data from both detection and psychophysics pickle files.
    Returns both datasets with matched mice.
    """
    detection_data = {}
    psycho_data = {}
    
    print("Loading Detection data...")
    for pkl_file in detection_pkl_files:
        print(f"  Loading: {pkl_file}")
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
            if "2set" in data:
                for comp_name, fraction_dict in data["2set"].items():
                    if comp_name not in detection_data:
                        detection_data[comp_name] = []
                    detection_data[comp_name].append(fraction_dict)
        except FileNotFoundError:
            print(f"Warning: Detection file not found: {pkl_file}")
        except Exception as e:
            print(f"Error loading detection file {pkl_file}: {e}")
    
    print("\nLoading Psychophysics data...")
    for pkl_file in psycho_pkl_files:
        print(f"  Loading: {pkl_file}")
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
            if "2set" in data:
                for comp_name, fraction_dict in data["2set"].items():
                    if comp_name not in psycho_data:
                        psycho_data[comp_name] = []
                    psycho_data[comp_name].append(fraction_dict)
        except FileNotFoundError:
            print(f"Warning: Psychophysics file not found: {pkl_file}")
        except Exception as e:
            print(f"Error loading psychophysics file {pkl_file}: {e}")
    
    return detection_data, psycho_data


def extract_data_arrays(data_dict, inc_key="Hit(reward)_LightOn_Increase", dec_key="Hit(reward)_LightOn_Decrease"):
    """
    Extract data arrays from the loaded data dictionary.
    Returns arrays for reward_only, light_only, both for both increase and decrease.
    """
    if inc_key not in data_dict or dec_key not in data_dict:
        print(f"Keys {inc_key} or {dec_key} not found in data.")
        print(f"Available keys: {list(data_dict.keys())}")
        return None, None, None, None, None, None
    
    inc_data = data_dict[inc_key]
    dec_data = data_dict[dec_key]
    
    # Extract increase data
    reward_only_inc = [d.get("set1_only", 0) for d in inc_data]
    light_only_inc = [d.get("set2_only", 0) for d in inc_data]
    both_inc = [d.get("both", 0) for d in inc_data]
    
    # Extract decrease data
    reward_only_dec = [d.get("set1_only", 0) for d in dec_data]
    light_only_dec = [d.get("set2_only", 0) for d in dec_data]
    both_dec = [d.get("both", 0) for d in dec_data]
    
    return reward_only_inc, light_only_inc, both_inc, reward_only_dec, light_only_dec, both_dec


def plot_side_by_side_comparison(detection_data, psycho_data, save_dir="."):
    """
    Creates side-by-side comparison plots showing Detection vs Psychophysics results.
    """
    # Extract data arrays for both datasets
    det_r_inc, det_l_inc, det_b_inc, det_r_dec, det_l_dec, det_b_dec = extract_data_arrays(detection_data)
    psy_r_inc, psy_l_inc, psy_b_inc, psy_r_dec, psy_l_dec, psy_b_dec = extract_data_arrays(psycho_data)
    
    if det_r_inc is None or psy_r_inc is None:
        print("Failed to extract data arrays. Check your data.")
        return
    
    # Calculate total responsive cells
    det_reward_total_inc = [r + b for r, b in zip(det_r_inc, det_b_inc)]
    det_light_total_inc = [l + b for l, b in zip(det_l_inc, det_b_inc)]
    det_reward_total_dec = [r + b for r, b in zip(det_r_dec, det_b_dec)]
    det_light_total_dec = [l + b for l, b in zip(det_l_dec, det_b_dec)]
    
    psy_reward_total_inc = [r + b for r, b in zip(psy_r_inc, psy_b_inc)]
    psy_light_total_inc = [l + b for l, b in zip(psy_l_inc, psy_b_inc)]
    psy_reward_total_dec = [r + b for r, b in zip(psy_r_dec, psy_b_dec)]
    psy_light_total_dec = [l + b for l, b in zip(psy_l_dec, psy_b_dec)]
    
    # Calculate means and SEMs
    det_reward_inc_mean, det_reward_inc_sem = mean_sem(det_reward_total_inc)
    det_light_inc_mean, det_light_inc_sem = mean_sem(det_light_total_inc)
    det_reward_dec_mean, det_reward_dec_sem = mean_sem(det_reward_total_dec)
    det_light_dec_mean, det_light_dec_sem = mean_sem(det_light_total_dec)
    
    psy_reward_inc_mean, psy_reward_inc_sem = mean_sem(psy_reward_total_inc)
    psy_light_inc_mean, psy_light_inc_sem = mean_sem(psy_light_total_inc)
    psy_reward_dec_mean, psy_reward_dec_sem = mean_sem(psy_reward_total_dec)
    psy_light_dec_mean, psy_light_dec_sem = mean_sem(psy_light_total_dec)
    
    # Create side-by-side plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Detection plot (left)
    x = np.arange(2)
    group_labels = ["Reward Responsive", "Light Responsive"]
    bar_width = 0.6
    
    # Detection - Increase bars
    ax1.bar(x, [det_reward_inc_mean, det_light_inc_mean],
            width=bar_width, color='salmon', edgecolor='black',
            label='Increase', alpha=0.8)
    ax1.errorbar(x, [det_reward_inc_mean, det_light_inc_mean],
                 yerr=[det_reward_inc_sem, det_light_inc_sem],
                 fmt='none', ecolor='black', capsize=3)
    
    # Detection - Decrease bars
    ax1.bar(x, [-det_reward_dec_mean, -det_light_dec_mean],
            width=bar_width, color='skyblue', edgecolor='black',
            label='Decrease', alpha=0.8)
    ax1.errorbar(x, [-det_reward_dec_mean, -det_light_dec_mean],
                 yerr=[det_reward_dec_sem, det_light_dec_sem],
                 fmt='none', ecolor='black', capsize=3)
    
    # Detection - Individual data lines
    for i in range(len(det_reward_total_inc)):
        ax1.plot(x, [det_reward_total_inc[i], det_light_total_inc[i]],
                 color='salmon', marker='o', linestyle='--', linewidth=1,
                 alpha=0.6, zorder=3)
    for i in range(len(det_reward_total_dec)):
        ax1.plot(x, [-det_reward_total_dec[i], -det_light_total_dec[i]],
                 color='skyblue', marker='o', linestyle='--', linewidth=1,
                 alpha=0.6, zorder=3)
    
    ax1.axhline(0, color='black', linewidth=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(group_labels, rotation=45, ha='right')
    yticks = np.arange(-50, 51, 10)
    ax1.set_yticks(yticks)
    ax1.set_yticklabels([f"{abs(y)}%" for y in yticks])
    ax1.set_ylabel("Percentage of total neurons")
    ax1.set_title("Detection Analysis", fontsize=14, fontweight='bold')
    ax1.legend()
    
    # Psychophysics plot (right)
    # Psychophysics - Increase bars
    ax2.bar(x, [psy_reward_inc_mean, psy_light_inc_mean],
            width=bar_width, color='salmon', edgecolor='black',
            label='Increase', alpha=0.8)
    ax2.errorbar(x, [psy_reward_inc_mean, psy_light_inc_mean],
                 yerr=[psy_reward_inc_sem, psy_light_inc_sem],
                 fmt='none', ecolor='black', capsize=3)
    
    # Psychophysics - Decrease bars
    ax2.bar(x, [-psy_reward_dec_mean, -psy_light_dec_mean],
            width=bar_width, color='skyblue', edgecolor='black',
            label='Decrease', alpha=0.8)
    ax2.errorbar(x, [-psy_reward_dec_mean, -psy_light_dec_mean],
                 yerr=[psy_reward_dec_sem, psy_light_dec_sem],
                 fmt='none', ecolor='black', capsize=3)
    
    # Psychophysics - Individual data lines
    for i in range(len(psy_reward_total_inc)):
        ax2.plot(x, [psy_reward_total_inc[i], psy_light_total_inc[i]],
                 color='salmon', marker='o', linestyle='--', linewidth=1,
                 alpha=0.6, zorder=3)
    for i in range(len(psy_reward_total_dec)):
        ax2.plot(x, [-psy_reward_total_dec[i], -psy_light_total_dec[i]],
                 color='skyblue', marker='o', linestyle='--', linewidth=1,
                 alpha=0.6, zorder=3)
    
    ax2.axhline(0, color='black', linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(group_labels, rotation=45, ha='right')
    ax2.set_yticks(yticks)
    ax2.set_yticklabels([f"{abs(y)}%" for y in yticks])
    ax2.set_ylabel("Percentage of total neurons")
    ax2.set_title("Psychophysics Analysis", fontsize=14, fontweight='bold')
    ax2.legend()
    
    # Overall title
    fig.suptitle('Detection vs Psychophysics: Hit(Reward) vs Light Comparison', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(join(save_dir, "Detection_vs_Psycho_Side_by_Side_Comparison.pdf"))
    plt.show()
    
    # Print comparison statistics
    print("\n=== Side-by-Side Comparison Statistics ===")
    print("DETECTION ANALYSIS:")
    print(f"  Reward Responsive Increase: {det_reward_inc_mean:.2f} ± {det_reward_inc_sem:.2f}")
    print(f"  Light Responsive Increase: {det_light_inc_mean:.2f} ± {det_light_inc_sem:.2f}")
    print(f"  Reward Responsive Decrease: {det_reward_dec_mean:.2f} ± {det_reward_dec_sem:.2f}")
    print(f"  Light Responsive Decrease: {det_light_dec_mean:.2f} ± {det_light_dec_sem:.2f}")
    
    print("\nPSYCHOPHYSICS ANALYSIS:")
    print(f"  Reward Responsive Increase: {psy_reward_inc_mean:.2f} ± {psy_reward_inc_sem:.2f}")
    print(f"  Light Responsive Increase: {psy_light_inc_mean:.2f} ± {psy_light_inc_sem:.2f}")
    print(f"  Reward Responsive Decrease: {psy_reward_dec_mean:.2f} ± {psy_reward_dec_sem:.2f}")
    print(f"  Light Responsive Decrease: {psy_light_dec_mean:.2f} ± {psy_light_dec_sem:.2f}")


def plot_side_by_side_venn_comparison(detection_data, psycho_data, save_dir="."):
    """
    Creates side-by-side Venn diagram comparison showing Detection vs Psychophysics results.
    """
    # Extract data arrays for both datasets
    det_r_inc, det_l_inc, det_b_inc, det_r_dec, det_l_dec, det_b_dec = extract_data_arrays(detection_data)
    psy_r_inc, psy_l_inc, psy_b_inc, psy_r_dec, psy_l_dec, psy_b_dec = extract_data_arrays(psycho_data)
    
    if det_r_inc is None or psy_r_inc is None:
        print("Failed to extract data arrays. Check your data.")
        return
    
    # Combine increase and decrease data for overall comparison
    det_reward_only_combined = [r_inc + r_dec for r_inc, r_dec in zip(det_r_inc, det_r_dec)]
    det_light_only_combined = [l_inc + l_dec for l_inc, l_dec in zip(det_l_inc, det_l_dec)]
    det_both_combined = [b_inc + b_dec for b_inc, b_dec in zip(det_b_inc, det_b_dec)]
    
    psy_reward_only_combined = [r_inc + r_dec for r_inc, r_dec in zip(psy_r_inc, psy_r_dec)]
    psy_light_only_combined = [l_inc + l_dec for l_inc, l_dec in zip(psy_l_inc, psy_l_dec)]
    psy_both_combined = [b_inc + b_dec for b_inc, b_dec in zip(psy_b_inc, psy_b_dec)]
    
    # Calculate means
    det_reward_only_mean = np.mean(det_reward_only_combined)
    det_light_only_mean = np.mean(det_light_only_combined)
    det_both_mean = np.mean(det_both_combined)
    # calculate SEMs
    det_reward_only_sem = np.std(det_reward_only_combined, ddof=1) / sqrt(len(det_reward_only_combined))
    det_light_only_sem = np.std(det_light_only_combined, ddof=1) / sqrt(len(det_light_only_combined))
    det_both_sem = np.std(det_both_combined, ddof=1) / sqrt(len(det_both_combined))
    
    psy_reward_only_mean = np.mean(psy_reward_only_combined)
    psy_light_only_mean = np.mean(psy_light_only_combined)
    psy_both_mean = np.mean(psy_both_combined)
    # calculate SEMs
    psy_reward_only_sem = np.std(psy_reward_only_combined, ddof=1) / sqrt(len(psy_reward_only_combined))
    psy_light_only_sem = np.std(psy_light_only_combined, ddof=1) / sqrt(len(psy_light_only_combined))
    psy_both_sem = np.std(psy_both_combined, ddof=1) / sqrt(len(psy_both_combined))
    
    # Create side-by-side Venn diagrams
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Detection Venn diagram + sem (left)
    venn_det = venn2(subsets=(det_reward_only_mean, det_light_only_mean, det_both_mean),
                     set_labels=('Reward\nResponsive', 'Light\nResponsive'),
                     ax=ax1)

    # Customize colors for detection
    if venn_det.get_patch_by_id('10'):  # Reward only
        venn_det.get_patch_by_id('10').set_color('lightblue')
        venn_det.get_patch_by_id('10').set_alpha(0.7)
    if venn_det.get_patch_by_id('01'):  # Light only
        venn_det.get_patch_by_id('01').set_color('lightcoral')
        venn_det.get_patch_by_id('01').set_alpha(0.7)
    if venn_det.get_patch_by_id('11'):  # Both
        venn_det.get_patch_by_id('11').set_color('mediumpurple')
        venn_det.get_patch_by_id('11').set_alpha(0.7)
    
    # Add percentage labels for detection
    if venn_det.get_label_by_id('10'):
        venn_det.get_label_by_id('10').set_text(f'{det_reward_only_mean:.1f}% ± {det_reward_only_sem:.1f}%')
        venn_det.get_label_by_id('10').set_fontsize(12)
        venn_det.get_label_by_id('10').set_fontweight('bold')
    if venn_det.get_label_by_id('01'):
        venn_det.get_label_by_id('01').set_text(f'{det_light_only_mean:.1f}% ± {det_light_only_sem:.1f}%')
        venn_det.get_label_by_id('01').set_fontsize(12)
        venn_det.get_label_by_id('01').set_fontweight('bold')
    if venn_det.get_label_by_id('11'):
        venn_det.get_label_by_id('11').set_text(f'{det_both_mean:.1f}% ± {det_both_sem:.1f}%')
        venn_det.get_label_by_id('11').set_fontsize(12)
        venn_det.get_label_by_id('11').set_fontweight('bold')
    
    ax1.set_title('Detection Analysis\n(Combined Increase + Decrease)', 
                  fontsize=14, fontweight='bold')
    
    # Psychophysics Venn diagram (right)
    venn_psy = venn2(subsets=(psy_reward_only_mean, psy_light_only_mean, psy_both_mean),
                     set_labels=('Reward\nResponsive', 'Light\nResponsive'),
                     ax=ax2)
    
    # Customize colors for psychophysics
    if venn_psy.get_patch_by_id('10'):  # Reward only
        venn_psy.get_patch_by_id('10').set_color('lightblue')
        venn_psy.get_patch_by_id('10').set_alpha(0.7)
    if venn_psy.get_patch_by_id('01'):  # Light only
        venn_psy.get_patch_by_id('01').set_color('lightcoral')
        venn_psy.get_patch_by_id('01').set_alpha(0.7)
    if venn_psy.get_patch_by_id('11'):  # Both
        venn_psy.get_patch_by_id('11').set_color('mediumpurple')
        venn_psy.get_patch_by_id('11').set_alpha(0.7)
    
    # Add percentage labels for psychophysics
    if venn_psy.get_label_by_id('10'):
        venn_psy.get_label_by_id('10').set_text(f'{psy_reward_only_mean:.1f}% ± {psy_reward_only_sem:.1f}%')
        venn_psy.get_label_by_id('10').set_fontsize(12)
        venn_psy.get_label_by_id('10').set_fontweight('bold')
    if venn_psy.get_label_by_id('01'):
        venn_psy.get_label_by_id('01').set_text(f'{psy_light_only_mean:.1f}% ± {psy_light_only_sem:.1f}%')
        venn_psy.get_label_by_id('01').set_fontsize(12)
        venn_psy.get_label_by_id('01').set_fontweight('bold')
    if venn_psy.get_label_by_id('11'):
        venn_psy.get_label_by_id('11').set_text(f'{psy_both_mean:.1f}% ± {psy_both_sem:.1f}%')
        venn_psy.get_label_by_id('11').set_fontsize(12)
        venn_psy.get_label_by_id('11').set_fontweight('bold')
    
    ax2.set_title('Psychophysics Analysis\n(Combined Increase + Decrease)', 
                  fontsize=14, fontweight='bold')
    
    # Overall title
    fig.suptitle('Detection vs Psychophysics: Venn Diagram Comparison', fontsize=16, fontweight='bold')
    
    # Add shared legend
    legend_elements = [
        patches.Patch(color='lightblue', alpha=0.7, label='Reward Only'),
        patches.Patch(color='lightcoral', alpha=0.7, label='Light Only'),
        patches.Patch(color='mediumpurple', alpha=0.7, label='Both')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.05))
    
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(join(save_dir, "Detection_vs_Psycho_Venn_Comparison.pdf"), bbox_inches='tight')
    plt.show()
    
    # Print Venn comparison statistics
    print("\n=== Venn Diagram Comparison Statistics ===")
    print("DETECTION ANALYSIS:")
    print(f"  Reward Only: {det_reward_only_mean:.1f}%")
    print(f"  Light Only: {det_light_only_mean:.1f}%")
    print(f"  Both: {det_both_mean:.1f}%")
    print(f"  Total Reward Responsive: {det_reward_only_mean + det_both_mean:.1f}%")
    print(f"  Total Light Responsive: {det_light_only_mean + det_both_mean:.1f}%")
    
    print("\nPSYCHOPHYSICS ANALYSIS:")
    print(f"  Reward Only: {psy_reward_only_mean:.1f}%")
    print(f"  Light Only: {psy_light_only_mean:.1f}%")
    print(f"  Both: {psy_both_mean:.1f}%")
    print(f"  Total Reward Responsive: {psy_reward_only_mean + psy_both_mean:.1f}%")
    print(f"  Total Light Responsive: {psy_light_only_mean + psy_both_mean:.1f}%")


import os
from os.path import join
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

ALPHA_NORMALITY = 0.05  # threshold for Shapiro-Wilk on paired differences

def _choose_paired_test(x, y, alpha=ALPHA_NORMALITY):
    """
    Decide between paired t-test and Wilcoxon based on Shapiro-Wilk on paired differences.
    Returns a dict with results and metadata.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    assert x.shape == y.shape, "Paired samples must have identical length"
    n = x.size

    # paired differences
    d = x - y
    n_eff = np.count_nonzero(d)  # Wilcoxon ignores exact zeros

    result = {
        "n": n,
        "n_effective": int(n_eff),
        "normality_test": "Shapiro-Wilk on paired differences",
        "shapiro_W": np.nan,
        "shapiro_p": np.nan,
        "test": None,
        "stat_name": None,
        "stat_value": np.nan,
        "df": np.nan,
        "p_value": np.nan,
        "mean_diff": float(np.nanmean(d)) if n > 0 else np.nan,
        "std_diff": float(np.nanstd(d, ddof=1)) if n > 1 else np.nan,
        "ci_low": np.nan,
        "ci_high": np.nan,
        "effect_size": np.nan,  # dz for t-test, leave blank for Wilcoxon
    }

    # Normality only valid for n >= 3
    use_normal = False
    if n >= 3:
        try:
            W, p_sh = stats.shapiro(d)
            result["shapiro_W"] = float(W)
            result["shapiro_p"] = float(p_sh)
            use_normal = (p_sh >= alpha)
        except Exception:
            # If Shapiro fails, fall back to nonparametric
            use_normal = False

    if use_normal:
        # Paired t-test
        t_stat, p_val = stats.ttest_rel(x, y, nan_policy="omit")
        result.update({
            "test": "paired t-test",
            "stat_name": "t",
            "stat_value": float(t_stat),
            "df": int(np.count_nonzero(~np.isnan(d)) - 1),
            "p_value": float(p_val),
        })
        # 95% CI for mean difference
        df = result["df"]
        if df >= 1 and not np.isnan(result["std_diff"]):
            se = result["std_diff"] / np.sqrt(df + 1)
            t_crit = stats.t.ppf(0.975, df)
            md = result["mean_diff"]
            result["ci_low"] = float(md - t_crit * se)
            result["ci_high"] = float(md + t_crit * se)
            # Cohen's dz for paired design
            result["effect_size"] = float(md / result["std_diff"]) if result["std_diff"] > 0 else np.nan
    else:
        # Wilcoxon signed-rank
        try:
            # zero_method "wilcox" excludes zero-diffs; correction uses continuity correction
            W, p_val = stats.wilcoxon(x, y, zero_method="wilcox", alternative="two-sided", correction=True)
            stat_value = float(W)
            p_value = float(p_val)
        except ValueError:
            # Happens if all differences are zero
            stat_value = 0.0
            p_value = 1.0
        result.update({
            "test": "Wilcoxon signed-rank",
            "stat_name": "W",
            "stat_value": stat_value,
            "df": np.nan,  # not defined for Wilcoxon
            "p_value": p_value,
        })

    return result

def perform_statistical_comparison(detection_data, psycho_data, save_dir="."):
    """
    Performs statistical comparison between detection and psychophysics data.
    For each paired comparison, pick the appropriate test automatically:
      - If paired differences are normal by Shapiro-Wilk -> paired t-test
      - Otherwise -> Wilcoxon signed-rank

    Also creates correlation plots and saves a stats Excel file with df and t or W.
    """
    # Extract data arrays
    det_r_inc, det_l_inc, det_b_inc, det_r_dec, det_l_dec, det_b_dec = extract_data_arrays(detection_data)
    psy_r_inc, psy_l_inc, psy_b_inc, psy_r_dec, psy_l_dec, psy_b_dec = extract_data_arrays(psycho_data)

    if det_r_inc is None or psy_r_inc is None:
        print("Failed to extract data arrays. Check your data.")
        return

    # Totals
    det_reward_total_inc = [r + b for r, b in zip(det_r_inc, det_b_inc)]
    det_light_total_inc  = [l + b for l, b in zip(det_l_inc, det_b_inc)]
    det_reward_total_dec = [r + b for r, b in zip(det_r_dec, det_b_dec)]
    det_light_total_dec  = [l + b for l, b in zip(det_l_dec, det_b_dec)]

    psy_reward_total_inc = [r + b for r, b in zip(psy_r_inc, psy_b_inc)]
    psy_light_total_inc  = [l + b for l, b in zip(psy_l_inc, psy_b_inc)]
    psy_reward_total_dec = [r + b for r, b in zip(psy_r_dec, psy_b_dec)]
    psy_light_total_dec  = [l + b for l, b in zip(psy_l_dec, psy_b_dec)]

    comparisons = [
        ("Reward Responsive Increase", det_reward_total_inc, psy_reward_total_inc),
        ("Light Responsive Increase",  det_light_total_inc,  psy_light_total_inc),
        ("Reward Responsive Decrease", det_reward_total_dec, psy_reward_total_dec),
        ("Light Responsive Decrease",  det_light_total_dec,  psy_light_total_dec),
    ]

    # Run tests and collect stats
    rows = []
    print("\n=== Statistical Comparison (auto test selection) ===")
    for label, x, y in comparisons:
        res = _choose_paired_test(x, y)
        res["comparison"] = label

        # Pearson correlation as in your original function
        try:
            r_corr, r_p = stats.pearsonr(x, y)
            res["correlation"] = "Pearson"
            res["r"] = float(r_corr)
            res["r_p"] = float(r_p)
        except Exception:
            res["correlation"] = "Pearson"
            res["r"] = np.nan
            res["r_p"] = np.nan

        print(f"{label}: {res['test']} -> {res['stat_name']}={res['stat_value']:.3f}, "
              f"df={res['df'] if pd.notna(res['df']) else 'NA'}, p={res['p_value']:.4f}")
        rows.append(res)

    # Save stats to Excel
    os.makedirs(save_dir, exist_ok=True)
    df_stats = pd.DataFrame(rows, columns=[
        "comparison", "n", "n_effective",
        "normality_test", "shapiro_W", "shapiro_p",
        "test", "stat_name", "stat_value", "df", "p_value",
        "mean_diff", "std_diff", "ci_low", "ci_high",
        "correlation", "r", "r_p", "effect_size"
    ])
    excel_path = join(save_dir, "stats_summary.xlsx")
    df_stats.to_excel(excel_path, index=False)
    print(f"\nSaved stats to {excel_path}")

    # Correlation plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # Reward increase
    ax1.scatter(det_reward_total_inc, psy_reward_total_inc, alpha=0.7)
    lim1 = max(max(det_reward_total_inc), max(psy_reward_total_inc))
    ax1.plot([0, lim1], [0, lim1], 'k--', alpha=0.5)
    ax1.set_xlabel('Detection (%)')
    ax1.set_ylabel('Psychophysics (%)')
    ax1.set_title(f"Reward Responsive Increase\nr={rows[0]['r']:.3f}, p={rows[0]['r_p']:.3f}")

    # Light increase
    ax2.scatter(det_light_total_inc, psy_light_total_inc, alpha=0.7)
    lim2 = max(max(det_light_total_inc), max(psy_light_total_inc))
    ax2.plot([0, lim2], [0, lim2], 'k--', alpha=0.5)
    ax2.set_xlabel('Detection (%)')
    ax2.set_ylabel('Psychophysics (%)')
    ax2.set_title(f"Light Responsive Increase\nr={rows[1]['r']:.3f}, p={rows[1]['r_p']:.3f}")

    # Reward decrease
    ax3.scatter(det_reward_total_dec, psy_reward_total_dec, alpha=0.7)
    lim3 = max(max(det_reward_total_dec), max(psy_reward_total_dec))
    ax3.plot([0, lim3], [0, lim3], 'k--', alpha=0.5)
    ax3.set_xlabel('Detection (%)')
    ax3.set_ylabel('Psychophysics (%)')
    ax3.set_title(f"Reward Responsive Decrease\nr={rows[2]['r']:.3f}, p={rows[2]['r_p']:.3f}")

    # Light decrease
    ax4.scatter(det_light_total_dec, psy_light_total_dec, alpha=0.7)
    lim4 = max(max(det_light_total_dec), max(psy_light_total_dec))
    ax4.plot([0, lim4], [0, lim4], 'k--', alpha=0.5)
    ax4.set_xlabel('Detection (%)')
    ax4.set_ylabel('Psychophysics (%)')
    ax4.set_title(f"Light Responsive Decrease\nr={rows[3]['r']:.3f}, p={rows[3]['r_p']:.3f}")

    plt.suptitle('Detection vs Psychophysics: Correlation Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    fig_path = join(save_dir, "Detection_vs_Psycho_Correlation_Analysis.pdf")
    plt.savefig(fig_path)
    plt.show()
    print(f"Saved figure to {fig_path}")


def plot_connected_mice_bars(detection_data, psycho_data, save_dir="."):
    """
    Creates a bar plot showing all categories together with connected mice.
    Shows tight pairs of bars for Reward, Light, and Both categories,
    with dashed gray lines connecting the same mice between Detection (Day 1) and Psychophysics (Day 2).
    Maintains separation between increase and decrease data.
    """
    # Extract data arrays for both datasets
    det_r_inc, det_l_inc, det_b_inc, det_r_dec, det_l_dec, det_b_dec = extract_data_arrays(detection_data)
    psy_r_inc, psy_l_inc, psy_b_inc, psy_r_dec, psy_l_dec, psy_b_dec = extract_data_arrays(psycho_data)
    
    if det_r_inc is None or psy_r_inc is None:
        print("Failed to extract data arrays. Check your data.")
        return
    
    # Calculate total responsive cells (reward + both, light + both) for increase and decrease separately
    det_reward_total_inc = [r + b for r, b in zip(det_r_inc, det_b_inc)]
    det_light_total_inc = [l + b for l, b in zip(det_l_inc, det_b_inc)]
    det_reward_total_dec = [r + b for r, b in zip(det_r_dec, det_b_dec)]
    det_light_total_dec = [l + b for l, b in zip(det_l_dec, det_b_dec)]
    
    psy_reward_total_inc = [r + b for r, b in zip(psy_r_inc, psy_b_inc)]
    psy_light_total_inc = [l + b for l, b in zip(psy_l_inc, psy_b_inc)]
    psy_reward_total_dec = [r + b for r, b in zip(psy_r_dec, psy_b_dec)]
    psy_light_total_dec = [l + b for l, b in zip(psy_l_dec, psy_b_dec)]
    
    # Calculate means and SEMs for increase and decrease separately
    det_reward_inc_mean, det_reward_inc_sem = mean_sem(det_reward_total_inc)
    det_light_inc_mean, det_light_inc_sem = mean_sem(det_light_total_inc)
    det_both_inc_mean, det_both_inc_sem = mean_sem(det_b_inc)
    det_reward_dec_mean, det_reward_dec_sem = mean_sem(det_reward_total_dec)
    det_light_dec_mean, det_light_dec_sem = mean_sem(det_light_total_dec)
    det_both_dec_mean, det_both_dec_sem = mean_sem(det_b_dec)
    
    psy_reward_inc_mean, psy_reward_inc_sem = mean_sem(psy_reward_total_inc)
    psy_light_inc_mean, psy_light_inc_sem = mean_sem(psy_light_total_inc)
    psy_both_inc_mean, psy_both_inc_sem = mean_sem(psy_b_inc)
    psy_reward_dec_mean, psy_reward_dec_sem = mean_sem(psy_reward_total_dec)
    psy_light_dec_mean, psy_light_dec_sem = mean_sem(psy_light_total_dec)
    psy_both_dec_mean, psy_both_dec_sem = mean_sem(psy_b_dec)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Set up bar positions - 6 groups total (3 categories x 2 conditions each)
    bar_width = 0.35
    group_spacing = 1.0
    x_groups = np.array([0, group_spacing, 2*group_spacing])  # Centers of the three main groups
    
    # Positions for Detection and Psychophysics bars within each group
    x_detection = x_groups - bar_width/2
    x_psycho = x_groups + bar_width/2
    
    # Plot Increase bars (positive direction)
    detection_inc_means = [det_reward_inc_mean, det_light_inc_mean, det_both_inc_mean]
    detection_inc_sems = [det_reward_inc_sem, det_light_inc_sem, det_both_inc_sem]
    psycho_inc_means = [psy_reward_inc_mean, psy_light_inc_mean, psy_both_inc_mean]
    psycho_inc_sems = [psy_reward_inc_sem, psy_light_inc_sem, psy_both_inc_sem]
    
    bars_det_inc = ax.bar(x_detection, detection_inc_means, bar_width, 
                          label='Detection Increase', color='salmon', edgecolor='black', alpha=0.8)
    bars_psy_inc = ax.bar(x_psycho, psycho_inc_means, bar_width,
                          label='Psychophysics Increase', color='lightcoral', edgecolor='black', alpha=0.8)
    
    # Plot Decrease bars (negative direction)
    detection_dec_means = [det_reward_dec_mean, det_light_dec_mean, det_both_dec_mean]
    detection_dec_sems = [det_reward_dec_sem, det_light_dec_sem, det_both_dec_sem]
    psycho_dec_means = [psy_reward_dec_mean, psy_light_dec_mean, psy_both_dec_mean]
    psycho_dec_sems = [psy_reward_dec_sem, psy_light_dec_sem, psy_both_dec_sem]
    
    bars_det_dec = ax.bar(x_detection, [-x for x in detection_dec_means], bar_width, 
                          label='Detection Decrease', color='skyblue', edgecolor='black', alpha=0.8)
    bars_psy_dec = ax.bar(x_psycho, [-x for x in psycho_dec_means], bar_width,
                          label='Psychophysics Decrease', color='lightsteelblue', edgecolor='black', alpha=0.8)
    
    # Add error bars
    ax.errorbar(x_detection, detection_inc_means, yerr=detection_inc_sems, 
                fmt='none', ecolor='black', capsize=3, linewidth=1.5)
    ax.errorbar(x_psycho, psycho_inc_means, yerr=psycho_inc_sems,
                fmt='none', ecolor='black', capsize=3, linewidth=1.5)
    ax.errorbar(x_detection, [-x for x in detection_dec_means], yerr=detection_dec_sems,
                fmt='none', ecolor='black', capsize=3, linewidth=1.5)
    ax.errorbar(x_psycho, [-x for x in psycho_dec_means], yerr=psycho_dec_sems,
                fmt='none', ecolor='black', capsize=3, linewidth=1.5)
    
    # Connect individual mice with dashed gray lines
    # For increase data
    individual_inc_data = [
        (det_reward_total_inc, psy_reward_total_inc),
        (det_light_total_inc, psy_light_total_inc), 
        (det_b_inc, psy_b_inc)
    ]
    
    # For decrease data  
    individual_dec_data = [
        (det_reward_total_dec, psy_reward_total_dec),
        (det_light_total_dec, psy_light_total_dec), 
        (det_b_dec, psy_b_dec)
    ]
    
    # Draw connecting lines for increase data
    for group_idx, (det_values, psy_values) in enumerate(individual_inc_data):
        x_det_pos = x_detection[group_idx]
        x_psy_pos = x_psycho[group_idx]
        
        for mouse_idx in range(len(det_values)):
            # Draw dashed gray connecting line for each mouse
            ax.plot([x_det_pos, x_psy_pos], 
                   [det_values[mouse_idx], psy_values[mouse_idx]],
                   color='gray', alpha=0.6, linewidth=1, linestyle='--', zorder=3)
    
    # Draw connecting lines for decrease data (negative values)
    for group_idx, (det_values, psy_values) in enumerate(individual_dec_data):
        x_det_pos = x_detection[group_idx]
        x_psy_pos = x_psycho[group_idx]
        
        for mouse_idx in range(len(det_values)):
            # Draw dashed gray connecting line for each mouse
            ax.plot([x_det_pos, x_psy_pos], 
                   [-det_values[mouse_idx], -psy_values[mouse_idx]],
                   color='gray', alpha=0.6, linewidth=1, linestyle='--', zorder=3)
    
    # Add a horizontal line at zero
    ax.axhline(0, color='black', linewidth=0.8)
    
    # Customize the plot
    ax.set_ylabel('Percentage of total neurons (%)', fontsize=14, fontweight='bold')
    ax.set_title('Detection vs Psychophysics',
                 fontsize=16, fontweight='bold', pad=20)
    
    # Set x-axis labels
    group_labels = ['Reward\nResponsive', 'Light\nResponsive', 'Both\nResponsive']
    ax.set_xticks(x_groups)
    ax.set_xticklabels(group_labels, fontsize=12)
    
    # Set y-axis with percentage labels (absolute values)
    all_values = (detection_inc_means + psycho_inc_means + detection_dec_means + psycho_dec_means)
    y_max = max(all_values) * 2
    y_min = -y_max
    # ax.set_ylim(y_min, y_max)
    
    # Create custom y-tick labels showing absolute percentages
    yticks = np.arange(int(y_min/10)*10, int(y_max/10)*10 + 10, 10)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{abs(y)}%" for y in yticks])
    
    # Add legend
    ax.legend(loc='upper right', fontsize=11)
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(join(save_dir, "Detection_vs_Psycho_Connected_Mice_Bars_IncDec.pdf"), bbox_inches='tight', dpi=300)
    plt.show()
    
    # Print statistics
    print("\n=== Connected Mice Bar Plot Statistics (Increase/Decrease Separated) ===")
    print("DETECTION (Day 1) - INCREASE:")
    print(f"  Reward Responsive: {det_reward_inc_mean:.2f} ± {det_reward_inc_sem:.2f}%")
    print(f"  Light Responsive: {det_light_inc_mean:.2f} ± {det_light_inc_sem:.2f}%")
    print(f"  Both Responsive: {det_both_inc_mean:.2f} ± {det_both_inc_sem:.2f}%")
    
    print("\nDETECTION (Day 1) - DECREASE:")
    print(f"  Reward Responsive: {det_reward_dec_mean:.2f} ± {det_reward_dec_sem:.2f}%")
    print(f"  Light Responsive: {det_light_dec_mean:.2f} ± {det_light_dec_sem:.2f}%")
    print(f"  Both Responsive: {det_both_dec_mean:.2f} ± {det_both_dec_sem:.2f}%")
    
    print("\nPSYCHOPHYSICS (Day 2) - INCREASE:")
    print(f"  Reward Responsive: {psy_reward_inc_mean:.2f} ± {psy_reward_inc_sem:.2f}%")
    print(f"  Light Responsive: {psy_light_inc_mean:.2f} ± {psy_light_inc_sem:.2f}%")
    print(f"  Both Responsive: {psy_both_inc_mean:.2f} ± {psy_both_inc_sem:.2f}%")
    
    print("\nPSYCHOPHYSICS (Day 2) - DECREASE:")
    print(f"  Reward Responsive: {psy_reward_dec_mean:.2f} ± {psy_reward_dec_sem:.2f}%")
    print(f"  Light Responsive: {psy_light_dec_mean:.2f} ± {psy_light_dec_sem:.2f}%")
    print(f"  Both Responsive: {psy_both_dec_mean:.2f} ± {psy_both_dec_sem:.2f}%")


def create_summary_table(detection_data, psycho_data, save_dir="."):
    """
    Creates a comprehensive summary table comparing detection and psychophysics results.
    """
    # Extract data arrays
    det_r_inc, det_l_inc, det_b_inc, det_r_dec, det_l_dec, det_b_dec = extract_data_arrays(detection_data)
    psy_r_inc, psy_l_inc, psy_b_inc, psy_r_dec, psy_l_dec, psy_b_dec = extract_data_arrays(psycho_data)
    
    if det_r_inc is None or psy_r_inc is None:
        print("Failed to extract data arrays. Check your data.")
        return
    
    # Calculate total responsive cells
    det_reward_total_inc = [r + b for r, b in zip(det_r_inc, det_b_inc)]
    det_light_total_inc = [l + b for l, b in zip(det_l_inc, det_b_inc)]
    det_reward_total_dec = [r + b for r, b in zip(det_r_dec, det_b_dec)]
    det_light_total_dec = [l + b for l, b in zip(det_l_dec, det_b_dec)]
    
    psy_reward_total_inc = [r + b for r, b in zip(psy_r_inc, psy_b_inc)]
    psy_light_total_inc = [l + b for l, b in zip(psy_l_inc, psy_b_inc)]
    psy_reward_total_dec = [r + b for r, b in zip(psy_r_dec, psy_b_dec)]
    psy_light_total_dec = [l + b for l, b in zip(psy_l_dec, psy_b_dec)]
    
    # Calculate means and SEMs
    det_reward_inc_mean, det_reward_inc_sem = mean_sem(det_reward_total_inc)
    det_light_inc_mean, det_light_inc_sem = mean_sem(det_light_total_inc)
    det_reward_dec_mean, det_reward_dec_sem = mean_sem(det_reward_total_dec)
    det_light_dec_mean, det_light_dec_sem = mean_sem(det_light_total_dec)
    
    psy_reward_inc_mean, psy_reward_inc_sem = mean_sem(psy_reward_total_inc)
    psy_light_inc_mean, psy_light_inc_sem = mean_sem(psy_light_total_inc)
    psy_reward_dec_mean, psy_reward_dec_sem = mean_sem(psy_reward_total_dec)
    psy_light_dec_mean, psy_light_dec_sem = mean_sem(psy_light_total_dec)
    
    # Create summary DataFrame
    summary_data = {
        'Metric': [
            'Reward Responsive Increase (%)',
            'Light Responsive Increase (%)',
            'Reward Responsive Decrease (%)',
            'Light Responsive Decrease (%)',
            'Reward Only Increase (%)',
            'Light Only Increase (%)',
            'Both Increase (%)',
            'Reward Only Decrease (%)',
            'Light Only Decrease (%)',
            'Both Decrease (%)'
        ],
        'Detection_Mean': [
            det_reward_inc_mean, det_light_inc_mean, det_reward_dec_mean, det_light_dec_mean,
            np.mean(det_r_inc), np.mean(det_l_inc), np.mean(det_b_inc),
            np.mean(det_r_dec), np.mean(det_l_dec), np.mean(det_b_dec)
        ],
        'Detection_SEM': [
            det_reward_inc_sem, det_light_inc_sem, det_reward_dec_sem, det_light_dec_sem,
            mean_sem(det_r_inc)[1], mean_sem(det_l_inc)[1], mean_sem(det_b_inc)[1],
            mean_sem(det_r_dec)[1], mean_sem(det_l_dec)[1], mean_sem(det_b_dec)[1]
        ],
        'Psychophysics_Mean': [
            psy_reward_inc_mean, psy_light_inc_mean, psy_reward_dec_mean, psy_light_dec_mean,
            np.mean(psy_r_inc), np.mean(psy_l_inc), np.mean(psy_b_inc),
            np.mean(psy_r_dec), np.mean(psy_l_dec), np.mean(psy_b_dec)
        ],
        'Psychophysics_SEM': [
            psy_reward_inc_sem, psy_light_inc_sem, psy_reward_dec_sem, psy_light_dec_sem,
            mean_sem(psy_r_inc)[1], mean_sem(psy_l_inc)[1], mean_sem(psy_b_inc)[1],
            mean_sem(psy_r_dec)[1], mean_sem(psy_l_dec)[1], mean_sem(psy_b_dec)[1]
        ]
    }
    
    df = pd.DataFrame(summary_data)
    df['Difference'] = df['Psychophysics_Mean'] - df['Detection_Mean']
    df['Percent_Change'] = (df['Difference'] / df['Detection_Mean']) * 100
    
    # Save to CSV
    os.makedirs(save_dir, exist_ok=True)
    csv_path = join(save_dir, "Detection_vs_Psycho_Summary_Table.csv")
    df.to_csv(csv_path, index=False)
    
    # Print formatted table
    print("\n=== COMPREHENSIVE SUMMARY TABLE ===")
    print(f"{'Metric':<35} {'Detection':<15} {'Psychophysics':<15} {'Difference':<12} {'% Change':<10}")
    print("-" * 90)
    
    for _, row in df.iterrows():
        det_str = f"{row['Detection_Mean']:.2f} ± {row['Detection_SEM']:.2f}"
        psy_str = f"{row['Psychophysics_Mean']:.2f} ± {row['Psychophysics_SEM']:.2f}"
        diff_str = f"{row['Difference']:.2f}"
        pct_str = f"{row['Percent_Change']:.1f}%"
        
        print(f"{row['Metric']:<35} {det_str:<15} {psy_str:<15} {diff_str:<12} {pct_str:<10}")
    
    print(f"\nSummary table saved to: {csv_path}")
    return df
def plot_total_reward_light_detection_vs_psycho(detection_data, psycho_data, save_dir="."):
    """
    New figure: compares Detection vs Psychophysics for
      - All reward cells = reward only + both
      - All light cells  = light only  + both
    Combines increase and decrease. Includes paired stats with assumption checks
    and significance annotation.
    """
    # Extract arrays
    det_r_inc, det_l_inc, det_b_inc, det_r_dec, det_l_dec, det_b_dec = extract_data_arrays(detection_data)
    psy_r_inc, psy_l_inc, psy_b_inc, psy_r_dec, psy_l_dec, psy_b_dec = extract_data_arrays(psycho_data)
    if det_r_inc is None or psy_r_inc is None:
        print("Failed to extract data arrays. Check your data.")
        return

    # Combine across polarities
    det_reward_all = [ri + rd + bi + bd for ri, rd, bi, bd in zip(det_r_inc, det_r_dec, det_b_inc, det_b_dec)]
    det_light_all  = [li + ld + bi + bd for li, ld, bi, bd in zip(det_l_inc, det_l_dec, det_b_inc, det_b_dec)]
    psy_reward_all = [ri + rd + bi + bd for ri, rd, bi, bd in zip(psy_r_inc, psy_r_dec, psy_b_inc, psy_b_dec)]
    psy_light_all  = [li + ld + bi + bd for li, ld, bi, bd in zip(psy_l_inc, psy_l_dec, psy_b_inc, psy_b_dec)]

    # Means and SEMs
    det_reward_mean, det_reward_sem = mean_sem(det_reward_all)
    psy_reward_mean, psy_reward_sem = mean_sem(psy_reward_all)
    det_light_mean,  det_light_sem  = mean_sem(det_light_all)
    psy_light_mean,  psy_light_sem  = mean_sem(psy_light_all)

    # Stats with assumption check per group
    p_reward, test_label_reward, info_reward = choose_test(det_reward_all, psy_reward_all, paired=True, mode="wilcoxon")
    p_light, test_label_light, info_light  = choose_test(det_light_all,  psy_light_all, paired=True, mode="wilcoxon")


    print("\n=== New Total Reward and Light - Detection vs Psychophysics ===")
    print("REWARD CELLS (reward only + both):")
    rep = info_reward["report"]
    print(f"{rep['test']}: {rep['statistic_name']} = {rep['statistic_value']}, "
          f"df = {rep['df']}, p = {rep['p']}, "
          f"{rep['effect_size']['name']} = {rep['effect_size']['value']}")
    if rep["ci"]:
        print(f"95% CI for {rep['ci']['parameter']}: [{rep['ci']['low']:.3g}, {rep['ci']['high']:.3g}]")
    print(f"[stats] n={info_reward['n']}  shapiro_p={info_reward['shapiro_p']:.3g}  outliers={info_reward['outliers']}  chosen={test_label_reward}")
    stats_reward = {'p': p_reward, 'test': test_label_reward}
    print("\nLIGHT CELLS (light only + both):")
    rep = info_light["report"]
    print(f"{rep['test']}: {rep['statistic_name']} = {rep['statistic_value']}, "
          f"df = {rep['df']}, p = {rep['p']}, "
          f"{rep['effect_size']['name']} = {rep['effect_size']['value']}")
    if rep["ci"]:
        print(f"95% CI for {rep['ci']['parameter']}: [{rep['ci']['low']:.3g}, {rep['ci']['high']:.3g}]")
    print(f"[stats] n={info_light['n']}  shapiro_p={info_light['shapiro_p']:.3g}  outliers={info_light['outliers']}  chosen={test_label_light}")
    stats_light = {'p': p_light, 'test': test_label_light}

    # Plot
    fig, ax = plt.subplots(figsize=(10, 7))
    groups = ['All Reward', 'All Light']
    bar_width = 0.35
    x = np.arange(len(groups))

    # Bars: Detection left, Psychophysics right
    det_means = [det_reward_mean, det_light_mean]
    det_sems  = [det_reward_sem,  det_light_sem]
    psy_means = [psy_reward_mean, psy_light_mean]
    psy_sems  = [psy_reward_sem,  psy_light_sem]

    det_bars = ax.bar(x - bar_width/2, det_means, bar_width, yerr=det_sems,
                      label='Detection', color='steelblue', edgecolor='black', alpha=0.85, capsize=4)
    psy_bars = ax.bar(x + bar_width/2, psy_means, bar_width, yerr=psy_sems,
                      label='Psychophysics', color='indianred', edgecolor='black', alpha=0.85, capsize=4)

    # Connect per-mouse paired points for each group
    # x positions for lines
    x_det = x - bar_width/2
    x_psy = x + bar_width/2

    # Reward group index 0
    for i in range(len(det_reward_all)):
        ax.plot([x_det[0], x_psy[0]], [det_reward_all[i], psy_reward_all[i]],
                color='gray', alpha=0.5, linewidth=1, linestyle='--', marker='o', zorder=3)
    # Light group index 1
    for i in range(len(det_light_all)):
        ax.plot([x_det[1], x_psy[1]], [det_light_all[i], psy_light_all[i]],
                color='gray', alpha=0.5, linewidth=1, linestyle='--', marker='o', zorder=3)

    # Axis cosmetics
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=12)
    ax.set_ylabel('Percentage of total neurons', fontsize=13)
    ax.set_title('Detection vs Psychophysics - All Reward vs All Light', fontsize=15, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend(loc='upper right')

    # Significance brackets
    # Compute tops to place brackets a bit above the taller bar in each group
    reward_top = max(det_means[0] + det_sems[0], psy_means[0] + psy_sems[0])
    light_top  = max(det_means[1] + det_sems[1], psy_means[1] + psy_sems[1])

    # Set a reasonable ylim to have headroom for brackets
    ymax = max(reward_top, light_top) * 1.20 if max(reward_top, light_top) > 0 else 1.0
    ymin = 0.0
    ax.set_ylim(ymin, ymax)

    add_sig_bracket(ax, x_det[0], x_psy[0], reward_top * 1.03, stats_reward['p'])
    add_sig_bracket(ax, x_det[1], x_psy[1], light_top  * 1.03, stats_light['p'])

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    out_path = join(save_dir, "Detection_vs_Psycho_TotalRewardTotalLight.pdf")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved figure: {out_path}")


if __name__ == "__main__":
    # File paths for detection analysis
    # detection_pkl_files = [
    #     r"Z:\2p_data\OI230\250326\ML\data\processed_data\auROC_z_no_sub_history\comparison_fractions_250326.pkl",
    #     r"Z:\2p_data\OI151\240826\ML\data\processed_data\auROC_z_no_sub_history\comparison_fractions_240826.pkl",
    #     r"Z:\2p_data\OI208\250224\ML\data\processed_data\auROC_z_no_sub_history\comparison_fractions_250224.pkl",
    #     r"Z:\2p_data\OI152\240829\ML\data\processed_data\auROC_z_no_sub_history\comparison_fractions_240829.pkl",
    # ]
    # #
    # # # File paths for psychophysics analysis
    # psycho_pkl_files = [
    #     # file paths:
    #     r"Z:\2p_data\OI230\250401\ML\data\processed_data\auROC_v2\comparison_fractions_250401.pkl",
    #     r"Z:\2p_data\OI151\241030\ML\data\processed_data\auROC_v2\comparison_fractions_241030.pkl",
    #     r"Z:\2p_data\OI208\250310\ML\data\processed_data\auROC_v2\comparison_fractions_250310.pkl",
    #     r"Z:\2p_data\OI152\240926\ML\data\processed_data\auROC_v2\comparison_fractions_240926.pkl",
    # ]
    detection_pkl_files = [
        r"Z:\2p_data\OI116\240519\ML\data\processed_data\auROC_z_no_sub_history\comparison_fractions_240519.pkl",
        r"Z:\2p_data\OI124\240530\ML\data\processed_data\auROC_z_no_sub_history\comparison_fractions_240530.pkl",
        r"Z:\2p_data\OI151\240826\ML\data\processed_data\auROC_z_no_sub_history\comparison_fractions_240826.pkl",
        r"Z:\2p_data\OI208\250224\ML\data\processed_data\auROC_z_no_sub_history\comparison_fractions_250224.pkl",
        r"Z:\2p_data\OI152\240829\ML\data\processed_data\auROC_z_no_sub_history\comparison_fractions_240829.pkl",
    ]

    # File paths for psychophysics analysis
    psycho_pkl_files = [
        r"Z:\2p_data\OI151\240827\ML\data\processed_data\auROC_freq_psycho_new\comparison_fractions_240827.pkl",
        r"Z:\2p_data\OI152\241003\ML\data\processed_data\auROC_freq_psycho_new\comparison_fractions_241003.pkl",
        r"Z:\2p_data\OI124\240610\ML\data\processed_data\auROC_freq_psycho_new\comparison_fractions_240610.pkl",
        r"Z:\2p_data\OI116\240529\ML\data\processed_data\auROC_freq_psycho_new\comparison_fractions_240529.pkl",
        r"Z:\2p_data\OI208\250227\ML\data\processed_data\auROC_freq_psycho_new\comparison_fractions_250227.pkl",
    ]
    save_dir = r"Z:\Stavsht\Omer_OXTR\detection_vs_psycho_comparison_v3"
    
    if detection_pkl_files and psycho_pkl_files:
        print("Loading data from both analyses...")
        detection_data, psycho_data = load_comparison_data(detection_pkl_files, psycho_pkl_files)
        
        print(f"\nDetection data keys: {list(detection_data.keys())}")
        print(f"Psychophysics data keys: {list(psycho_data.keys())}")
        print(f"Number of mice in detection: {len(detection_data.get('Hit(reward)_LightOn_Increase', []))}")
        print(f"Number of mice in psychophysics: {len(psycho_data.get('Hit(reward)_LightOn_Increase', []))}")
        
        # Generate comparison plots and analyses
        plot_side_by_side_comparison(detection_data, psycho_data, save_dir=save_dir)
        plot_side_by_side_venn_comparison(detection_data, psycho_data, save_dir=save_dir)
        plot_connected_mice_bars(detection_data, psycho_data, save_dir=save_dir)  # New connected mice plot
        perform_statistical_comparison(detection_data, psycho_data, save_dir=save_dir)
        summary_df = create_summary_table(detection_data, psycho_data, save_dir=save_dir)
        plot_total_reward_light_detection_vs_psycho(detection_data, psycho_data, save_dir=save_dir)

        print(f"\nAll comparison plots and analyses saved to: {save_dir}")
    else:
        print("Please update the file paths for both detection and psychophysics pickle files")