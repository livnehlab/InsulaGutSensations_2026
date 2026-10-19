# topography_light_reward_both_refined_with_quant_allpairs.py
import os
from os.path import join
from pathlib import Path
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from scipy.spatial import distance_matrix
from sklearn.metrics import auc
import pandas as pd

matplotlib.use('Qt5Agg')
plt.rcParams["font.family"] = "Arial"
RADII_FIXED = np.array([8, 12, 16, 24, 32, 48, 64, 96, 128, 160, 180], dtype=np.float64)
from scipy.spatial import distance_matrix
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
# ---------- Illustrator-friendly PDF saving ----------
import matplotlib as mpl
def setup_matplotlib_for_illustrator(font_family="Arial"):
    mpl.rcParams.update({
        "font.family": font_family,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "mathtext.default": "regular",
    })

def save_fig_pdf(path_base,
                 fig=None,
                 tight=True,
                 dpi=300,
                 facecolor="white",
                 transparent=False,
                 pad_inches=0.02,
                 rasterize_threshold=None):
    if fig is None:
        fig = plt.gcf()
    if rasterize_threshold is not None:
        for ax in fig.axes:
            for coll in ax.collections:
                if hasattr(coll, "get_offsets"):
                    if len(coll.get_offsets()) > rasterize_threshold:
                        coll.set_rasterized(True)
            for line in ax.lines:
                if len(getattr(line, "get_xdata", lambda: [])()) > rasterize_threshold:
                    line.set_rasterized(True)
    outdir = Path(path_base).parent
    outdir.mkdir(parents=True, exist_ok=True)
    save_kwargs = {"dpi": dpi, "facecolor": facecolor, "transparent": transparent}
    if tight:
        save_kwargs.update(bbox_inches="tight", pad_inches=pad_inches)
    path = f"{path_base}.pdf"
    fig.savefig(path, **save_kwargs)
    return path

setup_matplotlib_for_illustrator("Arial")

# =========================
# Settings for single run
# =========================
mouse = 'OI208'
day = '250224'

ROOT_PATH = fr'Z:\2p_data\{mouse}\{day}\ML\data'
load_loc  = fr'Z:\2p_data\{mouse}\{day}\ML\data\processed_data\auROC_z_no_sub_history'
save_loc  = fr'Z:\2p_data\{mouse}\{day}\ML\data\processed_data\topography_light_reward_v2'
s2p_dir   = fr'Z:\2p_data\{mouse}\{day}\{mouse}_run1\suite2p\plane0'
save_loc_all = r'Z:\Stavsht\Omer_OXTR\topography_light_reward_v2'  # shared for cross-mouse plots

os.makedirs(save_loc, exist_ok=True)

# recording params for building trial indices
rec_hz = 15
pre_trial_length = 4 * rec_hz
n_priming_trials = 23

frequency_map = {6: 0, 7: 15, 8: 10, 9: 7.5, 10: 5, 11: 3, 12: 1, 13: 0.5}
target_freqs = [0.5, 1, 3, 5, 7.5, 10, 15]

# Plot look
COLORS = dict(light="cyan", reward="orange", both="lime")
BG_ENHANCE = "rescale"
OUTLINE_LW = 0.5
ALPHA = 0.45
SAVE_FIG = join(save_loc, f'topography_light_reward_both_{mouse}_{day}.pdf')

# Consistent group naming and colors
GROUP_ORDER  = ["light", "reward", "both"]
GROUP_LABELS = {"light": "Light only", "reward": "Reward only", "both": "Both"}
GROUP_COLORS = {"light": COLORS["light"], "reward": COLORS["reward"], "both": COLORS["both"]}

def format_group_label(name, n=None):
    base = GROUP_LABELS[name]
    return f"{base} (n={n})" if n is not None else base

# =========================
# Minimal helpers you used
# =========================
def build_freq_indices(conditions_played, frequency_map, target_freqs):
    freq_indices = {}
    for code, freq in frequency_map.items():
        inds = [i for i, x in enumerate(conditions_played) if x == code]
        if freq in freq_indices:
            freq_indices[freq].extend(inds)
        else:
            freq_indices[freq] = inds
    freq_indices = {freq: inds for freq, inds in freq_indices.items() if freq in target_freqs}
    return freq_indices

def load_event_data(load_loc, target_freqs):
    with open(join(load_loc, 'res_reward_vs_lightHit.pkl'), 'rb') as f:
        res_reward_vs_lightHit = pickle.load(f)
    with open(join(load_loc, 'res_lick_on_vs_lightHit.pkl'), 'rb') as f:
        res_lick_on_vs_lightHit = pickle.load(f)
    with open(join(load_loc, 'auROC_results_hit.pkl'), 'rb') as f:
        auROC_results_hit = pickle.load(f)

    event_data_by_freq = {}
    results_by_freq = {}
    results_vs_hit_by_freq = {}
    for freq in target_freqs:
        freq_str = str(freq)
        with open(join(load_loc, f'res{freq_str}.pkl'), 'rb') as f:
            event_data_by_freq[freq] = pickle.load(f)
        with open(join(load_loc, f'res{freq_str}.pkl'), 'rb') as f:
            res = pickle.load(f)
        with open(join(load_loc, f'res_CR_vs_lightHit_{freq_str}.pkl'), 'rb') as f:
            res_vs_hit = pickle.load(f)
        results_by_freq[freq] = res
        results_vs_hit_by_freq[freq] = res_vs_hit

    return event_data_by_freq, results_by_freq, results_vs_hit_by_freq, \
           res_reward_vs_lightHit, res_lick_on_vs_lightHit, auROC_results_hit

def build_freq_matrix(n_neurons, target_freqs, results_by_freq, results_vs_hit_by_freq):
    freq_matrix = np.full((n_neurons, len(target_freqs)), np.nan)
    for j, freq in enumerate(target_freqs):
        res = results_by_freq[freq]
        res_vs_hit = results_vs_hit_by_freq[freq]
        for i in range(n_neurons):
            if np.isnan(res['aurocA'][i]) or np.isnan(res_vs_hit['aurocA'][i]):
                continue
            if (res['pvalA'][i] == 'above_95') and (res_vs_hit['pvalA'][i] == 'above_95'):
                freq_matrix[i, j] = res['aurocA'][i]
            elif (res['pvalA'][i] == 'below_5') and (res_vs_hit['pvalA'][i] == 'below_5'):
                freq_matrix[i, j] = res['aurocA'][i]
    return freq_matrix

def find_reward_significant_neurons(res_reward_vs_lightHit, res_lick_on_vs_lightHit, auROC_results_hit):
    reward_significant_increase = []
    reward_significant_decrease = []
    lick_on_significant_increase = []
    lick_on_significant_decrease = []

    for i in range(len(auROC_results_hit['reward']['auroc'])):
        if auROC_results_hit['reward']['p_value'][i] == 'above_95' and res_reward_vs_lightHit['classification'][i] == 'above_95':
            reward_significant_increase.append(i)
        elif auROC_results_hit['reward']['p_value'][i] == 'below_5' and res_reward_vs_lightHit['classification'][i] == 'below_5':
            reward_significant_decrease.append(i)

    for i in range(len(auROC_results_hit['lick_onset']['auroc'])):
        if auROC_results_hit['lick_onset']['p_value'][i] == 'above_95' and res_lick_on_vs_lightHit['classification'][i] == 'above_95':
            lick_on_significant_increase.append(i)
        elif auROC_results_hit['lick_onset']['p_value'][i] == 'below_5' and res_lick_on_vs_lightHit['classification'][i] == 'below_5':
            lick_on_significant_decrease.append(i)

    general_reward_significant_increase = set(reward_significant_increase).union(set(lick_on_significant_increase))
    general_reward_significant_decrease = set(reward_significant_decrease).union(set(lick_on_significant_decrease))
    general_reward_significant = general_reward_significant_increase.union(general_reward_significant_decrease)

    return general_reward_significant, list(general_reward_significant_increase), list(general_reward_significant_decrease)

def find_light_significant_neurons(freq_matrix):
    n_neurons, _ = freq_matrix.shape
    light_significant_increase = []
    light_significant_decrease = []
    for i in range(n_neurons):
        valid = freq_matrix[i, ~np.isnan(freq_matrix[i, :])]
        if len(valid) == 0:
            continue
        if np.any(valid > 0.5):
            light_significant_increase.append(i)
        if np.any(valid < 0.5):
            light_significant_decrease.append(i)
    return light_significant_increase, light_significant_decrease

# =========================
# Suite2p helpers
# =========================
def load_suite2p(s2p_plane):
    s2p_plane = Path(s2p_plane)
    ops = np.load(s2p_plane / 'ops.npy', allow_pickle=True).item()
    stat = np.load(s2p_plane / 'stat.npy', allow_pickle=True)
    iscell = np.load(s2p_plane / 'iscell_final.npy')
    bg = ops.get('meanImg', ops.get('meanImgE'))
    if bg is None:
        raise ValueError("meanImg not found in ops.npy")
    return ops, stat, iscell, bg

def enhance(img, method='rescale'):
    try:
        from skimage import exposure
    except Exception:
        return img
    if method == 'rescale':
        p2, p98 = np.percentile(img, (2, 98))
        return exposure.rescale_intensity(img, in_range=(p2, p98))
    if method == 'clahe':
        return exposure.equalize_adapthist(img / max(img.max(), 1e-9), clip_limit=0.03)
    return img

def contour_from_pixels(xpix, ypix, Ly, Lx):
    try:
        from skimage.draw import polygon
        from skimage.measure import find_contours
        mask = np.zeros((Ly, Lx), dtype=bool)
        rr, cc = polygon(ypix, xpix, shape=mask.shape)
        mask[rr, cc] = True
        conts = find_contours(mask.astype(float), 0.5)
        if not conts:
            raise RuntimeError
        c = max(conts, key=lambda a: a.shape[0])
        return np.c_[c[:,1], c[:,0]]
    except Exception:
        try:
            from scipy.spatial import ConvexHull
            pts = np.column_stack([xpix, ypix])
            if pts.shape[0] < 3:
                cx, cy = pts[:,0].mean(), pts[:,1].mean()
                return np.array([[cx-0.5, cy-0.5],[cx+0.5, cy-0.5],[cx, cy+0.5]])
            hull = ConvexHull(pts)
            verts = pts[hull.vertices]
            return np.vstack([verts, verts[0]])
        except Exception:
            cx, cy = np.mean(xpix), np.mean(ypix)
            return np.array([[cx-0.5, cy-0.5],[cx+0.5, cy-0.5],[cx, cy+0.5]])

# =========================
# Three-group plotting and stats
# =========================
def plot_three_class_outlines(ops, stat, bg, ids_light_only, ids_reward_only, ids_both,
                              colors=GROUP_COLORS, bg_enhance=BG_ENHANCE, alpha=ALPHA, outline_lw=OUTLINE_LW,
                              save_path=None):
    Ly, Lx = ops['Ly'], ops['Lx']
    bg_plot = enhance(bg, bg_enhance) if bg_enhance else bg

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(bg_plot, cmap='gray')
    ax.axis('off')

    def add_poly(ids, facecolor):
        if len(ids) == 0:
            return 0
        polys = []
        for i in ids:
            s = stat[i]
            verts = contour_from_pixels(s['xpix'], s['ypix'], Ly, Lx)
            polys.append(verts)
        coll = PolyCollection(polys, facecolors=facecolor, edgecolors='k',
                              linewidths=outline_lw, alpha=alpha)
        ax.add_collection(coll)
        return len(ids)

    nL = add_poly(ids_light_only, colors['light'])
    nR = add_poly(ids_reward_only, colors['reward'])
    nB = add_poly(ids_both,       colors['both'])

    leg_handles = []
    for key, n in [("light", nL), ("reward", nR), ("both", nB)]:
        h, = ax.plot([], [], color=colors[key], lw=4, label=format_group_label(key, n))
        leg_handles.append(h)

    ax.legend(handles=leg_handles, loc='lower right', frameon=True)
    ax.set_title(f'{mouse} {day}  {GROUP_LABELS["light"]} vs {GROUP_LABELS["reward"]} vs {GROUP_LABELS["both"]}')

    if save_path:
        save_fig_pdf(save_path)
    plt.show()

def _coords_from_ids(stat, ids):
    return np.array([[np.median(stat[i]['ypix']), np.median(stat[i]['xpix'])] for i in ids], float)

def three_group_coords(stat, ids_light_only, ids_reward_only, ids_both):
    id_map = {"light": np.asarray(ids_light_only, int),
              "reward": np.asarray(ids_reward_only, int),
              "both":   np.asarray(ids_both, int)}
    coords_list = []
    label_list  = []
    for gi, gname in enumerate(GROUP_ORDER):
        ids = id_map[gname]
        if len(ids) == 0:
            continue
        coords_g = np.array([[np.median(stat[i]['ypix']), np.median(stat[i]['xpix'])] for i in ids], float)
        coords_list.append(coords_g)
        label_list.append(np.full(len(coords_g), gi, int))
    coords = np.vstack(coords_list) if coords_list else np.empty((0,2))
    labels = np.concatenate(label_list) if label_list else np.empty((0,), int)
    group_map = {name: idx for idx, name in enumerate(GROUP_ORDER)}
    return coords, labels, group_map

def plot_within_across_hist_3groups(coords, labels, group_names=tuple(GROUP_ORDER),
                                    save_loc=None, mouse=None, day=None):
    D = distance_matrix(coords, coords)
    np.fill_diagonal(D, np.nan)
    cmap = {g: GROUP_COLORS[g] for g in GROUP_ORDER}

    for gname, gidx in zip(group_names, [0,1,2]):
        mask_g = labels == gidx
        within = D[np.ix_(mask_g, mask_g)]
        across = D[np.ix_(mask_g, ~mask_g)]
        within_vals = within[np.triu_indices(np.sum(mask_g), k=1)]
        across_vals = across.flatten()

        plt.figure(figsize=(6,4))
        plt.hist(within_vals[~np.isnan(within_vals)], bins=40, alpha=0.6, density=True,
                 label=f"Within {GROUP_LABELS[gname]}", color=cmap[gname])
        plt.hist(across_vals[~np.isnan(across_vals)], bins=40, alpha=0.5, density=True,
                 label=f"{GROUP_LABELS[gname]} to others", color="gray")
        plt.xlabel("Distance (px)")
        plt.ylabel("Density")
        plt.title(f"Within vs across distance: {GROUP_LABELS[gname]}")
        plt.legend(frameon=True, title="Pairs")
        plt.tight_layout()
        if save_loc:
            Path(save_loc).mkdir(parents=True, exist_ok=True)
            save_fig_pdf(join(save_loc, f"{mouse}_{day}_dist_hist_{gname}"))
        plt.show()

def within_across_gap_3groups(coords, labels):
    D = distance_matrix(coords, coords)
    np.fill_diagonal(D, np.nan)
    gaps = []
    for g in [0,1,2]:
        mask_g = labels == g
        if np.sum(mask_g) < 2:
            gaps.append(np.nan)
            continue
        within = D[np.ix_(mask_g, mask_g)]
        across = D[np.ix_(mask_g, ~mask_g)]
        within_vals = within[np.triu_indices(np.sum(mask_g), k=1)]
        across_vals = across.flatten()
        gaps.append(np.nanmean(within_vals) - np.nanmean(across_vals))
    return np.array(gaps)

def perm_test_within_across_3groups(coords, labels, n_perm=1000, seed=0):
    rng = np.random.default_rng(seed)
    gaps_obs = within_across_gap_3groups(coords, labels)
    counts = np.bincount(labels, minlength=3)
    N = len(labels)
    perm_gaps = np.empty((n_perm, 3))
    for k in range(n_perm):
        perm = rng.permutation(N)
        n0, n1, n2 = counts
        perm_labels = np.r_[np.zeros(n0, int), np.ones(n1, int), np.full(n2, 2, int)][np.argsort(perm)]
        perm_gaps[k] = within_across_gap_3groups(coords, perm_labels)
    perm_mean = np.nanmean(perm_gaps, axis=0)
    pvals = np.nanmean(np.abs(perm_gaps - perm_mean) >= np.abs(gaps_obs - perm_mean), axis=0)
    return gaps_obs, perm_gaps, pvals

def plot_gap_summary_3groups(gaps_obs, perm_gaps, pvals, save_loc=None, mouse=None, day=None):
    group_names = [GROUP_LABELS[g] for g in GROUP_ORDER]
    colors = [GROUP_COLORS[g] for g in GROUP_ORDER]
    means = np.nanmean(perm_gaps, axis=0)
    ci = np.nanpercentile(perm_gaps, [2.5,97.5], axis=0)

    plt.figure(figsize=(7,4))
    y = np.arange(3)
    plt.hlines(y, ci[0], ci[1], colors="gray", lw=6, alpha=0.35, label="Null 95% CI")
    plt.plot(means, y, "x", color="gray", label="Null mean")
    plt.scatter(gaps_obs, y, c=colors, s=80, edgecolors="k", zorder=3, label="Observed")
    for i, p in enumerate(pvals):
        plt.text(gaps_obs[i], y[i]+0.1, f"p={p:.3f}", ha="center", va="bottom", fontsize=9)
    plt.axvline(0, color="k", ls="--", lw=1)
    plt.yticks(y, group_names)
    plt.xlabel("Within minus across distance (px)")
    plt.title("3-group distance gap")
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    if save_loc:
        save_fig_pdf(join(save_loc, f"{mouse}_{day}_gap_summary_3groups"))
    plt.show()

def nearest_neighbor_3groups(coords, labels):
    means = np.full((3,3), np.nan)
    for src in [0,1,2]:
        src_mask = labels == src
        if np.sum(src_mask) == 0:
            continue
        for tgt in [0,1,2]:
            tgt_mask = labels == tgt
            if np.sum(tgt_mask) == 0:
                continue
            D = distance_matrix(coords[src_mask], coords[tgt_mask])
            means[src, tgt] = np.nanmean(np.min(D, axis=1))
    return means

def perm_test_nn_3groups(coords, labels, n_perm=1000, seed=1):
    rng = np.random.default_rng(seed)
    obs = nearest_neighbor_3groups(coords, labels)
    counts = np.bincount(labels, minlength=3)
    N = len(labels)
    permmats = np.empty((n_perm, 3, 3))
    for k in range(n_perm):
        perm = rng.permutation(N)
        n0, n1, n2 = counts
        perm_labels = np.r_[np.zeros(n0, int), np.ones(n1, int), np.full(n2, 2, int)][np.argsort(perm)]
        permmats[k] = nearest_neighbor_3groups(coords, perm_labels)
    perm_mean = np.nanmean(permmats, axis=0)
    pvals = np.nanmean(np.abs(permmats - perm_mean) >= np.abs(obs - perm_mean), axis=0)
    return obs, permmats, pvals

def plot_nn_heatmap(obs_nn, p_nn=None, save_loc=None, mouse=None, day=None, title="Mean nearest neighbor"):
    import matplotlib.ticker as mticker
    group_names = [GROUP_LABELS[g] for g in GROUP_ORDER]
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(obs_nn, origin="upper")
    ax.set_xticks([0,1,2]); ax.set_xticklabels(group_names)
    ax.set_yticks([0,1,2]); ax.set_yticklabels(group_names)
    ax.set_xlabel("Target group")
    ax.set_ylabel("Source group")
    ax.set_title(title + " (px)")
    for i in range(3):
        for j in range(3):
            val = obs_nn[i, j]
            if np.isnan(val):
                txt = "NA"
            else:
                if p_nn is not None and not np.isnan(p_nn[i, j]) and p_nn[i, j] < 0.05:
                    txt = f"{val:.1f}*"
                else:
                    txt = f"{val:.1f}"
            ax.text(j, i, txt, ha="center", va="center", color="white", fontsize=10, weight="bold")
    fig.colorbar(im, ax=ax, format=mticker.FormatStrFormatter("%.1f"))
    plt.tight_layout()
    if save_loc:
        save_fig_pdf(join(save_loc, f"{mouse}_{day}_nn_heatmap_3groups"))
    plt.show()

def radial_clustering_3groups(coords, labels, radii=None):
    """
    Observed curves only, using CSR adjacency per radius.
    same[g, r] - mean same-label neighbor count for group g at radius r
    other[g, r] - mean other-label neighbor count for group g at radius r (over the same focals)
    """
    coords = np.asarray(coords, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)
    n = len(coords)
    if n == 0:
        return {"radii": np.array([]), "same": np.zeros((3,0)), "other": np.zeros((3,0))}

    tree = cKDTree(coords)

    if radii is None:
        # small adaptive fallback
        kq = min(50, max(2, n))
        Dk, _ = tree.query(coords, k=kq)
        nonzero = Dk[:, 1:].ravel()
        nonzero = nonzero[nonzero > 0]
        if nonzero.size > 0:
            lo = float(max(5.0, np.quantile(nonzero, 0.05)))
            hi = float(max(lo + 1.0, np.quantile(nonzero, 0.95)))
            radii = np.linspace(lo, hi, 12, dtype=np.float64)
        else:
            radii = np.array([5.0, 6.0, 7.0], dtype=np.float64)

    radii = np.asarray(radii, dtype=np.float64)
    R = len(radii)

    # Build CSR adjacencies and degrees
    A_list, deg_list = [], []
    for r in radii:
        nbrs = tree.query_ball_tree(tree, float(r))
        rows, cols = [], []
        for i, nb in enumerate(nbrs):
            if not nb:
                continue
            if i in nb:
                nb = [j for j in nb if j != i]
            if nb:
                rows.extend([i]*len(nb))
                cols.extend(nb)
        if rows:
            data = np.ones(len(rows), dtype=np.float64)
            A = csr_matrix((data, (np.asarray(rows), np.asarray(cols))), shape=(n, n))
        else:
            A = csr_matrix((n, n), dtype=np.float64)
        A_list.append(A)
        deg_list.append(np.asarray(A.sum(axis=1)).ravel().astype(np.float64))

    # same and other curves for each group g, averaged over focals in group g
    same_curves = np.zeros((3, R), dtype=np.float64)
    other_curves = np.zeros((3, R), dtype=np.float64)

    for g in (0, 1, 2):
        focals = np.where(labels == g)[0]
        if focals.size == 0:
            continue
        yg = (labels == g).astype(np.float64)  # indicator vector for group g
        for r_idx, A in enumerate(A_list):
            deg = deg_list[r_idx]
            Sg = A @ yg                       # for each node, neighbors labeled g
            same_i = Sg                       # since yg is 1 when target group matches
            m_same = float(np.mean(same_i[focals]))
            same_curves[g, r_idx] = m_same
            other_curves[g, r_idx] = float(np.mean(deg[focals]) - m_same)

    return {"radii": radii, "same": same_curves, "other": other_curves}


def radial_clustering_perm_3groups(coords, labels, radii=None, n_perm=500, seed=0):
    """
    Streaming permutations for 3 groups.
    - Returns curves_obs, lightweight placeholders for same_perm/other_perm, and AUC stats
    """
    rng_perm = np.random.default_rng(seed)

    coords = np.asarray(coords, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)
    n = len(coords)
    if n == 0:
        empty_curves = {"radii": np.array([]), "same": np.zeros((3,0)), "other": np.zeros((3,0))}
        return empty_curves, np.array([]), np.array([]), np.array([np.nan]*3), np.array([[]]*3), np.array([np.nan]*3)

    # observed curves
    curves_obs = radial_clustering_3groups(coords, labels, radii=radii)
    radii = curves_obs["radii"]
    R = len(radii)

    # observed AUC per group
    obs_auc = np.zeros(3, dtype=np.float64)
    for g in (0, 1, 2):
        obs_auc[g] = auc(radii, curves_obs["same"][g] - curves_obs["other"][g])

    # permutation AUCs per group - streaming, do not store curves
    # preserve counts per group
    counts = np.bincount(labels, minlength=3)
    perm_auc = np.zeros((3, n_perm), dtype=np.float64)

    # Prebuild adjacency once to reuse in perms
    tree = cKDTree(coords)
    A_list, deg_list = [], []
    for r in radii:
        nbrs = tree.query_ball_tree(tree, float(r))
        rows, cols = [], []
        for i, nb in enumerate(nbrs):
            if not nb:
                continue
            if i in nb:
                nb = [j for j in nb if j != i]
            if nb:
                rows.extend([i]*len(nb))
                cols.extend(nb)
        if rows:
            data = np.ones(len(rows), dtype=np.float64)
            A = csr_matrix((data, (np.asarray(rows), np.asarray(cols))), shape=(n, n))
        else:
            A = csr_matrix((n, n), dtype=np.float64)
        A_list.append(A)
        deg_list.append(np.asarray(A.sum(axis=1)).ravel().astype(np.float64))

    # For each permutation, build three indicator vectors y0,y1,y2 and compute AUCs
    base_labels = np.r_[np.zeros(counts[0], int),
                        np.ones(counts[1], int),
                        np.full(counts[2], 2, int)]
    for k in range(n_perm):
        perm = rng_perm.permutation(n)
        perm_labels = base_labels[np.argsort(perm)]  # random reassignment with fixed counts

        # Indicator vectors for each group in this perm
        y0 = (perm_labels == 0).astype(np.float64)
        y1 = (perm_labels == 1).astype(np.float64)
        y2 = (perm_labels == 2).astype(np.float64)
        foc0 = np.where(perm_labels == 0)[0]
        foc1 = np.where(perm_labels == 1)[0]
        foc2 = np.where(perm_labels == 2)[0]

        # Build curves by streaming across radii
        curve0 = np.zeros(R, dtype=np.float64)
        curve1 = np.zeros(R, dtype=np.float64)
        curve2 = np.zeros(R, dtype=np.float64)

        for r_idx, A in enumerate(A_list):
            deg = deg_list[r_idx]
            S0 = A @ y0; S1 = A @ y1; S2 = A @ y2

            if foc0.size:
                m_same0 = float(np.mean(S0[foc0]))
                curve0[r_idx] = m_same0 - (float(np.mean(deg[foc0])) - m_same0)
            else:
                curve0[r_idx] = np.nan

            if foc1.size:
                m_same1 = float(np.mean(S1[foc1]))
                curve1[r_idx] = m_same1 - (float(np.mean(deg[foc1])) - m_same1)
            else:
                curve1[r_idx] = np.nan

            if foc2.size:
                m_same2 = float(np.mean(S2[foc2]))
                curve2[r_idx] = m_same2 - (float(np.mean(deg[foc2])) - m_same2)
            else:
                curve2[r_idx] = np.nan

        # AUCs for this perm
        perm_auc[0, k] = np.trapz(curve0, x=radii)
        perm_auc[1, k] = np.trapz(curve1, x=radii)
        perm_auc[2, k] = np.trapz(curve2, x=radii)

    # p values per group
    pvals = np.zeros(3, dtype=np.float64)
    for g in (0, 1, 2):
        good = np.isfinite(perm_auc[g])
        if not np.isfinite(obs_auc[g]) or np.sum(good) == 0:
            pvals[g] = np.nan
        else:
            pa = perm_auc[g, good]
            pvals[g] = np.mean(np.abs(pa - pa.mean()) >= np.abs(obs_auc[g] - pa.mean()))

    # same_perm and other_perm are not used downstream for plotting summary,
    # return lightweight placeholders to keep call sites unchanged
    return curves_obs, np.array([]), np.array([]), obs_auc, perm_auc, pvals


def plot_radial_curves_3groups(curves, save_loc=None, mouse=None, day=None):
    names  = [GROUP_LABELS[g] for g in GROUP_ORDER]
    colors = {g: GROUP_COLORS[g] for g in GROUP_ORDER}
    R = curves["radii"]

    for g, name in enumerate(names):
        key = GROUP_ORDER[g]
        plt.figure(figsize=(6,4))
        plt.plot(R, curves["same"][g], label="same", color=colors[key])
        plt.plot(R, curves["other"][g], label="other", color="gray")
        plt.xlabel("Radius (px)")
        plt.ylabel("Mean # neighbors")
        plt.title(f"Radial clustering: {name}")
        plt.legend()
        plt.tight_layout()
        if save_loc:
            save_fig_pdf(join(save_loc, f"{mouse}_{day}_radial_{GROUP_ORDER[g]}"))
        plt.show()

def plot_radial_auc_summary(obs_auc, perm_auc, pvals, save_loc=None, mouse=None, day=None):
    names  = [GROUP_LABELS[g] for g in GROUP_ORDER]
    colors = [GROUP_COLORS[g] for g in GROUP_ORDER]
    means = np.mean(perm_auc, axis=1)
    ci_lo = np.percentile(perm_auc, 2.5, axis=1)
    ci_hi = np.percentile(perm_auc, 97.5, axis=1)

    plt.figure(figsize=(7,4))
    y = np.arange(3)
    plt.hlines(y, ci_lo, ci_hi, colors="gray", lw=6, alpha=0.35, label="Null 95% CI")
    plt.plot(means, y, "x", color="gray", label="Null mean")
    plt.scatter(obs_auc, y, c=colors, s=80, edgecolors="k", zorder=3, label="Observed")
    for i, p in enumerate(pvals):
        plt.text(obs_auc[i], y[i]+0.1, f"p={p:.3f}", ha="center", va="bottom", fontsize=9)
    plt.axvline(0, color="k", ls="--", lw=1)
    plt.yticks(y, names)
    plt.xlabel("AUC of (same - other) across radii")
    plt.title("Radial clustering summary")
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    if save_loc:
        save_fig_pdf(join(save_loc, f"{mouse}_{day}_radial_auc_summary"))
    plt.show()

# =========================
# Two-group toolkit
# =========================
def plot_perm_hist(perm_vals, observed, p_two_sided, xlabel, title, save_loc=None, mouse_id=None, day_str=None, fname_stub="perm_hist"):
    plt.figure(figsize=(6.5, 4))
    plt.hist(perm_vals, bins=40, alpha=0.7, color='gray', edgecolor='black')
    plt.axvline(observed, color='red', linewidth=2, label=f"Observed = {observed:.2f}")
    plt.axvline(np.mean(perm_vals), color='blue', linestyle='--', label=f"Null mean = {np.mean(perm_vals):.2f}")
    plt.fill_betweenx(
        y=[0, plt.gca().get_ylim()[1]],
        x1=min(observed, np.mean(perm_vals)),
        x2=max(observed, np.mean(perm_vals)),
        color='red', alpha=0.15
    )
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.title(f"{title}\nTwo-sided p = {p_two_sided:.4f}")
    plt.legend(frameon=True)
    plt.tight_layout()
    if save_loc is not None:
        Path(save_loc).mkdir(parents=True, exist_ok=True)
        save_fig_pdf(join(save_loc, f'{mouse_id}_{day_str}_{fname_stub}'))
    plt.close()

def plot_within_group_histograms(coords_A,
                                 coords_B,
                                 bins=40,
                                 save_loc=None,
                                 mouse_id=None,
                                 day_str=None,
                                 label_A="Group A",
                                 label_B="Group B",
                                 color_A=None,
                                 color_B=None):
    # sensible defaults if colors not provided
    if color_A is None:
        color_A = "cyan"
    if color_B is None:
        color_B = "magenta"

    d_A = distance_matrix(coords_A, coords_A)
    d_B = distance_matrix(coords_B, coords_B)
    d_A = d_A[np.triu_indices_from(d_A, k=1)]
    d_B = d_B[np.triu_indices_from(d_B, k=1)]
    d_across = distance_matrix(coords_A, coords_B).flatten()

    # save the distances to a file
    np.save(join(save_loc, f'{mouse_id}_{day_str}_within_group_distances_{label_A.replace(" ", "_")}_vs_{label_B.replace(" ", "_")}.npy'), d_A)
    np.save(join(save_loc, f'{mouse_id}_{day_str}_within_group_distances_{label_A.replace(" ", "_")}_vs_{label_B.replace(" ", "_")}.npy'), d_B)
    np.save(join(save_loc, f'{mouse_id}_{day_str}_within_group_distances_{label_A.replace(" ", "_")}_vs_{label_B.replace(" ", "_")}.npy'), d_across)

    plt.figure(figsize=(6, 4))
    plt.hist(d_A, bins=bins, alpha=0.55, label=f"{label_A} (within)", density=True, color=color_A)
    plt.hist(d_B, bins=bins, alpha=0.55, label=f"{label_B} (within)", density=True, color=color_B)
    plt.hist(d_across, bins=bins, alpha=0.45, label=f"Across {label_A} vs {label_B}", density=True, color="orange")
    plt.xlabel("Pairwise distance (px)")
    plt.ylabel("Density")
    plt.title("Within and across group distance distributions")
    plt.legend(frameon=True)
    plt.tight_layout()
    if save_loc is not None:
        Path(save_loc).mkdir(parents=True, exist_ok=True)
        save_fig_pdf(join(save_loc, f'{mouse_id}_{day_str}_within_group_histograms_{label_A.replace(" ", "_")}_vs_{label_B.replace(" ", "_")}'))
    plt.close()


def within_across_gap(coords_group1, coords_group2, weighting='pooled', subsample_reps=500, seed=None):
    n1, n2 = len(coords_group1), len(coords_group2)
    if n1 == 0 or n2 == 0:
        return np.nan

    def pooled_gap(g1, g2):
        coords = np.vstack([g1, g2])
        labels = np.array([0]*len(g1) + [1]*len(g2))
        D = distance_matrix(coords, coords)
        iu = np.triu_indices_from(D, k=1)
        same_mask = labels[iu[0]] == labels[iu[1]]
        within = D[iu][same_mask]
        across = D[iu][~same_mask]
        return np.nanmean(within) - np.nanmean(across)

    if weighting == 'pooled':
        return pooled_gap(coords_group1, coords_group2)

    if weighting == 'balanced':
        d1 = distance_matrix(coords_group1, coords_group1)
        d2 = distance_matrix(coords_group2, coords_group2)
        iu1 = np.triu_indices_from(d1, k=1)
        iu2 = np.triu_indices_from(d2, k=1)
        within_mean_bal = 0.5 * (np.nanmean(d1[iu1]) + np.nanmean(d2[iu2]))
        across_mean = np.nanmean(distance_matrix(coords_group1, coords_group2))
        return within_mean_bal - across_mean

    if weighting == 'subsample':
        reps = max(1, int(subsample_reps or 1))
        rng = np.random.default_rng(seed)
        m = min(n1, n2)
        if m < 2:
            return np.nan
        vals = []
        for _ in range(reps):
            idx1 = rng.choice(n1, size=m, replace=False)
            idx2 = rng.choice(n2, size=m, replace=False)
            vals.append(pooled_gap(coords_group1[idx1], coords_group2[idx2]))
        return float(np.nanmean(vals))

    raise ValueError(f"Unknown weighting: {weighting}")
def perm_test_within_across(coords_group1, coords_group2, n_perm=10000, seed=0, weighting='pooled', subsample_reps=500):
    rng = np.random.default_rng(seed)
    coords = np.vstack([coords_group1, coords_group2])
    n1 = len(coords_group1)
    observed_gap = within_across_gap(coords_group1, coords_group2,
                                     weighting=weighting,
                                     subsample_reps=max(1, subsample_reps),
                                     seed=seed)
    perm_gaps = []
    for _ in range(n_perm):
        perm_idx = rng.permutation(len(coords))
        g1 = coords[perm_idx[:n1]]
        g2 = coords[perm_idx[n1:]]
        perm_gaps.append(within_across_gap(g1, g2,
                                           weighting=weighting,
                                           subsample_reps=max(1, subsample_reps),
                                           seed=rng.integers(1 << 32)))
    perm_gaps = np.array(perm_gaps)
    pval = np.mean(np.abs(perm_gaps - perm_gaps.mean()) >= np.abs(observed_gap - perm_gaps.mean()))
    return observed_gap, perm_gaps, pval


def clustering_radius_permtest_summary(coords_neuro, coords_sig, radii=None, n_perm=1000, seed=0, weighting='pooled', subsample_reps=500):
    rng = np.random.default_rng(seed)
    coords = np.vstack([coords_neuro, coords_sig])
    labels = np.array([0]*len(coords_neuro) + [1]*len(coords_sig))
    D = distance_matrix(coords, coords)
    if radii is None:
        max_dist = np.nanmax(D)
        radii = np.linspace(5, max_dist, 30)
    def compute_curves(lbls):
        same_curve, diff_curve = [], []
        idx_g0 = np.where(lbls == 0)[0]
        idx_g1 = np.where(lbls == 1)[0]
        if weighting == 'pooled':
            for r in radii:
                same_counts, diff_counts = [], []
                for i in range(len(coords)):
                    neighbors = np.where((D[i] > 0) & (D[i] <= r))[0]
                    same_counts.append(np.sum(lbls[neighbors] == lbls[i]))
                    diff_counts.append(np.sum(lbls[neighbors] != lbls[i]))
                same_curve.append(np.mean(same_counts))
                diff_curve.append(np.mean(diff_counts))
            return np.array(same_curve), np.array(diff_curve)
        if weighting == 'balanced':
            for r in radii:
                same_counts_g0, diff_counts_g0 = [], []
                same_counts_g1, diff_counts_g1 = [], []
                for i in idx_g0:
                    neighbors = np.where((D[i] > 0) & (D[i] <= r))[0]
                    same_counts_g0.append(np.sum(lbls[neighbors] == 0))
                    diff_counts_g0.append(np.sum(lbls[neighbors] == 1))
                for i in idx_g1:
                    neighbors = np.where((D[i] > 0) & (D[i] <= r))[0]
                    same_counts_g1.append(np.sum(lbls[neighbors] == 1))
                    diff_counts_g1.append(np.sum(lbls[neighbors] == 0))
                same_curve.append(0.5 * (np.mean(same_counts_g0) + np.mean(same_counts_g1)))
                diff_curve.append(0.5 * (np.mean(diff_counts_g0) + np.mean(diff_counts_g1)))
            return np.array(same_curve), np.array(diff_curve)
        if weighting == 'subsample':
            reps = max(1, int(subsample_reps or 1))
            acc_same = np.zeros(len(radii), dtype=float)
            acc_diff = np.zeros(len(radii), dtype=float)
            n0, n1 = len(idx_g0), len(idx_g1)
            m = min(n0, n1)
            if m < 2:
                return np.full(len(radii), np.nan), np.full(len(radii), np.nan)
            for _ in range(reps):
                sel0 = rng.choice(idx_g0, size=m, replace=False)
                sel1 = rng.choice(idx_g1, size=m, replace=False)
                sel = np.concatenate([sel0, sel1])
                D_sel = D[np.ix_(sel, sel)]
                lbls_sel = lbls[sel]
                same_counts_r, diff_counts_r = [], []
                for r in radii:
                    s_counts, d_counts = [], []
                    for i in range(len(sel)):
                        neighbors = np.where((D_sel[i] > 0) & (D_sel[i] <= r))[0]
                        s_counts.append(np.sum(lbls_sel[neighbors] == lbls_sel[i]))
                        d_counts.append(np.sum(lbls_sel[neighbors] != lbls_sel[i]))
                    same_counts_r.append(np.mean(s_counts))
                    diff_counts_r.append(np.mean(d_counts))
                acc_same += np.array(same_counts_r)
                acc_diff += np.array(diff_counts_r)
            return acc_same / reps, acc_diff / reps
        raise ValueError(f"Unknown weighting: {weighting}")
    obs_same, obs_diff = compute_curves(labels)
    obs_auc = auc(radii, obs_same - obs_diff)
    perm_aucs = []
    for _ in range(n_perm):
        perm_labels = rng.permutation(labels)
        psame, pdiff = compute_curves(perm_labels)
        perm_aucs.append(auc(radii, psame - pdiff))
    perm_aucs = np.array(perm_aucs)
    pval = np.mean(np.abs(perm_aucs - perm_aucs.mean()) >= np.abs(obs_auc - perm_aucs.mean()))
    return {"radii": radii, "obs_same": obs_same, "obs_diff": obs_diff, "obs_auc": obs_auc, "perm_aucs": perm_aucs, "pval": pval}

def plot_clustering_across_mice(results_list, mice_ids, save_all_loc=None, pair="groupA_vs_groupB"):
    obs_aucs = [res["obs_auc"] for res in results_list]
    pvals = [res["pval"] for res in results_list]
    plt.figure(figsize=(7, max(3.5, 0.5*len(results_list)+1)))
    y = np.arange(len(results_list))
    plt.barh(y, obs_aucs, color=["tomato" if p < 0.05 else "gray" for p in pvals])
    plt.axvline(0, color="k", ls="--", lw=1)
    labels = [f"{m}  (p={p:.3f})" for m, p in zip(mice_ids, pvals)]
    plt.yticks(y, labels)
    plt.xlabel("Clustering AUC (same - diff)")
    plt.title("Clustering analysis across mice")
    plt.tight_layout()
    if save_all_loc:
        Path(save_all_loc).mkdir(parents=True, exist_ok=True)
        save_fig_pdf(join(save_all_loc,f"clustering_across_mice_{pair}"))
    plt.show()

def plot_forest(df, pooled_obs=None, pooled_null=None, title="Within minus across distance by mouse", save_all_loc=None, pair="groupA_vs_groupB"):
    df_plot = df.sort_values("obs_gap")
    y = np.arange(len(df_plot))
    plt.figure(figsize=(8, max(3.5, 0.6*len(df_plot)+1.5)))
    plt.hlines(y, df_plot["ci_low"], df_plot["ci_high"], color="gray", lw=2)
    plt.plot(df_plot["obs_gap"], y, "o", color="black")
    plt.axvline(0, color="k", ls="--", lw=1)
    if pooled_obs is not None and pooled_null is not None:
        pooled_mean = pooled_null.mean()
        pooled_ci = np.quantile(pooled_null, [0.025, 0.975])
        pooled_obs_mean = pooled_obs / len(df_plot)
        pooled_ci_mean = pooled_ci / len(df_plot)
        plt.axvspan(pooled_ci_mean[0], pooled_ci_mean[1], color="tomato", alpha=0.15, label="Pooled 95% null")
        plt.axvline(pooled_obs_mean, color="tomato", lw=2, label=f"Pooled observed = {pooled_obs_mean:.2f}")
    labels = [
        f"{m} {d}  (n={n1}|{n2})  p={p:.3f}"
        for m, d, n1, n2, p in zip(df_plot["mouse"], df_plot["day"], df_plot["n_A"], df_plot["n_B"], df_plot["pval"])
    ]
    plt.yticks(y, labels)
    plt.xlabel("Within minus across mean distance (px)")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    if save_all_loc:
        Path(save_all_loc).mkdir(parents=True, exist_ok=True)
        save_fig_pdf(join(save_all_loc, f'forest_plot_within_across_{pair}'), tight=True)
    plt.show()

# =========================
# Pair adapter to reuse two-group pipeline on any two of {light, reward, both}
# =========================
def ids_for_pair(pair, ids_light_only, ids_reward_only, ids_both):
    name_to_ids = {
        "light": np.asarray(ids_light_only, int),
        "reward": np.asarray(ids_reward_only, int),
        "both": np.asarray(ids_both, int),
    }
    a, b = pair
    return name_to_ids[a], name_to_ids[b]

def _coords_from_ids_safe(stat, ids):
    return _coords_from_ids(stat, ids) if len(ids) else np.empty((0,2))

def two_group_from_three(stat, pair, ids_light_only, ids_reward_only, ids_both):
    ids_A, ids_B = ids_for_pair(pair, ids_light_only, ids_reward_only, ids_both)
    coords_A = _coords_from_ids_safe(stat, ids_A)
    coords_B = _coords_from_ids_safe(stat, ids_B)
    return coords_A, coords_B

# =========================
# Single-mouse run with both 3-group and all 2-group analyses
# =========================
def run_all_pairs_single_mouse(stat, ids_light_only, ids_reward_only, ids_both, save_loc, mouse, day,
                               radii=np.arange(10, 200, 20), n_perm_gap=1000, n_perm_radius=500, seed_base=123,
                               weighting='subsample', subsample_reps=500):
    PAIRS = [("light","reward"), ("light","both"), ("reward","both")]
    for i, pair in enumerate(PAIRS):
        coords_A, coords_B = two_group_from_three(stat, pair, ids_light_only, ids_reward_only, ids_both)
        if len(coords_A) == 0 or len(coords_B) == 0:
            print(f"\nTwo-group analysis skipped for {pair} because one group is empty.")
            continue

        obs_gap, perm_gaps_2, p_gap_2 = perm_test_within_across(
            coords_A, coords_B, n_perm=n_perm_gap, seed=seed_base+i, weighting=weighting, subsample_reps=subsample_reps
        )
        nameA, nameB = (GROUP_LABELS[pair[0]], GROUP_LABELS[pair[1]])
        colorA, colorB = (GROUP_COLORS[pair[0]], GROUP_COLORS[pair[1]])
        labelA = f"{nameA} (n={len(coords_A)})"
        labelB = f"{nameB} (n={len(coords_B)})"

        plot_perm_hist(
            perm_vals=perm_gaps_2,
            observed=obs_gap,
            p_two_sided=p_gap_2,
            xlabel="Within - Across mean distance (px)",
            title=f"Permutation Test: {nameA} vs {nameB}",
            save_loc=save_loc, mouse_id=mouse, day_str=day,
            fname_stub=f"{pair[0]}_vs_{pair[1]}_gap_perm_hist"
        )

        plot_within_group_histograms(
            coords_A, coords_B,
            save_loc=save_loc, mouse_id=mouse, day_str=day,
            label_A=labelA, label_B=labelB,
            color_A=colorA, color_B=colorB
        )

        res_clust = clustering_radius_permtest_summary(
            coords_A, coords_B, radii=radii, n_perm=n_perm_radius, seed=999 + i,
            weighting=weighting, subsample_reps=subsample_reps
        )
        print(f"  Clustering AUC = {res_clust['obs_auc']:.3f}, p = {res_clust['pval']:.4f}")

def main_single_mouse():

    RADII_FIXED = np.array([8, 12, 16, 24, 32, 48, 64, 96, 128, 160, 180], dtype=np.float64)
    conditions_played = np.load(join(ROOT_PATH, 'conditions_played.npy'))
    conditions_played = conditions_played[n_priming_trials:]
    _ = build_freq_indices(conditions_played, frequency_map, target_freqs)

    event_data_by_freq, results_by_freq, results_vs_hit_by_freq, \
        res_reward_vs_lightHit, res_lick_on_vs_lightHit, auROC_results_hit = load_event_data(load_loc, target_freqs)

    n_neurons = len(event_data_by_freq[target_freqs[0]]['aurocA'])
    freq_matrix = build_freq_matrix(n_neurons, target_freqs, results_by_freq, results_vs_hit_by_freq)

    light_inc, light_dec = find_light_significant_neurons(freq_matrix)
    reward_all, reward_inc, reward_dec = find_reward_significant_neurons(
        res_reward_vs_lightHit, res_lick_on_vs_lightHit, auROC_results_hit
    )

    light_set = set(light_inc) | set(light_dec)
    reward_set = set(reward_inc) | set(reward_dec)
    both_set = light_set & reward_set
    light_only_set = light_set - reward_set
    reward_only_set = reward_set - light_set

    ops, stat, iscell, bg = load_suite2p(s2p_dir)
    n_stat = len(stat)
    n_filt = int(np.sum(iscell[:, 0] == 1))
    print(f"Loaded Suite2p data: {n_stat} total ROIs, {n_filt} filtered ROIs.")

    if n_neurons == n_filt:
        print("Detected filtered indices. Mapping to Suite2p ROI indices.")
        cell_idx = np.flatnonzero(iscell[:, 0] == 1)
        map_fun = lambda s: cell_idx[np.fromiter(s, dtype=int)]
    elif n_neurons == n_stat:
        print("Detected raw Suite2p ROI indices.")
        map_fun = lambda s: np.fromiter(s, dtype=int)
    else:
        print(f"Warning: n_neurons={n_neurons}, n_filtered={n_filt}, n_stat={n_stat}. Using heuristic mapping.")
        cell_idx = np.flatnonzero(iscell[:, 0] == 1)
        if len(light_set | reward_set) and max(light_set | reward_set) < n_filt:
            map_fun = lambda s: cell_idx[np.fromiter(s, dtype=int)]
        else:
            map_fun = lambda s: np.fromiter(s, dtype=int)

    ids_light_only = map_fun(light_only_set)
    ids_reward_only = map_fun(reward_only_set)
    ids_both       = map_fun(both_set)

    print("\nSummary:")
    print(f"  Light increase: {len(light_inc)}")
    print(f"  Light decrease: {len(light_dec)}")
    print(f"  Reward increase: {len(reward_inc)}")
    print(f"  Reward decrease: {len(reward_dec)}")
    print(f"  Light only: {len(ids_light_only)}")
    print(f"  Reward only: {len(ids_reward_only)}")
    print(f"  Both: {len(ids_both)}")

    plot_three_class_outlines(
        ops, stat, bg,
        ids_light_only=ids_light_only,
        ids_reward_only=ids_reward_only,
        ids_both=ids_both,
        colors=GROUP_COLORS,
        bg_enhance=BG_ENHANCE,
        alpha=ALPHA,
        outline_lw=OUTLINE_LW,
        save_path=SAVE_FIG
    )

    coords_all, labels, group_map = three_group_coords(stat, ids_light_only, ids_reward_only, ids_both)

    # 3-group analyses
    plot_within_across_hist_3groups(coords_all, labels, save_loc=save_loc, mouse=mouse, day=day)
    gaps_obs, perm_gaps, p_gap = perm_test_within_across_3groups(coords_all, labels, n_perm=1000, seed=42)
    print("Within minus across gap per group [light, reward, both]:", gaps_obs)
    print("p-values:", p_gap)
    plot_gap_summary_3groups(gaps_obs, perm_gaps, p_gap, save_loc=save_loc, mouse=mouse, day=day)

    nn_obs, nn_perm, nn_p = perm_test_nn_3groups(coords_all, labels, n_perm=1000, seed=7)
    print("Mean NN matrix (rows=source, cols=target):\n", nn_obs)
    plot_nn_heatmap(nn_obs, p_nn=nn_p, save_loc=save_loc, mouse=mouse, day=day)

    curves, same_perm, other_perm, auc_obs, auc_perm, p_auc = radial_clustering_perm_3groups(
        coords_all, labels, radii=RADII_FIXED, n_perm=500, seed=11
    )
    print("Radial clustering AUC (same - other) per group:", auc_obs)
    print("p-values:", p_auc)
    plot_radial_curves_3groups(curves, save_loc=save_loc, mouse=mouse, day=day)
    plot_radial_auc_summary(auc_obs, auc_perm, p_auc, save_loc=save_loc, mouse=mouse, day=day)

    # All three two-group analyses
    run_all_pairs_single_mouse(
        stat, ids_light_only, ids_reward_only, ids_both,
        save_loc=save_loc, mouse=mouse, day=day,
        radii=RADII_FIXED,
        n_perm_gap=500, n_perm_radius=500, seed_base=123, weighting='subsample', subsample_reps=500
    )

# =========================
# Across-mice orchestration that runs all pairs
# =========================
def compute_sets_for_mouse(mouse_id, day_str):
    if mouse_id in ["OI116", "OI124"]:
        root_path = fr'Z:\2p_data\{mouse_id}\{day_str}\ML\data2'
    else:
        root_path = fr'Z:\2p_data\{mouse_id}\{day_str}\ML\data'
    load_path = fr'Z:\2p_data\{mouse_id}\{day_str}\ML\data\processed_data\auROC_freq_psycho_new'
    s2p_path  = fr'Z:\2p_data\{mouse_id}\{day_str}\{mouse_id}_run1\suite2p\plane0'
    save_path = fr'Z:\2p_data\{mouse_id}\{day_str}\ML\data\processed_data\topography_light_reward'

    os.makedirs(save_path, exist_ok=True)

    conditions_played = np.load(join(root_path, 'conditions_played.npy'))
    conditions_played = conditions_played[n_priming_trials:]
    _ = build_freq_indices(conditions_played, frequency_map, target_freqs)

    event_data_by_freq, results_by_freq, results_vs_hit_by_freq, \
        res_reward_vs_lightHit, res_lick_on_vs_lightHit, auROC_results_hit = load_event_data(load_path, target_freqs)

    n_neurons = len(event_data_by_freq[target_freqs[0]]['aurocA'])
    freq_matrix = build_freq_matrix(n_neurons, target_freqs, results_by_freq, results_vs_hit_by_freq)

    light_inc, light_dec = find_light_significant_neurons(freq_matrix)
    reward_all, reward_inc, reward_dec = find_reward_significant_neurons(
        res_reward_vs_lightHit, res_lick_on_vs_lightHit, auROC_results_hit
    )

    light_set = set(light_inc) | set(light_dec)
    reward_set = set(reward_inc) | set(reward_dec)
    both_set = light_set & reward_set
    light_only_set = light_set - reward_set
    reward_only_set = reward_set - light_set

    ops, stat, iscell, bg = load_suite2p(s2p_path)
    n_stat = len(stat)
    n_filt = int(np.sum(iscell[:, 0] == 1))
    if n_neurons == n_filt:
        cell_idx = np.flatnonzero(iscell[:, 0] == 1)
        map_fun = lambda s: cell_idx[np.fromiter(s, dtype=int)]
    elif n_neurons == n_stat:
        map_fun = lambda s: np.fromiter(s, dtype=int)
    else:
        cell_idx = np.flatnonzero(iscell[:, 0] == 1)
        if len(light_set | reward_set) and max(light_set | reward_set) < n_filt:
            map_fun = lambda s: cell_idx[np.fromiter(s, dtype=int)]
        else:
            map_fun = lambda s: np.fromiter(s, dtype=int)

    ids_light_only = map_fun(light_only_set)
    ids_reward_only = map_fun(reward_only_set)
    ids_both       = map_fun(both_set)

    return {
        "mouse": mouse_id, "day": day_str,
        "stat": stat, "save_loc": save_path, "s2p_dir": s2p_path,
        "ids_light_only": ids_light_only,
        "ids_reward_only": ids_reward_only,
        "ids_both": ids_both
    }

def run_all_pairs_across_mice(mice_days, radii=RADII_FIXED,
                              n_perm_gap=500, n_perm_radius=500, seed_base=1234,
                              weighting='subsample', subsample_reps=500):
    PAIRS = [("light","reward"), ("light","both"), ("reward","both")]

    results_by_pair = {}
    for pair in PAIRS:
        all_gap_results = []
        all_clust_results = []
        mice_ids = []

        for i, (m, d) in enumerate(mice_days):
            print(f"\nProcessing mouse {m} on day {d} for pair {pair}")
            pack = compute_sets_for_mouse(m, d)
            stat = pack["stat"]
            ids_light_only = pack["ids_light_only"]
            ids_reward_only = pack["ids_reward_only"]
            ids_both = pack["ids_both"]
            save_loc = pack["save_loc"]

            coords_A, coords_B = two_group_from_three(stat, pair, ids_light_only, ids_reward_only, ids_both)
            if len(coords_A) == 0 or len(coords_B) == 0:
                print("One of the groups is empty. Skipping this mouse.")
                continue

            obs_gap, perm_gaps, p_gap = perm_test_within_across(
                coords_A, coords_B, n_perm=n_perm_gap, seed=seed_base+i, weighting=weighting, subsample_reps=subsample_reps
            )
            null_mean = perm_gaps.mean()
            null_sd   = perm_gaps.std(ddof=1)
            ci_low, ci_high = np.quantile(perm_gaps, [0.025, 0.975])

            all_gap_results.append({
                "mouse": m, "day": d,
                "n_A": len(coords_A), "n_B": len(coords_B),
                "obs_gap": obs_gap, "pval": p_gap,
                "null_mean": null_mean, "null_sd": null_sd,
                "ci_low": ci_low, "ci_high": ci_high,
                "perm_gaps": perm_gaps
            })

            # Per-mouse figures
            nameA, nameB = (GROUP_LABELS[pair[0]], GROUP_LABELS[pair[1]])
            colorA, colorB = (GROUP_COLORS[pair[0]], GROUP_COLORS[pair[1]])
            labelA = f"{nameA} (n={len(coords_A)})"
            labelB = f"{nameB} (n={len(coords_B)})"

            plot_perm_hist(perm_gaps, obs_gap, p_gap,
                           xlabel="Within - Across mean distance (px)",
                           title=f"Permutation Test: {nameA} vs {nameB}",
                           save_loc=save_loc, mouse_id=m, day_str=d,
                           fname_stub=f"{pair[0]}_vs_{pair[1]}_gap_perm_hist")

            plot_within_group_histograms(
                coords_A, coords_B,
                save_loc=save_loc, mouse_id=m, day_str=d,
                label_A=labelA, label_B=labelB,
                color_A=colorA, color_B=colorB
            )
            res_clust = clustering_radius_permtest_summary(
                coords_A, coords_B, radii=radii, n_perm=n_perm_radius, seed=999+i, weighting=weighting, subsample_reps=subsample_reps
            )
            all_clust_results.append(res_clust)
            mice_ids.append(f"{m}_{d}")

        if not all_gap_results:
            print(f"No mice had both groups non-empty for the pair {pair}.")
            continue

        df = pd.DataFrame([{k: v for k, v in dct.items() if k != "perm_gaps"} for dct in all_gap_results])
        print(f"\nPer mouse summary for pair {pair} (within - across):")
        print(df[["mouse", "day", "n_A", "n_B", "obs_gap", "pval", "ci_low", "ci_high"]])

        obs_pooled = df["obs_gap"].sum()
        pooled_null = None
        for dct in all_gap_results:
            pooled_null = dct["perm_gaps"].copy() if pooled_null is None else pooled_null + dct["perm_gaps"]
        pval_pooled_two_sided = np.mean(np.abs(pooled_null - pooled_null.mean()) >= np.abs(obs_pooled - pooled_null.mean()))
        pooled_ci_low, pooled_ci_high = np.quantile(pooled_null, [0.025, 0.975])

        print(f"\nPooled observed gap sum = {obs_pooled:.2f}  for pair {pair}")
        print(f"Pooled two sided p = {pval_pooled_two_sided:.4f}")
        print(f"Pooled 95% null interval = [{pooled_ci_low:.2f}, {pooled_ci_high:.2f}]")

        title = f"Topography: within vs across per mouse ({GROUP_LABELS[pair[0]]} vs {GROUP_LABELS[pair[1]]})"
        plot_forest(
            df.rename(columns={"n_A":"n_A", "n_B":"n_B"}),
            pooled_obs=obs_pooled,
            pooled_null=pooled_null,
            title=title,
            save_all_loc=save_loc_all,
            pair=f"{pair[0]}_vs_{pair[1]}"
        )

        plot_clustering_across_mice(all_clust_results, mice_ids, save_all_loc=save_loc_all, pair=f"{pair[0]}_vs_{pair[1]}")
        results_by_pair[pair] = {"df": df, "all_gap_results": all_gap_results, "all_clust_results": all_clust_results, "mice_ids": mice_ids}

    return results_by_pair

if __name__ == "__main__":
    # 1) Single mouse run with both 3-group and all 2-group outputs
    # main_single_mouse()

    # 2) Across-mice, all pairs
    mice_days = [
        ('OI116', '240529'),
        ('OI124', '240610'),
        ('OI151', '240827'),
        ('OI152', '241003'),
        ('OI208', '250227'),
    ]
    run_all_pairs_across_mice(mice_days)
