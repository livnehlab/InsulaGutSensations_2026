# === Imports ===
from pathlib import Path
from os.path import join
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.spatial import distance_matrix, ConvexHull
from sklearn.metrics import auc
# ---------- Illustrator-friendly PDF saving ----------
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt

def setup_matplotlib_for_illustrator(font_family="Arial"):
    """
    Configure matplotlib so PDF exports stay fully editable in Illustrator.
    """
    mpl.rcParams.update({
        "font.family": font_family,
        "pdf.fonttype": 42,   # keep text as editable TrueType
        "ps.fonttype": 42,    # same for EPS (if you ever use it)
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
    """
    Save current figure (or given fig) as a single .pdf ready for Illustrator.
    - path_base: path without extension (e.g. 'results/plot1')
    """
    if fig is None:
        fig = plt.gcf()

    # selectively rasterize very large collections if needed
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

    save_kwargs = {
        "dpi": dpi,
        "facecolor": facecolor,
        "transparent": transparent,
    }
    if tight:
        save_kwargs.update(bbox_inches="tight", pad_inches=pad_inches)

    path = f"{path_base}.pdf"
    fig.savefig(path, **save_kwargs)
    return path

setup_matplotlib_for_illustrator("Arial")

# === IO and basic helpers ===
def _load_suite2p(s2p_plane):
    s2p_plane = Path(s2p_plane)
    ops = np.load(s2p_plane / 'ops.npy', allow_pickle=True).item()
    stat = np.load(s2p_plane / 'stat.npy', allow_pickle=True)
    iscell = np.load(s2p_plane / 'iscell_final.npy')
    bg = ops.get('meanImgE')
    if bg is None:
        raise ValueError("meanImg not found in ops.npy")
    return ops, stat, iscell, bg

def _filtered_indices(iscell):
    return np.flatnonzero(iscell[:, 0] == 1)


# === ROI outlines and coordinates ===
def _outline_xy_from_pixels(xpix, ypix, Ly, Lx):
    try:
        from skimage.draw import polygon
        from skimage.measure import find_contours
        mask = np.zeros((Ly, Lx), dtype=bool)
        rr, cc = polygon(ypix, xpix, shape=mask.shape)
        mask[rr, cc] = True
        contours = find_contours(mask.astype(float), level=0.5)
        if not contours:
            raise RuntimeError
        cont = max(contours, key=lambda c: c.shape[0])
        Y, X = cont[:, 0], cont[:, 1]
        return X, Y
    except Exception:
        pts = np.column_stack([xpix, ypix])
        if pts.shape[0] < 3:
            return np.array([xpix.mean()]), np.array([ypix.mean()])
        hull = ConvexHull(pts)
        verts = hull.vertices
        X = pts[verts, 0]
        Y = pts[verts, 1]
        X = np.r_[X, X[0]]
        Y = np.r_[Y, Y[0]]
        return X, Y

def _enhance_image(img, method='rescale'):
    from skimage import exposure
    if method == 'rescale':
        p2, p98 = np.percentile(img, (2, 98))
        return exposure.rescale_intensity(img, in_range=(p2, p98))
    elif method == 'clahe':
        return exposure.equalize_adapthist(img / max(1e-9, img.max()), clip_limit=0.03)
    else:
        return img

def plot_roi_outlines_with_enhancement(
    s2p_plane,
    neurometric_idx,
    sigmoid_idx,
    indices_are_filtered=True,
    lw=0.8,
    color_neuro="cyan",
    color_sig="magenta",
    enhance_method='rescale',
    save_loc=None,
    mouse_id=None,
    day_str=None
):
    ops, stat, iscell, bg = _load_suite2p(s2p_plane)
    Ly, Lx = ops["Ly"], ops["Lx"]
    cell_idx = _filtered_indices(iscell)

    if indices_are_filtered:
        neu_idx_s2p = cell_idx[np.asarray(neurometric_idx, int)]
        sig_idx_s2p = cell_idx[np.asarray(sigmoid_idx, int)]
    else:
        neu_idx_s2p = np.asarray(neurometric_idx, int)
        sig_idx_s2p = np.asarray(sigmoid_idx, int)

    bg_enh = _enhance_image(bg, enhance_method)

    plt.figure(figsize=(8, 8))
    plt.imshow(bg_enh, cmap="gray")
    plt.axis("off")

    for i in neu_idx_s2p:
        s = stat[i]
        X, Y = _outline_xy_from_pixels(s["xpix"], s["ypix"], Ly, Lx)
        plt.plot(X, Y, color_neuro, lw=lw)
    for i in sig_idx_s2p:
        s = stat[i]
        X, Y = _outline_xy_from_pixels(s["xpix"], s["ypix"], Ly, Lx)
        plt.plot(X, Y, color_sig, lw=lw)

    ln1, = plt.plot([], [], color_neuro, lw=lw, label="Neurometric")
    ln2, = plt.plot([], [], color_sig, lw=lw, label="Sigmoid")
    plt.legend(handles=[ln1, ln2], loc="lower right", frameon=True)
    plt.title("ROI outlines")
    if save_loc is not None:
        Path(save_loc).mkdir(parents=True, exist_ok=True)
        save_fig_pdf(join(save_loc, f'{mouse_id}_{day_str}_roi_outlines'))
    plt.show()

    # return medoid coordinates in [y, x] order (consistent everywhere)
    coords_neuro = np.array([[np.median(stat[i]['ypix']), np.median(stat[i]['xpix'])] for i in neu_idx_s2p])
    coords_sig   = np.array([[np.median(stat[i]['ypix']), np.median(stat[i]['xpix'])] for i in sig_idx_s2p])
    return coords_neuro, coords_sig


# === Generic plotting utilities ===
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
    plt.show()

def plot_within_group_histograms(coords_neuro, coords_sig, bins=40, save_loc=None, mouse_id=None, day_str=None):
    d_neuro = distance_matrix(coords_neuro, coords_neuro)
    d_sig   = distance_matrix(coords_sig, coords_sig)
    d_neuro = d_neuro[np.triu_indices_from(d_neuro, k=1)]
    d_sig   = d_sig[np.triu_indices_from(d_sig, k=1)]
    # add across distance between the two groups
    d_across = distance_matrix(coords_neuro, coords_sig)
    # Flatten the across-group distance matrix (no triu needed for rectangular matrix)
    d_across = d_across.flatten()

    plt.figure(figsize=(6, 4))
    plt.hist(d_neuro, bins=bins, alpha=0.5, label="Neurometric (within)", density=True, color="cyan")
    plt.hist(d_sig,   bins=bins, alpha=0.5, label="Sigmoid (within)",     density=True, color="magenta")
    plt.hist(d_across, bins=bins, alpha=0.5, label="Across groups", density=True, color="orange")
    plt.xlabel("Pairwise distance (px)")
    plt.ylabel("Density")
    plt.title("Within and across group distance distributions")
    plt.legend(frameon=True)
    plt.tight_layout()
    if save_loc is not None:
        Path(save_loc).mkdir(parents=True, exist_ok=True)
        save_fig_pdf(join(save_loc, f'{mouse_id}_{day_str}_within_group_histograms'))
    plt.show()


# === Core distance stats ===
def within_across_gap(coords_group1, coords_group2, weighting='pooled', subsample_reps=0, seed=None):
    """
    Compute (within mean distance) - (across mean distance) between two coordinate groups.

    weighting:
      - 'pooled': pooled mean of all within-pairs across both groups (default)
      - 'balanced': average of per-group within means to avoid over-weighting the larger group
      - 'subsample': repeatedly subsample min(n1, n2) cells from each group and average the pooled gap

    subsample_reps: number of subsamples when weighting == 'subsample'.
    seed: RNG seed for reproducibility when subsampling.
    """
    n1, n2 = len(coords_group1), len(coords_group2)
    if n1 == 0 or n2 == 0:
        return np.nan

    def pooled_gap(g1, g2):
        coords = np.vstack([g1, g2])
        labels = np.array([0]*len(g1) + [1]*len(g2))
        distmat = distance_matrix(coords, coords)
        np.fill_diagonal(distmat, np.nan)
        within = distmat[labels[:, None] == labels[None, :]]
        across = distmat[labels[:, None] != labels[None, :]]
        return np.nanmean(within) - np.nanmean(across)

    if weighting == 'pooled':
        return pooled_gap(coords_group1, coords_group2)

    if weighting == 'balanced':
        # Compute within means per group, then average them to avoid bias from unequal sizes
        d1 = distance_matrix(coords_group1, coords_group1)
        d2 = distance_matrix(coords_group2, coords_group2)
        np.fill_diagonal(d1, np.nan)
        np.fill_diagonal(d2, np.nan)
        within_mean_balanced = np.nanmean(d1) / 1.0 + np.nanmean(d2) / 1.0
        within_mean_balanced *= 0.5
        across_mean = np.nanmean(distance_matrix(coords_group1, coords_group2))
        return within_mean_balanced - across_mean

    if weighting == 'subsample':
        if subsample_reps is None or subsample_reps <= 0:
            subsample_reps = 1
        rng = np.random.default_rng(seed)
        m = min(n1, n2)
        if m < 2:
            return np.nan
        vals = []
        for _ in range(subsample_reps):
            idx1 = rng.choice(n1, size=m, replace=False)
            idx2 = rng.choice(n2, size=m, replace=False)
            vals.append(pooled_gap(coords_group1[idx1], coords_group2[idx2]))
        return float(np.nanmean(vals))

    raise ValueError(f"Unknown weighting mode: {weighting}")

def perm_test_within_across(coords_group1, coords_group2, n_perm=500, seed=0, weighting='pooled', subsample_reps=0):
    rng = np.random.default_rng(seed)
    coords = np.vstack([coords_group1, coords_group2])
    n1 = len(coords_group1)

    observed_gap = within_across_gap(coords_group1, coords_group2, weighting=weighting, subsample_reps=subsample_reps, seed=seed)

    perm_gaps = []
    for _ in range(n_perm):
        perm_idx = rng.permutation(len(coords))
        g1 = coords[perm_idx[:n1]]
        g2 = coords[perm_idx[n1:]]
        perm_gaps.append(within_across_gap(g1, g2, weighting=weighting, subsample_reps=subsample_reps, seed=rng.integers(1<<32)))
    perm_gaps = np.array(perm_gaps)

    # two sided
    pval = np.mean(np.abs(perm_gaps - perm_gaps.mean()) >= np.abs(observed_gap - perm_gaps.mean()))
    return observed_gap, perm_gaps, pval

# === Radial clustering with summary AUC ===
def clustering_radius_permtest_summary(coords_neuro, coords_sig, radii=None, n_perm=500, seed=0, weighting='pooled', subsample_reps=0, keep_perm_curves=False):
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
            reps = max(1, int(subsample_reps))
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
                # work on the submatrix to restrict neighbors to the sampled set
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

        raise ValueError(f"Unknown weighting mode: {weighting}")

    # observed
    obs_same, obs_diff = compute_curves(labels)
    obs_auc = auc(radii, obs_same - obs_diff)

    # permutations
    perm_aucs = []
    for _ in range(n_perm):
        perm_labels = rng.permutation(labels)
        psame, pdiff = compute_curves(perm_labels)
        perm_aucs.append(auc(radii, psame - pdiff))
    perm_aucs = np.array(perm_aucs)

    pval = np.mean(np.abs(perm_aucs - perm_aucs.mean()) >= np.abs(obs_auc - perm_aucs.mean()))

    return {
        "radii": radii,
        "obs_same": obs_same,
        "obs_diff": obs_diff,
        "obs_auc": obs_auc,
        "perm_aucs": perm_aucs,
        "pval": pval,
        "weighting": weighting
    }

def plot_clustering_across_mice(results_list, mice_ids, save_all_loc=None):
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
    plt.savefig(join(save_all_loc,"clustering_across_mice.pdf"), bbox_inches="tight")
    plt.show()


# === Forest plot for within vs across gap ===
def plot_forest(df, pooled_obs=None, pooled_null=None, title="Within minus across distance by mouse", save_all_loc=None):
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
        for m, d, n1, n2, p in zip(df_plot["mouse"], df_plot["day"], df_plot["n_neuro"], df_plot["n_sig"], df_plot["pval"])
    ]
    plt.yticks(y, labels)
    plt.xlabel("Within minus across mean distance (px)")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    if save_all_loc is not None:
        Path(save_all_loc).mkdir(parents=True, exist_ok=True)
        save_fig_pdf(join(save_all_loc, 'forest_plot_within_across'), tight=True)
    plt.show()


# === Orchestration for a list of mice ===
def run_all(
    mice_days,
    n_perm_per_mouse_gap=500,
    n_perm_nn=1000,
    n_perm_radius=500,
    radii=None,
    gap_weighting='subsample',
    gap_subsample_reps=500,
    clust_weighting='subsample',
    clust_subsample_reps=500
):
    all_gap_results = []
    all_clust_results = []
    mouse_ids = []

    # per mouse processing
    for i, (mouse_id, day_str) in enumerate(mice_days):
        print(f"\nProcessing mouse {mouse_id} on day {day_str}")
        data_dir = fr'Z:\2p_data\{mouse_id}\{day_str}\ML\data\processed_data\auroc_neurometric_zscore_no_sub_v3'
        s2p_dir = fr'Z:\2p_data\{mouse_id}\{day_str}\{mouse_id}_run1\suite2p\plane0'
        save_loc = fr'Z:\2p_data\{mouse_id}\{day_str}\ML\data\processed_data\topographic_analysis_sigmoid_vs_gaussian_v3'
        save_loc_all = fr'Z:\Stavsht\Omer_OXTR\topographic_analysis_sigmoid_vs_gaussian_v3'

        neurometric_cells = np.load(join(data_dir, 'significant_neurons.npy'))
        sigmoid_cells     = np.load(join(data_dir, 'significant_neurons_gauss.npy'))

        coords_neuro, coords_sig = plot_roi_outlines_with_enhancement(
            s2p_dir, neurometric_cells, sigmoid_cells,
            save_loc=save_loc, mouse_id=mouse_id, day_str=day_str
        )

        # within vs across permutation test
        obs_gap, perm_gaps, p_gap = perm_test_within_across(
            coords_neuro,
            coords_sig,
            n_perm=n_perm_per_mouse_gap,
            seed=1234 + i,
            weighting=gap_weighting,
            subsample_reps=gap_subsample_reps
        )
        null_mean = perm_gaps.mean()
        null_sd   = perm_gaps.std(ddof=1)
        ci_low, ci_high = np.quantile(perm_gaps, [0.025, 0.975])
        z_effect = (obs_gap - null_mean) / null_sd if null_sd > 0 else np.nan

        all_gap_results.append({
            "mouse": mouse_id,
            "day": day_str,
            "n_neuro": len(coords_neuro),
            "n_sig": len(coords_sig),
            "obs_gap": obs_gap,
            "pval": p_gap,
            "null_mean": null_mean,
            "null_sd": null_sd,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "perm_gaps": perm_gaps
        })

        # histogram of permutation for gap
        plot_perm_hist(
            perm_vals=perm_gaps,
            observed=obs_gap,
            p_two_sided=p_gap,
            xlabel="Within - Across mean distance (px)",
            title=f"Permutation Test: Within - Across ({gap_weighting})",
            save_loc=save_loc, mouse_id=mouse_id, day_str=day_str,
            fname_stub="gap_perm_hist"
        )

        # within group histograms
        plot_within_group_histograms(coords_neuro, coords_sig, save_loc=save_loc, mouse_id=mouse_id, day_str=day_str)

        # radial clustering summary AUC
        res_clust = clustering_radius_permtest_summary(
            coords_neuro,
            coords_sig,
            radii=radii,
            n_perm=n_perm_radius,
            seed=999 + i,
            weighting=clust_weighting,
            subsample_reps=clust_subsample_reps,
            keep_perm_curves=False
        )
        all_clust_results.append(res_clust)
        mouse_ids.append(f"{mouse_id}_{day_str}")

    # dataframe for gap results
    df = pd.DataFrame([{k: v for k, v in d.items() if k != "perm_gaps"} for d in all_gap_results])
    print("\nPer mouse summary (within - across):")
    print(df[["mouse", "day", "n_neuro", "n_sig", "obs_gap", "pval", "ci_low", "ci_high"]])

    # pooled gap across mice
    obs_pooled = df["obs_gap"].sum()
    pooled_null = None
    for d in all_gap_results:
        pooled_null = d["perm_gaps"].copy() if pooled_null is None else pooled_null + d["perm_gaps"]
    pval_pooled_two_sided = np.mean(np.abs(pooled_null - pooled_null.mean()) >= np.abs(obs_pooled - pooled_null.mean()))
    pooled_ci_low, pooled_ci_high = np.quantile(pooled_null, [0.025, 0.975])

    print(f"\nPooled observed gap sum = {obs_pooled:.2f}")
    print(f"Pooled two sided p = {pval_pooled_two_sided:.4f}")
    print(f"Pooled 95% null interval = [{pooled_ci_low:.2f}, {pooled_ci_high:.2f}]")

    # forest plot
    plot_forest(df, pooled_obs=obs_pooled, pooled_null=pooled_null, title="Topography: within vs across distance per mouse", save_all_loc=save_loc_all)

    # pooled clustering across mice
    obs_auc_pooled = np.sum([res["obs_auc"] for res in all_clust_results])
    pooled_clust_null = None
    for res in all_clust_results:
        pooled_clust_null = res["perm_aucs"].copy() if pooled_clust_null is None else pooled_clust_null + res["perm_aucs"]
    pval_clust_pooled_two_sided = np.mean(np.abs(pooled_clust_null - pooled_clust_null.mean()) >= np.abs(obs_auc_pooled - pooled_clust_null.mean()))
    pooled_clust_ci_low, pooled_clust_ci_high = np.quantile(pooled_clust_null, [0.025, 0.975])

    print(f"\nPooled clustering AUC sum = {obs_auc_pooled:.2f}")
    print(f"Pooled clustering two sided p = {pval_clust_pooled_two_sided:.4f}")
    print(f"Pooled clustering 95% null interval = [{pooled_clust_ci_low:.2f}, {pooled_clust_ci_high:.2f}]")

    # clustering across mice plot
    plot_clustering_across_mice(all_clust_results, mouse_ids, save_all_loc=save_loc_all)

    return df, all_gap_results, all_clust_results, mouse_ids


# === Example run ===
if __name__ == "__main__":
    mice_days = [
        ('OI116', '240529'),
        ('OI124', '240610'),
        ('OI151', '240827'),
        ('OI152', '241003'),
        ('OI208', '250227'),
    ]
    radii_fixed = np.array([8, 12, 16, 24, 32, 48, 64, 96, 128, 160, 180], dtype=np.float64)
    run_all(mice_days, gap_weighting='subsample', gap_subsample_reps=500, clust_weighting='subsample', clust_subsample_reps=500, radii=radii_fixed)
