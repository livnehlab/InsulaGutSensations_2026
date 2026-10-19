import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from pathlib import Path
import os
from os.path import join
import matplotlib as mpl
import pandas as pd
from scipy.spatial import distance_matrix
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
import numpy as np
import sklearn.metrics

# ---------------------------------------------------------------------
# Illustrator-friendly saving
# ---------------------------------------------------------------------
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
                 pad_inches=0.02):
    if fig is None:
        fig = plt.gcf()
    outdir = Path(path_base).parent
    outdir.mkdir(parents=True, exist_ok=True)
    kwargs = {"dpi": dpi, "facecolor": facecolor, "transparent": transparent}
    if tight:
        kwargs.update(bbox_inches="tight", pad_inches=pad_inches)
    fig.savefig(f"{path_base}.pdf", **kwargs)

setup_matplotlib_for_illustrator("Arial")

# ---------------------------------------------------------------------
# Numpy pickle compatibility
# ---------------------------------------------------------------------
class NumpyCoreUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core")
        return super().find_class(module, name)

# ---------------------------------------------------------------------
# Suite2p loaders and helpers
# ---------------------------------------------------------------------
def _load_suite2p(s2p_plane):
    s2p_plane = Path(s2p_plane)
    ops = np.load(s2p_plane / 'ops.npy', allow_pickle=True).item()
    stat = np.load(s2p_plane / 'stat.npy', allow_pickle=True)
    iscell = np.load(s2p_plane / 'iscell_final.npy')
    bg = ops.get('meanImg', ops.get('meanImgE'))
    if bg is None:
        raise ValueError("meanImg not found in ops.npy")
    return ops, stat, iscell, bg

def _filtered_indices(iscell):
    return np.flatnonzero(iscell[:, 0] == 1)

def _contour_from_pixels(xpix, ypix, Ly, Lx):
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
        return np.c_[cont[:, 1], cont[:, 0]]
    except Exception:
        try:
            from scipy.spatial import ConvexHull
            pts = np.column_stack([xpix, ypix])
            if pts.shape[0] < 3:
                cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
                return np.array([[cx-0.5, cy-0.5], [cx+0.5, cy-0.5], [cx, cy+0.5]])
            hull = ConvexHull(pts)
            verts = pts[hull.vertices]
            return np.vstack([verts, verts[0]])
        except Exception:
            cx, cy = np.mean(xpix), np.mean(ypix)
            return np.array([[cx-0.5, cy-0.5], [cx+0.5, cy-0.5], [cx, cy+0.5]])

def _enhance(img, method='rescale'):
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

# ---------------------------------------------------------------------
# Colormap with gray midpoint matching "neither"
# ---------------------------------------------------------------------
def compute_gray_band_from_passing(cp_values_filtered, passing_idx_filtered, center=0.5):
    """
    Compute a symmetric gray band [center - δ, center + δ] where δ is the
    minimum distance to center from the closest passing CP on each side.
    If one side has no passing CPs, fall back to the distance available.
    If both sides are empty, δ = 0.
    """
    cp = np.asarray(cp_values_filtered, float)
    pass_mask = np.zeros(len(cp), dtype=bool)
    if len(passing_idx_filtered) > 0:
        pass_mask[np.asarray(list(passing_idx_filtered), int)] = True

    hi = cp[pass_mask & np.isfinite(cp) & (cp > center)]
    lo = cp[pass_mask & np.isfinite(cp) & (cp < center)]

    delta_hi = np.min(hi) - center if hi.size > 0 else np.inf
    delta_lo = center - np.max(lo)  if lo.size > 0 else np.inf

    if np.isinf(delta_hi) and np.isinf(delta_lo):
        delta = 0.0
    elif np.isinf(delta_hi):
        delta = delta_lo
    elif np.isinf(delta_lo):
        delta = delta_hi
    else:
        delta = min(delta_hi, delta_lo)

    lo_edge = center - max(delta, 0.0)
    hi_edge = center + max(delta, 0.0)
    return lo_edge, hi_edge, delta

def make_cmap_with_gray_band(base_cmap_name, vmin, vmax, band_lo, band_hi, gray=(0.6, 0.6, 0.6, 1.0), N=512):
    """
    Build a ListedColormap from a base cmap, but force a flat gray band between
    band_lo and band_hi. vmin and vmax are the numeric limits for normalization.
    """
    base = mpl.cm.get_cmap(base_cmap_name, N)
    colors = base(np.linspace(0, 1, N))

    # map band edges to indices in 0..N-1
    def to_idx(val):
        x = (val - vmin) / max(vmax - vmin, 1e-9)
        x = np.clip(x, 0, 1)
        return int(np.round(x * (N - 1)))

    i_lo = to_idx(band_lo)
    i_hi = to_idx(band_hi)
    if i_hi < i_lo:
        i_lo, i_hi = i_hi, i_lo
    colors[i_lo:i_hi+1] = gray

    return mpl.colors.ListedColormap(colors, name=f"{base_cmap_name}_grayband")
# ---------------------------------------------------------------------
# FOV plot with gray midpoint colorbar
# ---------------------------------------------------------------------
def plot_cp_on_fov_filtered(
    s2p_plane,
    cp_values,                 # length == number of filtered cells
    passing_idx_filtered,      # set or list of indices in filtered space
    cmap='coolwarm',
    center=0.5,
    vmin=None, vmax=None,
    bg_enhance='rescale',
    alpha_fill=0.6,
    outline_color='black',
    outline_lw=0.3,
    save_path=None,
    mid_gray=(0.6, 0.6, 0.6, 1.0),
    show_gray_legend=True
):
    ops, stat, iscell, bg = _load_suite2p(s2p_plane)
    Ly, Lx = ops['Ly'], ops['Lx']
    cell_idx = _filtered_indices(iscell)

    if len(cp_values) != len(cell_idx):
        raise ValueError(f"cp_values length {len(cp_values)} must equal number of filtered cells {len(cell_idx)}")

    cp_values = np.asarray(cp_values, dtype=float)
    passing_idx_filtered = set(int(x) for x in passing_idx_filtered)

    # robust vmin/vmax from filtered CPs
    finite_cp = cp_values[np.isfinite(cp_values)]
    if finite_cp.size == 0:
        raise ValueError("No finite CP values among filtered cells")
    lo_q, hi_q = np.nanpercentile(finite_cp, [1, 99])
    # lo_q = np.nanmin(finite_cp)
    # hi_q = np.nanmax(finite_cp)
    if vmin is None: vmin = lo_q
    if vmax is None: vmax = hi_q
    if vmax <= vmin:
        vmax = vmin + 1e-6  # avoid zero range

    # compute gray band from passing CPs
    band_lo, band_hi, delta = compute_gray_band_from_passing(cp_values, passing_idx_filtered, center=center)

    # make sure color scale covers the band
    vmin = min(vmin, band_lo)
    vmax = max(vmax, band_hi)

    # colormap with gray band
    cmap_obj = make_cmap_with_gray_band(cmap, vmin, vmax, band_lo, band_hi, gray=mid_gray, N=512)

    # normalization centered at 0.5
    if center is not None:
        norm = mpl.colors.TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)
    else:
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    # background
    bg_plot = _enhance(bg, bg_enhance) if bg_enhance else bg
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(bg_plot, cmap='gray')
    ax.axis('off')

    polys, facecolors, edgecolors = [], [], []
    gray = mid_gray

    # draw only filtered ROIs
    for j, roi_i in enumerate(cell_idx):
        s = stat[roi_i]
        verts = _contour_from_pixels(s['xpix'], s['ypix'], Ly, Lx)
        polys.append(verts)

        cp = cp_values[j]
        if (j in passing_idx_filtered) and np.isfinite(cp):
            facecolors.append(cmap_obj(norm(cp)))
        else:
            facecolors.append(gray)
        edgecolors.append(outline_color)

    if polys:
        coll = PolyCollection(polys, facecolors=facecolors, edgecolors=edgecolors,
                              linewidths=outline_lw, alpha=alpha_fill)
        ax.add_collection(coll)

    # colorbar with gray band ticks
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)

    # build ticks highlighting the band
    ticks = [vmin, band_lo, center, band_hi, vmax]
    # remove duplicates in case band collapses
    ticks = [ticks[i] for i in range(len(ticks)) if (i == 0 or abs(ticks[i] - ticks[i-1]) > 1e-9)]
    cbar.set_ticks(ticks)
    ticklabels = []
    for t in ticks:
        if abs(t - center) < 1e-9:
            ticklabels.append(f"{center:.2f}")
        elif abs(t - band_lo) < 1e-9:
            ticklabels.append(f"{band_lo:.2f}  (gray start)")
        elif abs(t - band_hi) < 1e-9:
            ticklabels.append(f"{band_hi:.2f}  (gray end)")
        else:
            ticklabels.append(f"{t:.2f}")
    cbar.set_ticklabels(ticklabels)
    cbar.set_label('Choice probability')

    if show_gray_legend:
        from matplotlib.patches import Patch
        ax.legend(
            handles=[Patch(facecolor=gray, edgecolor=outline_color, label=f"Neither (CP in [{band_lo:.2f}, {band_hi:.2f}])")],
            loc="lower right", frameon=True
        )

    title_band = f"gray band = [{band_lo:.3f}, {band_hi:.3f}]"
    plt.title(f'Choice probability map (filtered only)  •  {title_band}')
    plt.tight_layout()
    if save_path:
        # save_path is a base path without extension in your earlier code
        # keep it consistent with your PDF save helper if you use it
        fig.savefig(f"{save_path}.pdf", dpi=300, bbox_inches='tight')
    plt.show()

    sel_mask = np.zeros(len(cell_idx), dtype=bool)
    if len(passing_idx_filtered) > 0:
        sel_mask[list(passing_idx_filtered)] = True
    return sel_mask
# ---------------------------------------------------------------------
# Grouping: high, low, neither - filtered space only
# ---------------------------------------------------------------------
def build_groups_filtered_with_neither(
    s2p_plane,
    cp_values_filtered,    # length == number of filtered cells
    passing_filtered       # indices in filtered space
):
    """
    High = in passing_filtered and CP > 0.5
    Low = in passing_filtered and CP < 0.5
    Neither = not in passing_filtered
    """
    ops, stat, iscell, _ = _load_suite2p(s2p_plane)
    cell_idx = _filtered_indices(iscell)

    cp_vals = np.asarray(cp_values_filtered, float)
    if len(cp_vals) != len(cell_idx):
        raise ValueError("cp_values_filtered length must equal number of filtered cells")

    pass_mask = np.zeros(len(cell_idx), dtype=bool)
    if len(passing_filtered) > 0:
        pass_mask[np.asarray(list(passing_filtered), int)] = True

    high_mask = pass_mask & np.isfinite(cp_vals) & (cp_vals > 0.5)
    low_mask  = pass_mask & np.isfinite(cp_vals) & (cp_vals < 0.5)
    neither_mask = ~pass_mask

    def coords_from_mask(msk):
        idx = cell_idx[msk]
        if idx.size == 0:
            return np.empty((0, 2))
        return np.array([[np.median(stat[i]['ypix']), np.median(stat[i]['xpix'])] for i in idx])

    coords_high    = coords_from_mask(high_mask)
    coords_low     = coords_from_mask(low_mask)
    coords_neither = coords_from_mask(neither_mask)

    sizes = {"high": int(high_mask.sum()), "low": int(low_mask.sum()), "neither": int(neither_mask.sum())}
    masks = {"high": high_mask, "low": low_mask, "neither": neither_mask, "pass": pass_mask}
    return coords_high, coords_low, coords_neither, sizes, masks

# ---------------------------------------------------------------------
# Core stats
# ---------------------------------------------------------------------
def within_across_gap(coords_group1, coords_group2, weighting='pooled', subsample_reps=0, seed=None):
    """
    Returns within minus across mean distance.
    Uses upper triangle only for within to avoid double counting.
    """
    n1, n2 = len(coords_group1), len(coords_group2)
    if n1 == 0 or n2 == 0:
        return np.nan

    def pooled_gap(g1, g2):
        coords = np.vstack([g1, g2])
        labels = np.array([0]*len(g1) + [1]*len(g2))
        D = distance_matrix(coords, coords)
        iu = np.triu_indices_from(D, k=1)
        i, j = iu
        same = (labels[i] == labels[j])
        within = D[iu][same]
        across = D[iu][~same]
        return np.nanmean(within) - np.nanmean(across)

    if weighting == 'pooled':
        return pooled_gap(coords_group1, coords_group2)

    if weighting == 'balanced':
        # average of within means per group minus across mean
        D11 = distance_matrix(coords_group1, coords_group1)
        D22 = distance_matrix(coords_group2, coords_group2)
        iu1 = np.triu_indices_from(D11, k=1)
        iu2 = np.triu_indices_from(D22, k=1)
        within_bal = 0.5 * (np.nanmean(D11[iu1]) + np.nanmean(D22[iu2]))
        across_mean = np.nanmean(distance_matrix(coords_group1, coords_group2))
        return within_bal - across_mean

    if weighting == 'subsample':
        rng = np.random.default_rng(seed)
        reps = max(1, int(subsample_reps or 1))
        m = min(n1, n2)
        if m < 2:
            return np.nan
        coords = np.vstack([coords_group1, coords_group2])
        labels0 = np.array([0]*n1 + [1]*n2)
        D = distance_matrix(coords, coords)
        iu = np.triu_indices_from(D, k=1)
        i_all, j_all = iu
        vals = []
        for _ in range(reps):
            sel0 = rng.choice(np.where(labels0 == 0)[0], size=m, replace=False)
            sel1 = rng.choice(np.where(labels0 == 1)[0], size=m, replace=False)
            focals = np.sort(np.concatenate([sel0, sel1]))
            # restrict to pairs where both endpoints are focals
            mask_focals = np.isin(i_all, focals) & np.isin(j_all, focals)
            i_f = i_all[mask_focals]; j_f = j_all[mask_focals]
            same = (labels0[i_f] == labels0[j_f])
            d_flat = D[i_f, j_f]
            vals.append(np.nanmean(d_flat[same]) - np.nanmean(d_flat[~same]))
        return float(np.nanmean(vals))

    raise ValueError(f"Unknown weighting: {weighting}")

def perm_test_within_across(coords_group1, coords_group2, n_perm=2000, seed=0,
                            weighting='pooled', subsample_reps=0):
    rng = np.random.default_rng(seed)
    n1, n2 = len(coords_group1), len(coords_group2)
    if n1 == 0 or n2 == 0:
        return np.nan, np.array([]), np.nan

    coords = np.vstack([coords_group1, coords_group2])
    y0 = np.array([0]*n1 + [1]*n2)
    D = distance_matrix(coords, coords)
    iu = np.triu_indices_from(D, k=1)
    i_all, j_all = iu
    d_flat = D[i_all, j_all]

    def gap_from_y(y, weighting):
        if weighting == 'pooled':
            same = (y[i_all] == y[j_all])
            return np.nanmean(d_flat[same]) - np.nanmean(d_flat[~same])

        elif weighting == 'balanced':
            same0 = (y[i_all] == 0) & (y[j_all] == 0)
            same1 = (y[i_all] == 1) & (y[j_all] == 1)
            across = (y[i_all] != y[j_all])
            within_bal = 0.5 * (np.nanmean(d_flat[same0]) + np.nanmean(d_flat[same1]))
            return within_bal - np.nanmean(d_flat[across])

        elif weighting == 'subsample':
            reps = max(1, int(subsample_reps or 1))
            idx0 = np.where(y == 0)[0]
            idx1 = np.where(y == 1)[0]
            m = min(len(idx0), len(idx1))
            if m < 2:
                return np.nan

            vals = []
            for _ in range(reps):
                sel0 = rng.choice(idx0, size=m, replace=False)
                sel1 = rng.choice(idx1, size=m, replace=False)
                focals = np.sort(np.concatenate([sel0, sel1]))

                mask_focals = np.isin(i_all, focals) & np.isin(j_all, focals)
                i_f = i_all[mask_focals]
                j_f = j_all[mask_focals]
                d_flat_f = d_flat[mask_focals]          # <<< key fix

                same = (y[i_f] == y[j_f])
                vals.append(np.nanmean(d_flat_f[same]) - np.nanmean(d_flat_f[~same]))
            return float(np.nanmean(vals))

        else:
            raise ValueError(f"Unknown weighting: {weighting}")

    observed_gap = gap_from_y(y0, weighting)
    perm_gaps = np.empty(n_perm, dtype=float)
    for k in range(n_perm):
        yk = rng.permutation(y0)
        perm_gaps[k] = gap_from_y(yk, weighting)

    pval = np.mean(np.abs(perm_gaps - perm_gaps.mean()) >= np.abs(observed_gap - perm_gaps.mean()))
    return observed_gap, perm_gaps, pval


def clustering_radius_permtest_summary(coords_group1, coords_group2,
                                       radii=None, n_perm=500, seed=0,
                                       keep_perm_curves=False,
                                       weighting='pooled', subsample_reps=0):
    """
    Radial clustering with streaming permutations.
    Returns:
      dict with keys: radii, obs_same, obs_diff, obs_diff_curve, obs_auc, perm_aucs, pval
    """
    # Decoupled RNGs
    rng_perm = np.random.default_rng(seed)
    rng_foc  = np.random.default_rng(seed + 1)

    # Labels: group1 -> 0, group2 -> 1
    coords = np.vstack([coords_group1, coords_group2])
    y_obs_i8 = np.array([0]*len(coords_group1) + [1]*len(coords_group2), dtype=np.int8)
    y_obs = y_obs_i8.astype(np.float64)
    n = len(coords)
    if n == 0:
        return {"radii": np.array([]), "obs_same": np.array([]), "obs_diff": np.array([]),
                "obs_diff_curve": np.array([]), "obs_auc": np.nan, "perm_aucs": np.array([]), "pval": np.nan}

    # KDTree
    tree = cKDTree(coords)

    # Radii choice:
    # - If radii is provided, use it as-is for reproducibility across runs
    # - Else build an adaptive grid from kNN distance quantiles
    if radii is None:
        kq = min(50, max(2, n))
        Dk, _ = tree.query(coords, k=kq)  # includes 0 at self
        nonzero = Dk[:, 1:].ravel()
        nonzero = nonzero[nonzero > 0]
        if nonzero.size > 0:
            lo = np.quantile(nonzero, 0.05)
            hi = np.quantile(nonzero, 0.95)
            # Avoid degenerate small values
            lo = float(max(5.0, lo))
            hi = float(max(lo + 1.0, hi))
            radii = np.linspace(lo, hi, 12)
        else:
            radii = np.array([5.0, 6.0, 7.0], dtype=np.float64)

    radii = np.asarray(radii, dtype=np.float64)
    R = len(radii)

    # Build adjacency per radius (CSR) and degrees in float64
    A_list = []
    deg_list = []
    data_dtype = np.float64
    for r in radii:
        nbrs = tree.query_ball_tree(tree, float(r))
        rows, cols = [], []
        for i, nb in enumerate(nbrs):
            if not nb:
                continue
            # drop self if present
            if i in nb:
                nb = [j for j in nb if j != i]
            if nb:
                rows.extend([i]*len(nb))
                cols.extend(nb)
        if rows:
            data = np.ones(len(rows), dtype=data_dtype)
            A = csr_matrix((data, (np.asarray(rows), np.asarray(cols))), shape=(n, n))
        else:
            A = csr_matrix((n, n), dtype=data_dtype)
        A_list.append(A)
        deg = np.asarray(A.sum(axis=1)).ravel().astype(np.float64)
        deg_list.append(deg)

    def curve_from_labels(y_vec, focals=None):
        """Compute same and diff curves for given labels. y_vec must be float64 in {0.0, 1.0}."""
        same = np.zeros(R, dtype=np.float64)
        diff = np.zeros(R, dtype=np.float64)
        if focals is None:
            for r_idx, A in enumerate(A_list):
                deg = deg_list[r_idx]
                S = A @ y_vec  # counts of neighbors labeled 1, shape (n,)
                same_i = y_vec * S + (1.0 - y_vec) * (deg - S)
                m_same = float(np.mean(same_i))
                same[r_idx] = m_same
                diff[r_idx] = float(np.mean(deg) - m_same)
        else:
            focals = np.asarray(focals, dtype=int)
            if focals.size == 0:
                return np.full(R, np.nan), np.full(R, np.nan), np.full(R, np.nan)
            for r_idx, A in enumerate(A_list):
                deg = deg_list[r_idx]
                S = A @ y_vec
                same_i = y_vec * S + (1.0 - y_vec) * (deg - S)
                m_same = float(np.mean(same_i[focals]))
                same[r_idx] = m_same
                diff[r_idx] = float(np.mean(deg[focals]) - m_same)
        return same, diff, same - diff

    # Observed
    if weighting == 'pooled':
        obs_same, obs_diff, obs_diff_curve = curve_from_labels(y_obs)
    elif weighting == 'subsample':
        reps = max(1, int(subsample_reps or 1))
        idx0 = np.where(y_obs_i8 == 0)[0]
        idx1 = np.where(y_obs_i8 == 1)[0]
        m = min(len(idx0), len(idx1))
        if m < 2:
            return {"radii": radii, "obs_same": np.full(R, np.nan),
                    "obs_diff": np.full(R, np.nan), "obs_diff_curve": np.full(R, np.nan),
                    "obs_auc": np.nan, "perm_aucs": np.array([]), "pval": np.nan}
        acc_same = np.zeros(R, dtype=np.float64)
        acc_diff = np.zeros(R, dtype=np.float64)
        for _ in range(reps):
            f0 = rng_foc.choice(idx0, size=m, replace=False)
            f1 = rng_foc.choice(idx1, size=m, replace=False)
            focals = np.concatenate([f0, f1])
            s, d, _ = curve_from_labels(y_obs, focals=focals)
            acc_same += s
            acc_diff += d
        obs_same = acc_same / reps
        obs_diff = acc_diff / reps
        obs_diff_curve = obs_same - obs_diff
    else:
        raise ValueError("weighting must be 'pooled' or 'subsample'")

    obs_auc = np.nan if np.any(~np.isfinite(obs_diff_curve)) else np.trapz(obs_diff_curve, x=radii)

    # Streaming permutations
    if keep_perm_curves:
        perm_diff_curves = np.empty((n_perm, R), dtype=np.float64)
    perm_aucs = np.empty(n_perm, dtype=np.float64)

    for k in range(n_perm):
        yk_i8 = rng_perm.permutation(y_obs_i8)
        yk = yk_i8.astype(np.float64)

        if weighting == 'pooled':
            perm_curve = np.zeros(R, dtype=np.float64)
            for r_idx, A in enumerate(A_list):
                deg = deg_list[r_idx]
                S = A @ yk
                same_i = yk * S + (1.0 - yk) * (deg - S)
                m_same = float(np.mean(same_i))
                perm_curve[r_idx] = m_same - (float(np.mean(deg)) - m_same)
        else:
            reps = max(1, int(subsample_reps or 1))
            idx0 = np.where(yk_i8 == 0)[0]
            idx1 = np.where(yk_i8 == 1)[0]
            m = min(len(idx0), len(idx1))
            if m < 2:
                perm_curve = np.full(R, np.nan, dtype=np.float64)
            else:
                # choose focals per rep once and reuse across radii
                focals_list = []
                for _ in range(reps):
                    f0 = rng_foc.choice(idx0, size=m, replace=False)
                    f1 = rng_foc.choice(idx1, size=m, replace=False)
                    focals_list.append(np.concatenate([f0, f1]))
                perm_curve = np.zeros(R, dtype=np.float64)
                for r_idx, A in enumerate(A_list):
                    deg = deg_list[r_idx]
                    S = A @ yk
                    same_i = yk * S + (1.0 - yk) * (deg - S)
                    acc = 0.0
                    for focals in focals_list:
                        m_same = float(np.mean(same_i[focals]))
                        acc += m_same - (float(np.mean(deg[focals])) - m_same)
                    perm_curve[r_idx] = acc / reps

        if keep_perm_curves:
            perm_diff_curves[k, :] = perm_curve
        perm_aucs[k] = np.trapz(perm_curve, x=radii)

    # p value on finite perms
    good = np.isfinite(perm_aucs)
    perm_aucs = perm_aucs[good]
    if perm_aucs.size == 0 or not np.isfinite(obs_auc):
        pval = np.nan
    else:
        pval = np.mean(np.abs(perm_aucs - perm_aucs.mean()) >= np.abs(obs_auc - perm_aucs.mean()))

    out = {
        "radii": radii,
        "obs_same": obs_same,
        "obs_diff": obs_diff,
        "obs_diff_curve": obs_diff_curve,
        "obs_auc": obs_auc,
        "perm_aucs": perm_aucs,
        "pval": pval
    }
    if keep_perm_curves:
        out["perm_diff_curves"] = perm_diff_curves
    return out

# ---------------------------------------------------------------------
# Plotting for the topographic analysis
# ---------------------------------------------------------------------
def plot_perm_hist(perm_vals, observed, p_two_sided, xlabel, title, save_base=None):
    fig = plt.figure(figsize=(6.4, 4.0))
    plt.hist(perm_vals, bins=40, alpha=0.7, color='gray', edgecolor='black')
    plt.axvline(observed, color='red', linewidth=2, label=f"Observed = {observed:.2f}")
    plt.axvline(np.mean(perm_vals), color='blue', linestyle='--', label=f"Null mean = {np.mean(perm_vals):.2f}")
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.title(f"{title}\nTwo-sided p = {p_two_sided:.4f}")
    plt.legend(frameon=True)
    plt.tight_layout()
    if save_base:
        save_fig_pdf(save_base, fig=fig)
    plt.show()

def plot_distance_histograms(coords_a, coords_b, name_a, name_b, bins=40, save_base=None):
    d_aa = distance_matrix(coords_a, coords_a)
    d_bb = distance_matrix(coords_b, coords_b)
    d_aa = d_aa[np.triu_indices_from(d_aa, k=1)]
    d_bb = d_bb[np.triu_indices_from(d_bb, k=1)]
    d_ab = distance_matrix(coords_a, coords_b).ravel()

    fig = plt.figure(figsize=(6.4, 4.0))
    plt.hist(d_aa, bins=bins, alpha=0.5, density=True, label=f"{name_a} within", color="tab:blue")
    plt.hist(d_bb, bins=bins, alpha=0.5, density=True, label=f"{name_b} within", color="tab:orange")
    plt.hist(d_ab, bins=bins, alpha=0.5, density=True, label="Across", color="tab:green")
    plt.xlabel("Pairwise distance (px)")
    plt.ylabel("Density")
    plt.title(f"Within and across distances - {name_a} vs {name_b}")
    plt.legend(frameon=True)
    plt.tight_layout()
    if save_base:
        save_fig_pdf(save_base, fig=fig)
    plt.show()

def plot_clustering_curves(res, name_a, name_b, save_base=None, show_ribbon=True):
    radii = res["radii"]
    obs_same = res["obs_same"]
    obs_diff = res["obs_diff"]
    obs_diff_curve = res["obs_diff_curve"]
    pval = res["pval"]
    fig1 = plt.figure(figsize=(6.4, 4.0))
    plt.plot(radii, obs_same, label="Same-label neighbors", lw=2)
    plt.plot(radii, obs_diff, label="Different-label neighbors", lw=2)
    plt.xlabel("Radius (px)")
    plt.ylabel("Mean neighbor count")
    plt.title(f"Neighbor curves - {name_a} vs {name_b}")
    plt.legend(frameon=True)
    plt.tight_layout()
    if save_base:
        save_fig_pdf(save_base + "_curves", fig=fig1)
    plt.show()

    fig2 = plt.figure(figsize=(6.4, 4.0))
    plt.plot(radii, obs_diff_curve, lw=2, label="Same - Diff")
    if show_ribbon and ("perm_diff_curves" in res):
        diffs = res["perm_diff_curves"]  # [n_perm, len(radii)]
        lo = np.quantile(diffs, 0.025, axis=0)
        hi = np.quantile(diffs, 0.975, axis=0)
        plt.fill_between(radii, lo, hi, color="gray", alpha=0.25, label="Null 95%")
    plt.axhline(0, color="k", lw=1, ls="--")
    plt.xlabel("Radius (px)")
    plt.ylabel("Same - Diff neighbors")
    plt.title(f"Clustering diff curve (AUC p={pval:.4f}) - {name_a} vs {name_b}")
    plt.legend(frameon=True)
    plt.tight_layout()
    if save_base:
        save_fig_pdf(save_base + "_diff", fig=fig2)
    plt.show()

def plot_forest(df, pooled_obs=None, pooled_null=None, title="Within minus across distance by mouse", save_all_loc=None):
    # df must have: mouse, day, n_a, n_b, obs_gap, pval, ci_low, ci_high
    df_plot = df.sort_values("obs_gap")
    y = np.arange(len(df_plot))

    plt.figure(figsize=(8, max(3.6, 0.6*len(df_plot)+1.6)))
    # CIs
    plt.hlines(y, df_plot["ci_low"], df_plot["ci_high"], color="gray", lw=2)
    # point estimates
    plt.plot(df_plot["obs_gap"], y, "o", color="black")
    plt.axvline(0, color="k", ls="--", lw=1)

    # pooled shading + line if provided
    if pooled_obs is not None and pooled_null is not None:
        pooled_mean = pooled_null.mean()
        pooled_ci = np.quantile(pooled_null, [0.025, 0.975])
        # Convert pooled sum to mean per mouse for readability
        pooled_obs_mean = pooled_obs / max(1, len(df_plot))
        pooled_ci_mean = pooled_ci / max(1, len(df_plot))
        plt.axvspan(pooled_ci_mean[0], pooled_ci_mean[1], color="tomato", alpha=0.15, label="Pooled 95% null")
        plt.axvline(pooled_obs_mean, color="tomato", lw=2, label=f"Pooled observed mean = {pooled_obs_mean:.2f}")

    labels = [f"{m} {d}  (n={na}|{nb})  p={p:.5f}"
              for m, d, na, nb, p in zip(df_plot["mouse"], df_plot["day"], df_plot["n_a"], df_plot["n_b"], df_plot["pval"])]
    plt.yticks(y, labels)
    plt.xlabel("Within minus across mean distance (px)")
    plt.title(title)
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    if save_all_loc is not None:
        Path(save_all_loc).mkdir(parents=True, exist_ok=True)
        plt.savefig(join(save_all_loc, 'forest_plot.pdf'), bbox_inches="tight")
    plt.show()

def plot_clustering_across_mice_by_pair(results_list, mice_ids, title="Clustering AUC across mice", save_all_loc=None):
    # results_list: list of dicts with keys obs_auc, pval
    obs_aucs = [r["obs_auc"] for r in results_list]
    pvals = [r["pval"] for r in results_list]
    plt.figure(figsize=(7, max(3.6, 0.5*len(results_list)+1)))
    y = np.arange(len(results_list))
    colors = ["tomato" if p < 0.05 else "gray" for p in pvals]
    plt.barh(y, obs_aucs, color=colors)
    plt.axvline(0, color="k", ls="--", lw=1)
    labels = [f"{m}  (p={p:.3f})" for m, p in zip(mice_ids, pvals)]
    plt.yticks(y, labels)
    plt.xlabel("Clustering AUC (same - diff)")
    plt.title(title)
    plt.tight_layout()
    if save_all_loc is not None:
        Path(save_all_loc).mkdir(parents=True, exist_ok=True)
        plt.savefig(join(save_all_loc, 'clustering_across_mice.pdf'), bbox_inches="tight")
    plt.show()

# ---------------------------------------------------------------------
# Load CP and run per mouse
# ---------------------------------------------------------------------
mice_days = [
    ('OI116', '240529'),
    ('OI124', '240610'),
    ('OI151', '240827'),
    ('OI152', '241003'),
    ('OI208', '250227'),
]

for mouse_id, day_str in mice_days:
    real_data_path = fr'Z:\2p_data\{mouse_id}\{day_str}\ML\data\processed_data\auROC_freq_history'
    cp_results_path = fr'Z:\2p_data\{mouse_id}\{day_str}\ML\data\processed_data\auROC_freq_history_control_CR-CR_rep1\test_all_reps_v2'

    with open(os.path.join(real_data_path, 'res_cr_subsample_global.pkl'), 'rb') as f:
        real_choice_prob = NumpyCoreUnpickler(f).load()
    real_cp = [np.nanmean(real_choice_prob['all_distributions'][i])
               for i in range(len(real_choice_prob['all_distributions']))]

    s2p_dir = fr'Z:\2p_data\{mouse_id}\{day_str}\{mouse_id}_run1\suite2p\plane0'

    with open(join(cp_results_path, "comparison_results.pkl"), "rb") as f:
        comparison_results = pickle.load(f)

    passing_idx = set(comparison_results['indices_passing_thresholds']['exc_above_97_5th_idx']) | \
                  set(comparison_results['indices_passing_thresholds']['inh_below_2_5th_idx'])

    # 1) FOV plot with gray midpoint colorbar
    fov_save_base = fr'Z:\2p_data\{mouse_id}\{day_str}\ML\data\processed_data\topography_plots_97_5_2_5v3\{mouse_id}_{day_str}_cp_fov'
    Path(Path(fov_save_base).parent).mkdir(parents=True, exist_ok=True)
    sel_mask = plot_cp_on_fov_filtered(
        s2p_dir,
        real_cp,
        passing_idx_filtered=passing_idx,
        cmap='coolwarm',
        center=0.5,
        bg_enhance='clahe',
        alpha_fill=0.55,
        outline_lw=0.25,
        save_path=fov_save_base,
        mid_gray=(0.6, 0.6, 0.6, 1.0),
        show_gray_legend=True
    )

    # 2) Build groups and run stats for all three comparisons
    coords_high, coords_low, coords_neither, sizes, masks = build_groups_filtered_with_neither(
        s2p_dir, np.asarray(real_cp), passing_idx
    )
    print(f"\n{mouse_id} {day_str} - group sizes: high={sizes['high']}, low={sizes['low']}, neither={sizes['neither']}")

    pairs = {
        "high_vs_low":      ("High", "Low", coords_high, coords_low),
        "high_vs_neither":  ("High", "Neither", coords_high, coords_neither),
        "low_vs_neither":   ("Low", "Neither", coords_low, coords_neither),
    }

    pair_save_dir = fr'Z:\2p_data\{mouse_id}\{day_str}\ML\data\processed_data\topography_plots_95_5_v3\pairs'
    Path(pair_save_dir).mkdir(parents=True, exist_ok=True)

    for key, (name_a, name_b, A, B) in pairs.items():
        if len(A) == 0 or len(B) == 0:
            print(f"{key}: one group empty - skipping plots and stats.")
            continue

        # 2a) Within-across gap permutation + histogram plot (subsampled, no replacement)
        obs_gap, perm_gaps, p_gap = perm_test_within_across(A, B, n_perm=500, seed=123, weighting='subsample', subsample_reps=500)
        print(f"{key}: gap obs={obs_gap:.2f}, p={p_gap:.4f}")
        plot_perm_hist(
            perm_vals=perm_gaps,
            observed=obs_gap,
            p_two_sided=p_gap,
            xlabel="Within - Across mean distance (px)",
            title=f"{name_a} vs {name_b}",
            save_base=join(pair_save_dir, f"{mouse_id}_{day_str}_{key}_gap_perm_hist")
        )

        # 2b) Distance histograms
        plot_distance_histograms(
            A, B, name_a, name_b,
            bins=40,
            save_base=join(pair_save_dir, f"{mouse_id}_{day_str}_{key}_distance_hists")
        )

        # 2c) Clustering curves and diff curves with ribbon from permutations
        print("starting clustering...")
        radii_fixed = np.array([8, 12, 16, 24, 32, 48, 64, 96, 128, 160, 180], dtype=np.float64)
        clust = clustering_radius_permtest_summary(A, B, radii=radii_fixed, n_perm=500, seed=999, keep_perm_curves=False, weighting='subsample', subsample_reps=500)
        print(f"{key}: clustering AUC={clust['obs_auc']:.2f}, p={clust['pval']:.4f}")
        plot_clustering_curves(
            clust, name_a, name_b,
            save_base=join(pair_save_dir, f"{mouse_id}_{day_str}_{key}_clustering"),
            show_ribbon=False
        )

        if 'cp_gap_across_mice' not in globals():
            cp_gap_across_mice = []
        if 'cp_clust_across_mice' not in globals():
            cp_clust_across_mice = []

        mouse_tag = f"{mouse_id}_{day_str}"

        # example inside your pairs loop:
        # for key, (name_a, name_b, A, B) in pairs.items():
        if len(A) > 0 and len(B) > 0:
            # store gap summary + raw perms for pooled null
            cp_gap_across_mice.append({
                "mouse": mouse_id,
                "day": day_str,
                "mouse_tag": mouse_tag,
                "pair": key,  # "high_vs_low" etc.
                "n_a": len(A),
                "n_b": len(B),
                "obs_gap": obs_gap,
                "pval": p_gap,
                "ci_low": np.quantile(perm_gaps, 0.025),
                "ci_high": np.quantile(perm_gaps, 0.975),
                "perm_gaps": perm_gaps  # keep for pooled null
            })
            # store clustering summary + perms
            cp_clust_across_mice.append({
                "mouse": mouse_id,
                "day": day_str,
                "mouse_tag": mouse_tag,
                "pair": key,
                "obs_auc": clust["obs_auc"],
                "pval": clust["pval"],
                "perm_aucs": clust["perm_aucs"]  # keep for pooled null of summed AUC
            })

# ==== Across-mice summaries for the three comparisons ====
pairs_order = ["high_vs_low", "high_vs_neither", "low_vs_neither"]
pair_titles = {
    "high_vs_low": "High vs Low",
    "high_vs_neither": "High vs Neither",
    "low_vs_neither": "Low vs Neither",
}

summary_root = r"Z:\Stavsht\Omer_OXTR\topography_cp_summary_95_5_v3"
Path(summary_root).mkdir(parents=True, exist_ok=True)

# GAP summaries
df_gap_all = pd.DataFrame([{k: v for k, v in d.items() if k != "perm_gaps"} for d in cp_gap_across_mice])
print("\nAcross-mice CP gap summary rows:")
if not df_gap_all.empty:
    print(df_gap_all[["mouse", "day", "pair", "n_a", "n_b", "obs_gap", "pval", "ci_low", "ci_high"]])

for pair in pairs_order:
    sub = [d for d in cp_gap_across_mice if
           d["pair"] == pair and isinstance(d.get("perm_gaps", None), np.ndarray) and d["perm_gaps"].size > 0]
    if len(sub) == 0:
        print(f"\n{pair_titles[pair]}: no valid mice for pooling.")
        continue

    # build DataFrame for forest plot
    df_pair = pd.DataFrame([{k: v for k, v in d.items() if k != "perm_gaps"} for d in sub])

    # pooled observed sum and pooled null as element-wise sum of perms
    # ensure same number of permutations across mice
    n_perms = min(len(d["perm_gaps"]) for d in sub)
    pooled_null = None
    for d in sub:
        pg = d["perm_gaps"][:n_perms]
        pooled_null = pg.copy() if pooled_null is None else pooled_null + pg
    obs_pooled = df_pair["obs_gap"].sum()

    pval_pooled_two_sided = np.mean(
        np.abs(pooled_null - pooled_null.mean()) >= np.abs(obs_pooled - pooled_null.mean()))
    pooled_ci_low, pooled_ci_high = np.quantile(pooled_null, [0.025, 0.975])

    print(f"\n{pair_titles[pair]} - pooled observed gap sum = {obs_pooled:.2f}")
    print(f"{pair_titles[pair]} - pooled two sided p = {pval_pooled_two_sided:.4f}")
    print(f"{pair_titles[pair]} - pooled 95% null interval = [{pooled_ci_low:.2f}, {pooled_ci_high:.2f}]")

    # forest plot per pair
    pair_dir = join(summary_root, f"{pair}_forest")
    plot_forest(
        df_pair[["mouse", "day", "n_a", "n_b", "obs_gap", "pval", "ci_low", "ci_high"]],
        pooled_obs=obs_pooled,
        pooled_null=pooled_null,
        title=f"Topography CP: {pair_titles[pair]} within-across per mouse",
        save_all_loc=pair_dir
    )

# CLUSTERING AUC summaries across mice
for pair in pairs_order:
    sub = [d for d in cp_clust_across_mice if
           d["pair"] == pair and isinstance(d.get("perm_aucs", None), np.ndarray) and d["perm_aucs"].size > 0]
    if len(sub) == 0:
        print(f"\n{pair_titles[pair]}: no valid mice for clustering summary.")
        continue

    # barh plot across mice for this pair
    mice_ids = [d["mouse_tag"] for d in sub]
    results_list = [{"obs_auc": d["obs_auc"], "pval": d["pval"]} for d in sub]
    pair_dir = join(summary_root, f"{pair}_clustering")
    plot_clustering_across_mice_by_pair(results_list, mice_ids, title=f"Clustering AUC - {pair_titles[pair]}",
                                        save_all_loc=pair_dir)

    # optional pooled AUC test on summed AUCs
    n_perms = min(len(d["perm_aucs"]) for d in sub)
    pooled_null_auc = None
    for d in sub:
        pa = d["perm_aucs"][:n_perms]
        pooled_null_auc = pa.copy() if pooled_null_auc is None else pooled_null_auc + pa
    obs_auc_sum = np.sum([d["obs_auc"] for d in sub])
    p_pooled_auc = np.mean(
        np.abs(pooled_null_auc - pooled_null_auc.mean()) >= np.abs(obs_auc_sum - pooled_null_auc.mean()))
    ci_auc_low, ci_auc_high = np.quantile(pooled_null_auc, [0.025, 0.975])

    print(
        f"{pair_titles[pair]} - pooled AUC sum = {obs_auc_sum:.2f}, pooled p = {p_pooled_auc:.4f}, pooled 95% null = [{ci_auc_low:.2f}, {ci_auc_high:.2f}]")
