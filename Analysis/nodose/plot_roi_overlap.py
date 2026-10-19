#!/usr/bin/env python
"""
Plot saved 2-channel ROIs and list overlaps.

Usage
-----
python plot_roi_overlap.py path/to/int_000_roi_traces.npz \
                           --raw path/to/Image_001_001.raw \
                           --width 512 --height 512 --n_planes 5 \
                           --plane 2           # plot just plane 2
If --raw is omitted, ROIs are drawn on a blank canvas.
"""

import os, argparse, json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from itertools import product
import matplotlib
import matplotlib.patheffects as pe
matplotlib.use("TkAgg")  # use TkAgg backend for interactive plotting
DEFAULT_NPZ  = r"Z:\2p_data\Omer_gastric_balloons\OI348\OI348int_balloon1\Image_001_001_roi_traces.npz"
DEFAULT_RAW  = r"Z:\2p_data\Omer_gastric_balloons\OI348\OI348int_balloon1\Image_001_001.raw"   # leave "" if no raw imag
# ---------- helpers ----------------------------------------------------------
def load_roi_table(npz_path):
    """Return list of dicts with cx, cy, rx, ry, channel, plane."""
    roi_tbl = np.load(npz_path, allow_pickle=True)["roi_table"]
    # Stored as 0-D ndarray of dicts → convert to plain list
    if isinstance(roi_tbl, np.ndarray) and roi_tbl.dtype == object:
        roi_tbl = list(roi_tbl)
    return roi_tbl
def enhance_contrast(img, low=1, high=99, gamma=None):
    """
    Stretch intensities between the given percentiles.
    Optionally apply gamma correction.
    Returns a float image in [0, 1].
    """
    img = img.astype(np.float32)
    v_min, v_max = np.percentile(img, (low, high))
    img = np.clip(img, v_min, v_max)
    img = (img - v_min) / (v_max - v_min + 1e-6)
    if gamma is not None and gamma > 0:
        img = img ** gamma
    return img
def std_projection(memmap, n_sample=500):
    idx = np.linspace(0, memmap.shape[0]-1, n_sample, dtype=int)
    return memmap[idx].std(axis=0)

def load_raw_memmap(path, width, height, dtype=np.uint16):
    n_pix  = os.path.getsize(path) // np.dtype(dtype).itemsize
    frames = n_pix // (width * height)
    return np.memmap(path, dtype=dtype, mode="r",
                     shape=(frames, height, width))

def ellipse_overlap(e1, e2, n_pts=10_000):
    """Monte-Carlo estimate of area overlap between two ellipses."""
    cx1, cy1, rx1, ry1 = e1
    cx2, cy2, rx2, ry2 = e2
    # Sample points in the bounding box of both ellipses
    xmin = min(cx1-rx1, cx2-rx2); xmax = max(cx1+rx1, cx2+rx2)
    ymin = min(cy1-ry1, cy2-ry2); ymax = max(cy1+ry1, cy2+ry2)
    xs = np.random.uniform(xmin, xmax, size=n_pts)
    ys = np.random.uniform(ymin, ymax, size=n_pts)
    in1 = ((xs-cx1)**2)/(rx1**2) + ((ys-cy1)**2)/(ry1**2) <= 1
    in2 = ((xs-cx2)**2)/(rx2**2) + ((ys-cy2)**2)/(ry2**2) <= 1
    inter = (in1 & in2).sum()
    return inter / n_pts  # fraction of box; use only for ranking

# ---------- main plotting routine -------------------------------------------
def plot_plane(rois, plane_idx, ax, bg=None, alpha=.35, lab_fs=7, dedup_map=None):
    """Draw all ROIs for one plane on ax; bg can be a 2-D image."""
    if bg is not None:
        ax.imshow(bg, cmap="gray")

    for r in rois:
        col = "green" if r["channel"] == "green" else "red"
        # highlight deduplicated (excluded) green cells
        is_excluded = False
        if dedup_map is not None and r.get("channel") == "green":
            seg = r["segment"]
            pl  = r["plane"]
            row = r["roi_id"] - 1
            drop = dedup_map.get((seg, pl))
            if drop is not None and row in drop:
                is_excluded = True

        edge_col = "magenta" if is_excluded else col
        lw       = 2.2 if is_excluded else 1.4
        ell = patches.Ellipse((r["cx"], r["cy"]),
                              width=2*r["rx"], height=2*r["ry"],
                              edgecolor=edge_col, facecolor="none",
                              linewidth=lw, alpha=1)
        ax.add_patch(ell)
        ell.set_alpha(alpha)

        if is_excluded:
            ax.plot(r["cx"], r["cy"], marker="x", color="magenta", markersize=4, mew=1.2)

        # ------- Add small numeric label ---------
        # label = f"{r['roi_id']}"           # just the number
        # x_txt = r["cx"] + r["rx"] + 2      # a couple pixels to the right
        # y_txt = r["cy"]
        # ax.text(
        #     x_txt, y_txt, label,
        #     color=col, fontsize=lab_fs, weight="light",
        #     va="center", ha="left",
        #     path_effects=[pe.withStroke(linewidth=0.5, foreground="black")]
        # )

    ax.set_title(f"Plane {plane_idx+1}")
    ax.set_axis_off()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("npz",
                   nargs="?",
                   default=DEFAULT_NPZ,
                   help="path to *_roi_traces.npz")
    p.add_argument("--raw",
                   default=DEFAULT_RAW if DEFAULT_RAW else None,
                   help="optional raw file for background image")
    p.add_argument("--width",  type=int, default=512)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--n_planes", type=int, default=1)
    p.add_argument("--plane", type=int, default=None,
                   help="plot just this plane (1-based); omit for all")
    p.add_argument("--sample_bg", type=int, default=500,
                   help="frames used for std projection")
    p.add_argument("--clip_lo", type=float, default=1.0)
    p.add_argument("--clip_hi", type=float, default=99.0)
    p.add_argument("--gamma", type=float, default=1)
    # cross-plane deduplication params
    p.add_argument("--dedup_px", type=float, default=200.0,
                   help="centroid distance threshold (pixels) to consider the same cell across planes")
    p.add_argument("--dedup_all_planes", action="store_true",
                   help="compare all planes for deduplication (not just adjacent planes)")
    p.add_argument("--dedup_corr", type=float, default=0.6,
                   help="if set, additionally require Pearson r between traces >= this value")
    p.add_argument("--drop_dedup", action="store_true",
                   help="exclude deduplicated green ROIs from plotting and overlap outputs; also save roi_table_deduped to NPZ")
    args = p.parse_args()

    roi_tbl = load_roi_table(args.npz)
    
    if args.plane is not None:
        roi_tbl = [r for r in roi_tbl if r["plane"] == args.plane-1]

    # group by plane
    planes = sorted(set(r["plane"] for r in roi_tbl))
    if len(planes) == 1:
        # one plane → one axes that fills the canvas
        fig, axes = plt.subplots(figsize=(6, 6))  # adjust size as you like
        axes = np.array([axes])  # keep later analysis_code unchanged
    else:
        ncols = 3
        fig, axes = plt.subplots(
            nrows=int(np.ceil(len(planes) / ncols)), ncols=ncols,
            figsize=(4 * ncols, 4 * len(planes) / ncols)
        )

    if not isinstance(axes, np.ndarray):
        axes = np.array([[axes]])
    axes = axes.flatten()

    bg_imgs = {}
    if args.raw:
        raw = load_raw_memmap(args.raw, args.width, args.height)
        for pl in planes:
            # frames alternate green/red, so pick green frames for bg
            plane_frames = raw[0::2][pl::args.n_planes]
            std_proj = std_projection(plane_frames, n_sample=args.sample_bg)
            bg_imgs[pl] = enhance_contrast(std_proj,
                                           low=args.clip_lo,
                                           high=args.clip_hi,
                                           gamma=args.gamma)

    # -----------------------------------------------------------------
    # Compute cross-plane de-duplication BEFORE drawing, to annotate
    # -----------------------------------------------------------------
    # Build helper maps for quick access
    # -------------------------------------------------------------------------
    # EXPLANATION OF THIS SECTION:
    #
    # This block performs cross-plane de-duplication of green-channel ROIs (cells).
    # The goal is to identify and mark green ROIs that likely represent the same cell
    # imaged in multiple planes, so that they can be annotated or excluded from Analysis.
    #
    # The logic is:
    # 1. For each segment and each pair of planes (either only adjacent or all pairs),
    #    compare all green ROIs in those planes.
    # 2. For each green ROI in plane q, check if there is a green ROI in plane p
    #    (with p < q) that is spatially close (within a pixel threshold).
    # 3. Optionally, also require that the traces (time series) of the two ROIs
    #    are highly correlated (above a user-specified Pearson r threshold).
    # 4. If both criteria are met, mark the ROI in plane q for dropping (as a duplicate).
    # 5. The result is a dictionary dedup_to_drop_arr mapping (segment, plane) to
    #    arrays of green ROI indices to drop.
    # -------------------------------------------------------------------------

    # Load the .npz file as a dict for trace access
    npz_dict = dict(np.load(args.npz, allow_pickle=True))

    # Build a mapping: (segment, plane) -> list of green ROIs in that segment/plane
    greens_by_seg_plane = {}
    for r in roi_tbl:
        if r.get("channel") != "green":
            continue
        seg = r["segment"]
        pl  = r["plane"]
        greens_by_seg_plane.setdefault((seg, pl), []).append(r)

    def get_green_row(seg_zero_based, plane_zero_based, roi_one_based):
        key = f"green_seg{seg_zero_based+1}_plane{plane_zero_based+1}_traces"
        arr = npz_dict.get(key)
        if arr is None:
            return None
        idx = roi_one_based - 1
        if 0 <= idx < arr.shape[0]:
            return arr[idx]
        return None

    plane_list = sorted(set(r["plane"] for r in roi_tbl))
    if args.dedup_all_planes:
        plane_pairs = [(p, q) for i, p in enumerate(plane_list) for q in plane_list[i+1:]]
    else:
        plane_pairs = [(p, q) for p, q in zip(plane_list[:-1], plane_list[1:])]

    dedup_to_drop = {}
    for seg in sorted(set(r["segment"] for r in roi_tbl)):
        for p, q in plane_pairs:
            A = greens_by_seg_plane.get((seg, p), [])
            B = greens_by_seg_plane.get((seg, q), [])
            if not A or not B:
                continue

            A_xy = np.array([[a["cx"], a["cy"]] for a in A], dtype=float)
            B_xy = np.array([[b["cx"], b["cy"]] for b in B], dtype=float)

            drop_rows_q = set(dedup_to_drop.get(f"dedup_drop_green_seg{seg}_plane{q}", []))
            for j, b in enumerate(B):
                row_b = b["roi_id"] - 1
                if row_b in drop_rows_q:
                    continue
                dists = np.linalg.norm(A_xy - B_xy[j], axis=1)
                i_min = int(np.argmin(dists))
                d_min = float(dists[i_min])
                if d_min > args.dedup_px:
                    continue
                # Optional correlation filter: only apply when threshold > 0
                if args.dedup_corr is not None and args.dedup_corr > 0:
                    tr_a = get_green_row(seg, p, A[i_min]["roi_id"])
                    tr_b = get_green_row(seg, q, B[j]["roi_id"])
                    ok = False
                    if tr_a is not None and tr_b is not None and tr_a.size == tr_b.size and tr_a.size > 0:
                        ta = (tr_a - tr_a.mean()) / (tr_a.std() + 1e-6)
                        tb = (tr_b - tr_b.mean()) / (tr_b.std() + 1e-6)
                        r_val = float(np.corrcoef(ta, tb)[0, 1])
                        ok = r_val >= args.dedup_corr
                    if not ok:
                        continue
                key_q = f"dedup_drop_green_seg{seg}_plane{q}"
                dedup_to_drop.setdefault(key_q, set()).add(row_b)

    dedup_to_drop_arr = {k: np.array(sorted(list(v)), dtype=int) for k, v in dedup_to_drop.items()}

    # Build a quick lookup to annotate during plotting
    dedup_map = {}
    for key, arr in dedup_to_drop_arr.items():
        # key format: dedup_drop_green_seg{seg}_plane{pl}
        try:
            seg = int(key.split("_seg")[1].split("_plane")[0])
            pl  = int(key.split("_plane")[1])
            dedup_map[(seg, pl)] = set(arr.tolist())
        except Exception:
            continue

    # Optionally drop deduplicated green ROIs from downstream outputs and save table
    if args.drop_dedup and dedup_map:
        before_n = len([r for r in roi_tbl if r.get("channel") == "green"])
        roi_tbl = [
            r for r in roi_tbl
            if not (
                r.get("channel") == "green"
                and (r["segment"], r["plane"]) in dedup_map
                and (r["roi_id"] - 1) in dedup_map[(r["segment"], r["plane"])])
        ]
        after_n = len([r for r in roi_tbl if r.get("channel") == "green"])
        print(f"Dropped {before_n - after_n} deduplicated green ROIs (kept {after_n}).")

        try:
            np.savez_compressed(
                args.npz,
                **{**dict(np.load(args.npz, allow_pickle=True)),
                   **{"roi_table_deduped": np.array(roi_tbl, dtype=object)}}
            )
            print("Saved 'roi_table_deduped' to NPZ.")
        except Exception as e:
            print(f"Warning: failed to save roi_table_deduped: {e}")

    # draw with annotations
    for ax, pl in zip(axes, planes):
        rois = [r for r in roi_tbl if r["plane"] == pl]
        bg   = bg_imgs.get(pl) if bg_imgs else None
        plot_plane(rois, pl, ax, bg=bg, dedup_map=dedup_map)

    for ax in axes[len(planes):]:
        ax.set_axis_off()

    plt.tight_layout()
    plt.savefig(args.npz.replace(".npz", "_roi_planes.png"))
    plt.show()

    # ----------- simple numeric overlap report ------------------------------
    print("\nPossible overlaps (plane, red_ID, green_ID):")
    for pl in planes:
        reds   = [r for r in roi_tbl if r["plane"] == pl and r["channel"]=="red"]
        greens = [r for r in roi_tbl if r["plane"] == pl and r["channel"]=="green"]
        for rr, gg in product(reds, greens):
            frac = ellipse_overlap(
                (rr["cx"], rr["cy"], rr["rx"], rr["ry"]),
                (gg["cx"], gg["cy"], gg["rx"], gg["ry"])
            )
            if frac > 0.1:   # 10 % of bounding box → tweak as needed
                print(f" plane {pl+1}:  red#{rr['roi_id']}  &  green#{gg['roi_id']}  (≈{frac*100:.1f}% overlap)")

    # ------------------------------------------------------------
    # build a dict  key = 'overlap_green_seg{seg}_plane{pl}'
    #               val = np.array of ZERO-BASED green row indices
    # ------------------------------------------------------------
    OVERLAP_FRAC_THRESHOLD = 0.3
    overlap_dict = {}

    for seg in sorted(set(r["segment"] for r in roi_tbl)):
        for pl in planes:
            reds = [r for r in roi_tbl
                    if r["segment"] == seg and r["plane"] == pl
                    and r["channel"] == "red"]
            greens = [r for r in roi_tbl
                      if r["segment"] == seg and r["plane"] == pl
                      and r["channel"] == "green"]

            # which green ROIs intersect any red ROI?
            overlapping = []
            for g in greens:
                g_box = (g["cx"], g["cy"], g["rx"], g["ry"])
                for r in reds:
                    r_box = (r["cx"], r["cy"], r["rx"], r["ry"])
                    if ellipse_overlap(g_box, r_box, n_pts=8_000) > OVERLAP_FRAC_THRESHOLD:
                        overlapping.append(g["roi_id"] - 1)  # zero-based row
                        break

            key = f"overlap_green_seg{seg}_plane{pl}"
            overlap_dict[key] = np.array(sorted(set(overlapping)), dtype=int)

    # save into the same NPZ
    np.savez_compressed(
        args.npz,  # overwrite the file you opened
        **{**dict(np.load(args.npz, allow_pickle=True)), **overlap_dict}
    )
    print("Saved overlap index arrays to NPZ:")
    for k, v in overlap_dict.items():
        print(f"  {k}: {len(v)} rows")

    if dedup_to_drop_arr:
        np.savez_compressed(
            args.npz,
            **{**dict(np.load(args.npz, allow_pickle=True)), **dedup_to_drop_arr}
        )
        print("Saved cross-plane de-duplication indices to NPZ:")
        for k, v in dedup_to_drop_arr.items():
            print(f"  {k}: {len(v)} rows to drop")


if __name__ == "__main__":
    main()
