"""
NODOSE GANGLION 2-PHOTON CALCIUM IMAGING ANALYSIS PIPELINE
==========================================================

Fixes in this version:
1) Phase-aware mapping from global to per-plane time that includes flyback.
   - Scan cycle length is pinned to 4 (your setup), even though you ignore flyback in Analysis.
   - Plane phase for Analysis planes is computed as 1,2,3 when flyback is phase 0.

2) Correct de-interleave formula:
   per-plane index k = (g - plane_phase) // n_planes_scan   (no +1)

3) Preserve open/close parity from the original global ordering.
   - map_valves_to_plane now returns both per-plane indices and the original global positions.
   - events_for_segment uses those original positions to decide open vs close.

4) Safer key lookups for NPZ channels and overlap indices (tries both 0-based and 1-based plane keys).
"""

# ==================================================================
# 0) IMPORTS & GLOBAL SETTINGS
# ==================================================================
import re
import csv
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.colors import TwoSlopeNorm
from scipy.optimize import curve_fit
from scipy.signal import butter, filtfilt
from scipy.stats import linregress


# Use Qt backend if available (set before importing pyplot ideally)
matplotlib.use('Qt5Agg')
matplotlib.rcParams["pdf.fonttype"] = 42

# Regex for segment keys in the NPZ
seg_pat = re.compile(r'^(green|red|red_ctrl)_seg(\d+).*_traces$')

# ==================================================================
# 1) CONFIGURATION
# ==================================================================

ROI_NPZ_PATH = Path(r'Z:\2p_data\Omer_gastric_balloons\OI305\OI305int_balloon1\Image_001_001_roi_traces.npz')
VALVE_TXT_PATH = Path(r'Z:\2p_data\Omer_gastric_balloons\OI305\OI305_sync_run1\Episode_0000_rising_indices.txt')

rec_freq = 3  # frames per second

# Scanner cycle includes a flyback plane that is not analyzed
N_PLANES_SCAN = 4
FLYBACK_PHASE = 0
AUTO_ALIGN_BY_CTRL = False

# Small quality of life options
AUTO_ALIGN_BY_CTRL = True  # set False to disable cross-correlation alignment
MAX_ALIGN_SEC = 2.0  # search window for auto-alignment in seconds
MANUAL_SHIFT_FRAMES = 0  # constant shift applied after mapping and optional auto-align

# If None, will process all Analysis planes found in NPZ in 0-based Analysis indexing
PLANES_TO_PROCESS = None
CSV_HEADER = [
    "plane", "segment", "cell_index",
    "is_responder", "n_opens", "n_closes", "n_pairs",
    "total_event_shift_frames", "mapping_mode"
]

def append_rows_to_csv(csv_path: Path, rows):
    file_exists = csv_path.exists()
    with open(csv_path, 'a', newline='') as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(CSV_HEADER)
        if rows:
            w.writerows(rows)

# ==================================================================
# 2) DATA DISCOVERY & BASIC HELPERS
# ==================================================================
def plane_phase_in_scan(analysis_plane_idx_0based: int) -> int:
    return (FLYBACK_PHASE + 1 + analysis_plane_idx_0based) % N_PLANES_SCAN
def get_available_planes(npz):
    """
    Return sorted unique plane identifiers as they appear in roi_table.
    These may be 1-based depending on the pipeline that created the NPZ.
    We will treat Analysis indices as 0..len-1 separately.
    """
    if 'roi_table' not in npz:
        print("Warning: No roi_table found in NPZ file")
        return []
    roi_table = npz['roi_table']
    if isinstance(roi_table, np.ndarray) and roi_table.dtype == object:
        roi_table = list(roi_table)
    planes = sorted(set(int(r['plane']) for r in roi_table))
    return planes


def determine_planes_to_process(npz, planes_to_process=None):
    """
    Analysis indexing is 0..(n_analysis_planes-1) regardless of the numeric labels
    stored in the NPZ. If planes_to_process is None, return all Analysis indices.
    If a list is given, it is interpreted as 0-based Analysis indices.
    """
    available_planes = get_available_planes(npz)
    n = len(available_planes)
    if n == 0:
        print("No planes found in data")
        return []
    if planes_to_process is None:
        return list(range(n))
    valid = [p for p in planes_to_process if isinstance(p, int) and 0 <= p < n]
    if not valid:
        print(f"Warning: none of the specified Analysis plane indices {planes_to_process} are valid; using all")
        return list(range(n))
    return valid


def collect_segments(npz):
    seg_dict = {}
    for key in npz.keys():
        m = seg_pat.match(key)
        if m:
            channel = m.group(1)
            segment_id = int(m.group(2))
            seg_dict.setdefault(segment_id, {})[channel] = npz[key]
    return {k: v for k, v in sorted(seg_dict.items())}


def frame_offsets(segs):
    offsets = {}
    running = 0
    for seg, data in segs.items():
        offsets[seg] = running
        length = next(iter(data.values())).shape[1]
        running += length
    return offsets


def total_frames_per_plane(segs):
    return sum(next(iter(ch.values())).shape[1] for ch in segs.values())


def plane_phase_in_scan(analysis_plane_idx_0based: int) -> int:
    """
    Map Analysis plane index 0,1,2 (after dropping flyback) to scan phases 1,2,3
    when flyback is phase 0 in a 4-phase cycle.
    """
    return (FLYBACK_PHASE + 1 + analysis_plane_idx_0based) % N_PLANES_SCAN


def map_valves_to_plane(valve_all_raw, n_planes_scan, plane_phase, frames_per_plane_total):
    """
    Map global TTL frame indices to the nearest per-plane frame index for a given plane phase.
    Keep original global positions to preserve open/close parity.
    """
    if isinstance(valve_all_raw, np.ndarray):
        valve_all_raw = valve_all_raw.tolist()
    raw = [int(v) for v in valve_all_raw]
    if not raw:
        return [], [], {"mode": "empty", "n_planes_scan": n_planes_scan}

    mapped = []
    order  = []
    for pos, g in enumerate(raw):
        # convert global frame g to nearest per-plane frame index k for this plane phase
        k_float = (g - plane_phase) / float(n_planes_scan)
        k = int(np.round(k_float))   # nearest, not floor
        mapped.append(k)
        order.append(pos)

    info = {
        "mode": "global_to_plane_nearest",
        "n_planes_scan": n_planes_scan,
        "plane_phase": plane_phase,
        "raw_events": len(raw),
        "mapped_min": int(np.min(mapped)),
        "mapped_max": int(np.max(mapped)),
        "frames_per_plane_total": int(frames_per_plane_total),
    }
    return mapped, order, info

def events_for_segment(valve_all_plane_idx, valve_global_pos, start_raw, seg_len):
    """
    Select all valve events that fall into this segment and keep their open/close parity.
    Parity is based on the original global ordering: even index is open, odd is close.
    """
    end_raw = start_raw + seg_len
    picked = [(gi, idx - start_raw)
              for gi, idx in zip(valve_global_pos, valve_all_plane_idx)
              if start_raw <= idx < end_raw]

    if not picked:
        return [], [], [], [], []

    # Sort by local index to plot in time order, but parity uses gi from global order
    picked.sort(key=lambda x: x[1])

    events_local = [li for _, li in picked]
    is_open = [(gi % 2 == 0) for gi, _ in picked]

    opens_local = [li for li, op in zip(events_local, is_open) if op]
    closes_local = [li for li, op in zip(events_local, is_open) if not op]

    # Pair opens to subsequent closes inside the segment
    pairs = []
    current_open = None
    for li, op in zip(events_local, is_open):
        if op:
            current_open = li
        else:
            if current_open is not None and li >= current_open:
                pairs.append((current_open, li))
                current_open = None
    return events_local, is_open, opens_local, closes_local, pairs


def apply_shift_to_events(events_local, is_open_flags, opens_local, closes_local, pairs, shift_frames, seg_len):
    """Shift all event positions by shift_frames and clip to [0, seg_len-1] for plotting."""

    def _shift_list(lst):
        if not lst:
            return []
        x = [v + shift_frames for v in lst]
        return [int(max(0, min(seg_len - 1, v))) for v in x]

    ev2 = _shift_list(events_local)
    op2 = list(is_open_flags)  # parity does not change
    opn2 = _shift_list(opens_local)
    cls2 = _shift_list(closes_local)
    pr2 = [(_shift_list([a])[0], _shift_list([b])[0]) for a, b in pairs] if pairs else []
    return ev2, op2, opn2, cls2, pr2


def build_valve_vector(seg_len, pairs):
    """Binary vector of length seg_len that is 1 during open-close intervals."""
    vec = np.zeros(seg_len, dtype=float)
    for a, b in pairs:
        a = int(max(0, min(seg_len - 1, a)))
        b = int(max(0, min(seg_len - 1, b)))
        if b >= a:
            vec[a:b + 1] = 1.0
    return vec


# ==================================================================
# 3) SIGNAL PROCESSING HELPERS
# ==================================================================

def double_exponential(t, const, amp_fast, amp_slow, tau_slow, tau_multiplier):
    tau_fast = tau_slow * tau_multiplier
    return const + amp_slow * np.exp(-t / tau_slow) + amp_fast * np.exp(-t / tau_fast)


def bleaching_correction(traces, rec_freq=3):
    all_signal_expfit = []
    for i in range(traces.shape[0]):
        cell = traces[i]
        time_seconds = np.arange(len(cell)) / rec_freq
        max_sig = np.percentile(cell, 99)
        initial_params = [max_sig / 2, max_sig / 4, max_sig / 4, 3600, 0.1]
        bounds = ([0, 0, 0, 600, 0], [max_sig, max_sig, max_sig, 36000, 1])
        try:
            signal_parms, _ = curve_fit(
                double_exponential, time_seconds, cell,
                p0=initial_params, bounds=bounds, maxfev=1000
            )
            signal_expfit = double_exponential(time_seconds, *signal_parms)
            corrected_cell = cell - signal_expfit
            all_signal_expfit.append(corrected_cell)
        except RuntimeError:
            print(f"Fitting failed for cell {i}, using original trace.")
            all_signal_expfit.append(cell)
    return np.array(all_signal_expfit)


def lowpass(tr, cutoff=0.05, order=3):
    # cutoff is in normalized cycles per sample where Nyquist is 0.5
    nyq = 0.5
    b, a = butter(order, cutoff / nyq, btype='low', analog=False)
    padlen = 3 * (max(len(a), len(b)) - 1)
    if tr.shape[1] <= padlen:
        return tr.astype(float, copy=True)
    padlen = min(padlen, tr.shape[1] - 1)
    return filtfilt(b, a, tr, axis=1, padlen=padlen)


def zscore(baseline, tr):
    mu = baseline.mean(axis=1, keepdims=True)
    std = baseline.std(axis=1, keepdims=True)
    std[std == 0] = 1.0
    return (tr - mu) / std


def running_mean(tr, win=5, align="center"):
    if win <= 1:
        return tr.copy()
    if align not in ("center", "leading", "trailing"):
        raise ValueError("align must be 'center', 'leading', or 'trailing'")
    if align == "center":
        left_pad = win // 2
        right_pad = win - 1 - left_pad
    elif align == "trailing":
        left_pad, right_pad = win - 1, 0
    else:
        left_pad, right_pad = 0, win - 1
    left = np.repeat(tr[:, :1], left_pad, axis=1)
    right = np.repeat(tr[:, -1:], right_pad, axis=1)
    x = np.concatenate([left, tr, right], axis=1)
    kernel = np.ones(win, dtype=float) / win
    return np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="valid"), axis=1, arr=x)


# ==================================================================
# 4) PLOTTING HELPERS
# ==================================================================

def _draw_all_events_as_lines(ax, events_all, is_open_flags, color_open='green', color_close='red'):
    for idx, op in zip(events_all, is_open_flags):
        ax.axvline(idx, color=(color_open if op else color_close), lw=1, alpha=0.9)


def plot_sorted_heatmap(data, name, seg_id, events_all, is_open_flags, valve_pairs, cmap):
    cg = sns.clustermap(
        data,
        row_cluster=True,
        col_cluster=False,
        method='average',
        cmap=cmap,
        xticklabels=False,
        yticklabels=False,
        figsize=(10, 8),
        cbar_kws={'label': 'Intensity (z-scored)'}
    )
    order = cg.dendrogram_row.reordered_ind
    plt.close(cg.fig)

    sorted_data = data[order, :]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        sorted_data,
        ax=ax,
        cmap=cmap,
        vmax=1.5, vmin=-1.5,
        xticklabels=False,
        yticklabels=False,
        cbar_kws={'label': 'Intensity (z-scored)'}
    )
    ax.invert_yaxis()

    # thick bars for matched open-close intervals
    for open_idx, close_idx in valve_pairs:
        ax.hlines(y=-0.5, xmin=open_idx, xmax=close_idx, colors='black', linewidth=5, alpha=0.7)
    # thin lines for all events (so unmatched events are still visible)
    _draw_all_events_as_lines(ax, events_all, is_open_flags)

    ax.set_ylim(-1, sorted_data.shape[0])
    ax.set_xlabel('Frame within segment')
    ax.set_ylabel('Cell (clustered)')
    ax.set_title(f'{name} heatmap, segment {seg_id}')
    plt.tight_layout()
    return fig


def plot_segment(seg_id, traces, events_all, is_open, channel, out_dir):
    traces = lowpass(traces)
    n_cells, _ = traces.shape
    seg_dir = out_dir / f'segment_{seg_id:02d}'
    seg_dir.mkdir(exist_ok=True)

    for cell in range(n_cells):
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(traces[cell], lw=0.8)
        ax.set_xlabel('Frame (within segment)')
        ax.set_ylabel('Intensity')
        ax.set_title(f'{channel.capitalize()}   cell {cell}   segment {seg_id}')
        _draw_all_events_as_lines(ax, events_all, is_open)
        plt.tight_layout()
        fname = seg_dir / f'{channel}_cell{cell:03d}.png'
        plt.savefig(fname, dpi=150)
        plt.close(fig)
        print(f'  saved {fname.relative_to(out_dir.parent)}')


# ==================================================================
# 5) RESPONDER QUANTIFICATION AND PLOTS
# ==================================================================

def _preprocess_green_with_ctrl_for_detection(green_tr, ctrl_tr, rec_freq=3, last_stimulus_idx=None, return_processed=False):
    cutoff = 0.02
    green_lp = lowpass(green_tr, cutoff=cutoff)
    ctrl_lp = lowpass(ctrl_tr, cutoff=cutoff)
    green_bleach = bleaching_correction(green_lp, rec_freq=rec_freq)
    ctrl_bleach = bleaching_correction(ctrl_lp, rec_freq=rec_freq)

    slope = None
    intercept = None

    if last_stimulus_idx is not None and last_stimulus_idx < green_bleach.shape[1]:
        green_fit = green_bleach[:, :last_stimulus_idx]
        ctrl_fit = ctrl_bleach[:, :last_stimulus_idx]
        print(f"    Using data until frame {last_stimulus_idx} for motion correction fit")
    else:
        green_fit = green_bleach
        ctrl_fit = ctrl_bleach

    mask = ~np.isnan(ctrl_fit) & ~np.isnan(green_fit)
    ctrl_clean = ctrl_fit[mask]
    green_clean = green_fit[mask]
    if ctrl_clean.size > 0 and green_clean.size > 0:
        slope, intercept, _, _, _ = linregress(x=ctrl_clean, y=green_clean)
        est_motion = intercept + slope * ctrl_bleach
        green_corrected = green_bleach - est_motion
    else:
        green_corrected = green_bleach

    green_denoised = running_mean(green_corrected, win=10*rec_freq, align="center")
    ctrl_denoised = running_mean(ctrl_bleach, win=10*rec_freq, align="center")

    if return_processed:
        return {
            "green_lowpass": green_lp,
            "ctrl_lowpass": ctrl_lp,
            "green_bleached": green_bleach,
            "ctrl_bleached": ctrl_bleach,
            "green_corrected": green_corrected,
            "green_denoised": green_denoised,
            "ctrl_denoised": ctrl_denoised,
            "motion_fit": {
                "slope": slope,
                "intercept": intercept,
                "last_stimulus_idx": last_stimulus_idx,
            }
        }

    return green_denoised


def calculate_correlation_red_vs_corrected_green(green_tr, ctrl_tr, opens_local, closes_local, rec_freq=3, return_preprocessed=False):
    """
    Calculate correlation between red channel and motion-corrected green channel.
    Only analyzes data until the last stimulus to avoid post-stimulus artifacts.
    Optionally returns the processed traces used for the calculation so that plots
    can reflect exactly what drove the classification logic.
    """
    if green_tr.size == 0 or ctrl_tr.size == 0:
        correlations = np.zeros(green_tr.shape[0] if green_tr.size else 0)
        if return_preprocessed:
            return correlations, {
                "last_stim_idx": None,
                "preprocessed": None,
            }
        return correlations

    correlations = np.zeros(green_tr.shape[0])

    last_stim_idx = green_tr.shape[1]
    all_stim_times = list(opens_local[:-1])
    if all_stim_times:
        last_stim_idx = int(max(all_stim_times)) + 1
        print(f"  Truncating correlation Analysis at frame {last_stim_idx} (last stimulus)")

    preprocess = _preprocess_green_with_ctrl_for_detection(
        green_tr,
        ctrl_tr,
        rec_freq=rec_freq,
        last_stimulus_idx=last_stim_idx,
        return_processed=True,
    )

    green_processed = preprocess["green_denoised"]
    ctrl_processed = preprocess["ctrl_denoised"]

    baseline_limit = int(240 * rec_freq)

    for cell in range(green_processed.shape[0]):
        green_final = green_processed[cell]
        ctrl_final = ctrl_processed[cell] if cell < ctrl_processed.shape[0] else np.zeros_like(green_final)

        green_trunc = green_final[:last_stim_idx] if last_stim_idx < len(green_final) else green_final
        ctrl_trunc = ctrl_final[:last_stim_idx] if last_stim_idx < len(ctrl_final) else ctrl_final

        if len(green_trunc) <= 10 or len(ctrl_trunc) <= 10:
            correlations[cell] = 0.0
            continue

        baseline_frames = min(baseline_limit, max(1, len(green_trunc) // 4))
        green_baseline = green_trunc[:baseline_frames]
        ctrl_baseline = ctrl_trunc[:baseline_frames]

        green_mean = np.mean(green_baseline)
        green_std = np.std(green_baseline)
        ctrl_mean = np.mean(ctrl_baseline)
        ctrl_std = np.std(ctrl_baseline)

        if green_std > 0:
            green_norm = (green_trunc - green_mean) / green_std
        else:
            green_norm = green_trunc - green_mean

        if ctrl_std > 0:
            ctrl_norm = (ctrl_trunc - ctrl_mean) / ctrl_std
        else:
            ctrl_norm = ctrl_trunc - ctrl_mean

        final_mask = (
            ~np.isnan(green_norm)
            & ~np.isnan(ctrl_norm)
            & np.isfinite(green_norm)
            & np.isfinite(ctrl_norm)
        )

        if np.sum(final_mask) > 50:
            try:
                corr_matrix = np.corrcoef(ctrl_norm[final_mask], green_norm[final_mask])
                if corr_matrix.shape == (2, 2) and not np.isnan(corr_matrix[0, 1]):
                    correlations[cell] = corr_matrix[0, 1]
                else:
                    correlations[cell] = 0.0
            except Exception:
                correlations[cell] = 0.0
        else:
            correlations[cell] = 0.0

    if return_preprocessed:
        return correlations, {
            "last_stim_idx": last_stim_idx,
            "preprocessed": preprocess,
        }

    return correlations


def detect_single_channel_responses(
        traces,
        opens_local,
        closes_local,
        rec_freq=3,
        baseline_sec=240.0,
        post_sec=40.0,
        z_thresh=2.0,
        min_frames=75,
        min_resp_trials=1,
        pre_window_sec=30.0,
        max_baseline_slope=1,
        return_processed=False,
):
    """
    Detect responses in a single channel (green or red) without motion correction.
    Returns responsive cells and which stimuli each cell responds to.
    Optionally returns the processed traces that were evaluated for classification.
    """
    if traces.size == 0:
        if return_processed:
            return np.zeros((0,), dtype=bool), {}, None
        return np.zeros((0,), dtype=bool), {}

    # Apply basic preprocessing without motion correction
    cutoff = 0.02
    traces_lp = lowpass(traces, cutoff=cutoff)
    traces_bleach = bleaching_correction(traces_lp, rec_freq=rec_freq)
    traces_smooth = running_mean(traces_bleach, win=10*rec_freq, align="center")

    n_base = int(rec_freq * baseline_sec)
    if len(opens_local) + len(closes_local) > 0:
        first_idx = min(opens_local[0] if len(opens_local) else np.inf,
                        closes_local[0] if len(closes_local) else np.inf)
        if first_idx == np.inf:
            first_idx = traces_smooth.shape[1]
    else:
        first_idx = traces_smooth.shape[1]
    
    start_idx = max(0, first_idx - n_base)
    end_idx = min(first_idx, traces_smooth.shape[1])
    traces_norm = zscore(traces_smooth[:, start_idx:end_idx], traces_smooth)

    opens = list(opens_local[:-1])
    closes = list(closes_local)
    n_pairs = min(len(opens), len(closes))

    T = traces_norm.shape[1]
    post_frames = int(post_sec * rec_freq)
    n_cells = traces_norm.shape[0]
    responsive = np.zeros(n_cells, dtype=bool)
    pre_frames_dyn = int(max(1, round(pre_window_sec * rec_freq)))
    
    # Track which stimuli each cell responds to
    cell_stimulus_responses = {}

    for cell in range(n_cells):
        baseline_segment = traces_norm[cell, start_idx:end_idx]
        if baseline_segment.size < 2:
            continue
        baseline_time = np.arange(len(baseline_segment)) / rec_freq
        slope, _, _, _, _ = linregress(baseline_time, baseline_segment)
        if round(slope, 2) > max_baseline_slope:
            continue

        resp_trials = 0
        responding_stimuli = []
        
        for k in range(n_pairs):
            op = opens[k]
            cl = closes[k]
            win_end = min(cl + post_frames, T)
            if op >= win_end:
                continue

            pre_start = max(0, op - pre_frames_dyn)
            if pre_start >= op:
                continue

            pre_seg = traces_norm[cell, pre_start:op]
            pre_mean = np.mean(pre_seg)
            dynamic_threshold = pre_mean + z_thresh

            stim_seg = traces_norm[cell, op:cl + post_frames]
            if stim_seg.size == 0:
                continue

            stim_above = stim_seg > dynamic_threshold
            if stim_above.size == 0:
                stim_frames = 0
            else:
                diff = np.diff(np.concatenate(([False], stim_above, [False])).astype(int))
                starts = np.where(diff == 1)[0]
                ends = np.where(diff == -1)[0]
                if len(starts) > len(ends):
                    ends = np.append(ends, len(stim_above))
                lengths = ends - starts
                stim_frames = np.max(lengths) if lengths.size > 0 else 0

            if stim_frames >= min_frames:
                resp_trials += 1
                responding_stimuli.append(k)

        cell_stimulus_responses[cell] = responding_stimuli
        if resp_trials >= min_resp_trials:
            responsive[cell] = True

    if not return_processed:
        return responsive, cell_stimulus_responses

    processed = {
        "traces_lowpass": traces_lp,
        "traces_bleach": traces_bleach,
        "traces_smooth": traces_smooth,
        "traces_norm": traces_norm,
        "start_idx": start_idx,
        "end_idx": end_idx,
    }
    return responsive, cell_stimulus_responses, processed


def detect_responders_green_ctrl_original_logic(
        green_tr,
        ctrl_tr,
        opens_local,
        closes_local,
        rec_freq=3,
        baseline_sec=240.0,
        post_sec=40.0,
        z_thresh=2.0,
        min_frames=5 * 15,
        min_resp_trials=1,
        pre_window_sec=30.0,
        max_baseline_slope=1,
        return_processed=False,
):
    """
    Original logic for detecting responders - kept exactly the same for low correlation cells.
    Optionally returns the processed traces used for classification.
    """
    if green_tr.size == 0:
        if return_processed:
            return np.zeros((0,), dtype=bool), None
        return np.zeros((0,), dtype=bool)

    preprocess = _preprocess_green_with_ctrl_for_detection(
        green_tr,
        ctrl_tr,
        rec_freq=rec_freq,
        return_processed=return_processed,
    )

    if return_processed and isinstance(preprocess, dict):
        g_proc = preprocess["green_denoised"]
    else:
        g_proc = preprocess

    n_base = int(rec_freq * baseline_sec)
    if len(opens_local) + len(closes_local) > 0:
        first_idx = min(opens_local[0] if len(opens_local) else np.inf,
                        closes_local[0] if len(closes_local) else np.inf)
        if first_idx == np.inf:
            first_idx = g_proc.shape[1]
    else:
        first_idx = g_proc.shape[1]
    start_idx = max(0, first_idx - n_base)
    end_idx = min(first_idx, g_proc.shape[1])
    g_norm = zscore(g_proc[:, start_idx:end_idx], g_proc)

    opens = list(opens_local[:-1])
    closes = list(closes_local)
    n_pairs = min(len(opens), len(closes))

    T = g_norm.shape[1]
    post_frames = int(post_sec * rec_freq)
    n_cells = g_norm.shape[0]
    responsive = np.zeros(n_cells, dtype=bool)
    pre_frames_dyn = int(max(1, round(pre_window_sec * rec_freq)))

    for cell in range(n_cells):
        baseline_segment = g_norm[cell, start_idx:end_idx]
        if baseline_segment.size < 2:
            continue
        baseline_time = np.arange(len(baseline_segment)) / rec_freq
        slope, _, _, _, _ = linregress(baseline_time, baseline_segment)
        if round(slope, 2) > max_baseline_slope:
            continue

        resp_trials = 0
        for k in range(n_pairs):
            op = opens[k]
            cl = closes[k]
            win_end = min(cl + post_frames, T)
            if op >= win_end:
                continue

            pre_start = max(0, op - pre_frames_dyn)
            if pre_start >= op:
                continue

            pre_seg = g_norm[cell, pre_start:op]
            pre_mean = np.mean(pre_seg)
            dynamic_threshold = pre_mean + z_thresh

            stim_seg = g_norm[cell, op:cl + post_frames]
            if stim_seg.size == 0:
                continue

            stim_above = stim_seg > dynamic_threshold
            if stim_above.size == 0:
                stim_frames = 0
            else:
                diff = np.diff(np.concatenate(([False], stim_above, [False])).astype(int))
                starts = np.where(diff == 1)[0]
                ends = np.where(diff == -1)[0]
                if len(starts) > len(ends):
                    ends = np.append(ends, len(stim_above))
                lengths = ends - starts
                stim_frames = np.max(lengths) if lengths.size > 0 else 0

            if stim_frames >= min_frames:
                resp_trials += 1

        if resp_trials >= min_resp_trials:
            responsive[cell] = True

    if not return_processed:
        return responsive

    processed = {
        "g_proc": g_proc,
        "g_norm": g_norm,
        "start_idx": start_idx,
        "end_idx": end_idx,
    }
    if return_processed and isinstance(preprocess, dict):
        processed.update({
            "motion_fit": preprocess.get("motion_fit"),
            "green_lowpass": preprocess.get("green_lowpass"),
            "ctrl_lowpass": preprocess.get("ctrl_lowpass"),
            "green_bleached": preprocess.get("green_bleached"),
            "ctrl_bleached": preprocess.get("ctrl_bleached"),
            "ctrl_denoised": preprocess.get("ctrl_denoised"),
        })

    return responsive, processed


def detect_responders_green_ctrl(
        green_tr,
        ctrl_tr,
        opens_local,
        closes_local,
        rec_freq=3,
        baseline_sec=240.0,
        post_sec=40.0,
        z_thresh=2.0,
        min_frames=5 * 15,
        min_resp_trials=1,
        pre_window_sec=30.0,
        max_baseline_slope=1,
        correlation_threshold=0.5,  # threshold for high correlation
):
    """
    Detect responder cells with new correlation-based criteria:
    1. High negative correlation: use green without red subtraction
    2. High positive correlation: test both channels, require green-only responses
    3. Low correlation: use original method with motion correction
    """
    if green_tr.size == 0:
        return np.zeros((0,), dtype=bool), {
            "correlations": np.zeros((0,)),
            "last_stim_idx": None,
            "preprocess_data": None,
        }

    correlations, corr_info = calculate_correlation_red_vs_corrected_green(
        green_tr,
        ctrl_tr,
        opens_local,
        closes_local,
        rec_freq=rec_freq,
        return_preprocessed=True,
    )

    last_stim_idx = None
    preprocess_data = None
    if isinstance(corr_info, dict):
        last_stim_idx = corr_info.get("last_stim_idx")
        preprocess_data = corr_info.get("preprocessed")

    n_cells = green_tr.shape[0]
    responsive = np.zeros(n_cells, dtype=bool)
    diagnostics = {
        "correlations": correlations,
        "last_stim_idx": last_stim_idx,
        "preprocess_data": preprocess_data,
        "high_neg": {},
        "high_pos": {},
        "low_corr": {},
    }

    # Separate cells by correlation
    high_neg_corr_cells = []
    high_pos_corr_cells = []
    low_corr_cells = []
    
    for cell in range(n_cells):
        corr = correlations[cell] if not np.isnan(correlations[cell]) else 0.0
        
        if corr < -correlation_threshold:
            high_neg_corr_cells.append(cell)
            print(f"Cell {cell}: High negative correlation between red and corrected green ({corr:.3f}) - over-correction, using green without red subtraction")
        elif corr > correlation_threshold:
            high_pos_corr_cells.append(cell)
            print(f"Cell {cell}: High positive correlation between red and corrected green ({corr:.3f}) - still contaminated, testing both channels")
        else:
            low_corr_cells.append(cell)
            print(f"Cell {cell}: Low correlation between red and corrected green ({corr:.3f}) - good correction, using original method")
    
    # Process high negative correlation cells
    for cell in high_neg_corr_cells:
        green_single = green_tr[cell:cell+1, :]
        resp_single, _, proc = detect_single_channel_responses(
            green_single,
            opens_local,
            closes_local,
            rec_freq=rec_freq,
            baseline_sec=baseline_sec,
            post_sec=post_sec,
            z_thresh=z_thresh,
            min_frames=min_frames,
            min_resp_trials=min_resp_trials,
            pre_window_sec=pre_window_sec,
            max_baseline_slope=max_baseline_slope,
            return_processed=True,
        )
        responsive[cell] = resp_single[0] if len(resp_single) > 0 else False
        diagnostics["high_neg"][cell] = proc

    diagnostics["high_pos"] = {}
    for cell in high_pos_corr_cells:
        green_single = green_tr[cell:cell+1, :]
        ctrl_single = ctrl_tr[cell:cell+1, :] if cell < ctrl_tr.shape[0] else np.zeros_like(green_single)

        resp_green, stimuli_green, proc_green = detect_single_channel_responses(
            green_single,
            opens_local,
            closes_local,
            rec_freq=rec_freq,
            baseline_sec=baseline_sec,
            post_sec=post_sec,
            z_thresh=z_thresh,
            min_frames=min_frames,
            min_resp_trials=min_resp_trials,
            pre_window_sec=pre_window_sec,
            max_baseline_slope=max_baseline_slope,
            return_processed=True,
        )

        resp_red, stimuli_red, proc_red = detect_single_channel_responses(
            ctrl_single,
            opens_local,
            closes_local,
            rec_freq=rec_freq,
            baseline_sec=baseline_sec,
            post_sec=post_sec,
            z_thresh=z_thresh,
            min_frames=min_frames,
            min_resp_trials=min_resp_trials,
            pre_window_sec=pre_window_sec,
            max_baseline_slope=max_baseline_slope,
            return_processed=True,
        )

        diagnostics["high_pos"][cell] = {
            "green": proc_green,
            "red": proc_red,
            "stimuli_green": stimuli_green,
            "stimuli_red": stimuli_red,
        }
        
        # Check if green responds but there's at least one stimulus where only green responds
        if resp_green[0]:
            green_stimuli = set(stimuli_green.get(0, []))
            red_stimuli = set(stimuli_red.get(0, []))
            green_only_stimuli = green_stimuli - red_stimuli
            
            if len(green_only_stimuli) > 0:
                print(f"  Green responds to stimuli {green_stimuli}, red to {red_stimuli}")
                print(f"  Green-only stimuli: {green_only_stimuli} - SIGNIFICANT")
                responsive[cell] = True
            else:
                print(f"  Both channels respond to same stimuli - NOT SIGNIFICANT")
                responsive[cell] = False
        else:
            responsive[cell] = False
    
    # Process low correlation cells using EXACT original logic
    if low_corr_cells:
        print(f"Processing {len(low_corr_cells)} low correlation cells with original logic")
        low_corr_indices = np.array(low_corr_cells)
        green_low_corr = green_tr[low_corr_indices, :]
        ctrl_low_corr = ctrl_tr[low_corr_indices, :] if ctrl_tr.size > 0 else np.zeros_like(green_low_corr)

        resp_low_corr, proc_low = detect_responders_green_ctrl_original_logic(
            green_low_corr,
            ctrl_low_corr,
            opens_local,
            closes_local,
            rec_freq=rec_freq,
            baseline_sec=baseline_sec,
            post_sec=post_sec,
            z_thresh=z_thresh,
            min_frames=min_frames,
            min_resp_trials=min_resp_trials,
            pre_window_sec=pre_window_sec,
            max_baseline_slope=max_baseline_slope,
            return_processed=True,
        )

        for i, cell in enumerate(low_corr_cells):
            responsive[cell] = resp_low_corr[i]
            if isinstance(proc_low, dict):
                diagnostics["low_corr"][cell] = {
                    "g_proc": proc_low.get("g_proc")[i] if proc_low.get("g_proc") is not None else None,
                    "g_norm": proc_low.get("g_norm")[i] if proc_low.get("g_norm") is not None else None,
                    "start_idx": proc_low.get("start_idx"),
                    "end_idx": proc_low.get("end_idx"),
                }

    diagnostics["responsive"] = responsive
    return responsive, diagnostics


def estimate_alignment_offset(ctrl_tr, pairs, seg_len, rec_freq, max_align_sec):
    """
    Estimate small integer shift that best aligns valve bars to the red control mean.
    Positive shift moves events to the right.
    """
    if ctrl_tr.size == 0 or len(pairs) == 0:
        return 0
    ctrl_lp = lowpass(ctrl_tr, cutoff=0.05)
    ctrl_mean = np.nanmean(ctrl_lp, axis=0)
    if ctrl_mean.size != seg_len:
        ctrl_mean = np.resize(ctrl_mean, seg_len)
    # zscore-like normalizing
    cm = ctrl_mean - np.mean(ctrl_mean)
    cm_std = np.std(cm)
    if cm_std > 0:
        cm /= cm_std

    vec = build_valve_vector(seg_len, pairs)
    vv = vec - np.mean(vec)
    vv_std = np.std(vv)
    if vv_std > 0:
        vv /= vv_std

    max_shift = int(round(max_align_sec * rec_freq))
    if max_shift < 1:
        return 0

    best_k = 0
    best_corr = -1e9
    for k in range(-max_shift, max_shift + 1):
        if k < 0:
            vv_shift = np.pad(vv[-k:], (0, -k), mode='constant')
        else:
            vv_shift = np.pad(vv[:-k] if k > 0 else vv, (k, 0), mode='constant')
        c = float(np.dot(cm, vv_shift))
        if c > best_corr:
            best_corr = c
            best_k = k
    return int(best_k)


# ==================================================================
# 6) NPZ KEY HELPERS
# ==================================================================

def plane_key(npz, base, seg_id, analysis_plane_idx):
    """
    Return the NPZ key for channel `base` at segment seg_id for the given Analysis-plane index.
    Try both plane{idx+1} and plane{idx} to tolerate 1-based or 0-based storage.
    """
    k1 = f"{base}_seg{seg_id}_plane{analysis_plane_idx + 1}_traces"
    k0 = f"{base}_seg{seg_id}_plane{analysis_plane_idx}_traces"
    if k1 in npz:
        return k1
    if k0 in npz:
        return k0
    return None


def overlap_key(npz, seg_id, analysis_plane_idx):
    """
    Try common patterns for overlap indices.
    """
    cands = [
        f"overlap_green_seg{seg_id - 1}_plane{analysis_plane_idx}",
        f"overlap_green_seg{seg_id}_plane{analysis_plane_idx}",
        f"overlap_green_seg{seg_id - 1}_plane{analysis_plane_idx + 1}",
        f"overlap_green_seg{seg_id}_plane{analysis_plane_idx + 1}",
    ]
    for k in cands:
        if k in npz:
            return k
    return None


def dedup_key(npz, seg_id, analysis_plane_idx):
    cands = [
        f"dedup_drop_green_seg{seg_id - 1}_plane{analysis_plane_idx}",
        f"dedup_drop_green_seg{seg_id}__plane{analysis_plane_idx}",
        f"dedup_drop_green_seg{seg_id - 1}_plane{analysis_plane_idx + 1}",
        f"dedup_drop_green_seg{seg_id}_plane{analysis_plane_idx + 1}",
    ]
    for k in cands:
        if k in npz:
            return k
    return None


# ==================================================================
# 7) MAIN EXECUTION
# ==================================================================

# Load NPZ and valve indices
npz = np.load(ROI_NPZ_PATH, allow_pickle=True)
segs = collect_segments(npz)
offsets = frame_offsets(segs)
valve_all_raw = np.loadtxt(VALVE_TXT_PATH, dtype=int).tolist()

# Analysis planes (0-based across the analyzed planes only, flyback excluded)
planes_to_process = determine_planes_to_process(npz, PLANES_TO_PROCESS)
print(f"Analysis planes (0-based): {planes_to_process}")

frames_total_per_plane = total_frames_per_plane(segs)
print(f"Scan cycle length (including flyback): {N_PLANES_SCAN}")
print(f"Total frames per-plane timeline: {frames_total_per_plane}")

# Loop over Analysis planes and segments
for analysis_plane_idx in planes_to_process:
    print(f"\n{'=' * 60}")
    print(f"PROCESSING ANALYSIS PLANE {analysis_plane_idx} (scan phase {plane_phase_in_scan(analysis_plane_idx)})")
    print(f"{'=' * 60}")

    OUT_DIR = ROI_NPZ_PATH.parent / f'plots_plane{analysis_plane_idx + 1}_zscore_by_baseline_motion_corrected_v9'
    OUT_DIR.mkdir(exist_ok=True)

    csv_path = OUT_DIR / 'responding_cells_summary.csv'
    if csv_path.exists():
        csv_path.unlink()

    # Map valves for this plane using phase-aware deinterleave over the full scan cycle
    plane_phase = plane_phase_in_scan(analysis_plane_idx)
    phase = plane_phase_in_scan(analysis_plane_idx)
    valve_all_plane_idx, valve_global_pos, map_info = map_valves_to_plane(
        valve_all_raw, N_PLANES_SCAN, phase, frames_total_per_plane
    )
    print("Valve mapping info:", map_info)

    # Optional quick residue check to verify mapping
    raw_arr = np.asarray(valve_all_raw, dtype=int)
    if len(raw_arr):
        resid = (raw_arr - plane_phase) % N_PLANES_SCAN
        counts = {r: int((resid == r).sum()) for r in range(N_PLANES_SCAN)}
        print("Residue counts against this phase:", counts)

    for seg_id in sorted(segs):
        ch_dict = segs[seg_id]
        seg_len = next(iter(ch_dict.values())).shape[1]

        start_raw = offsets[seg_id]
        end_raw = start_raw + seg_len
        print(f'\nSegment {seg_id}: raw frames {start_raw} - {end_raw - 1}')

        # Events inside this segment before any alignment or manual shift
        valve_seg_all, valve_seg_is_open, opens_local, closes_local, valve_pairs = events_for_segment(
            valve_all_plane_idx, valve_global_pos, start_raw, seg_len
        )
        print(
            f'  events found in segment: {len(valve_seg_all)}  opens: {len(opens_local)}  closes: {len(closes_local)}  pairs: {len(valve_pairs)}')

        # Optional micro alignment to red control
        align_k = 0
        if AUTO_ALIGN_BY_CTRL:
            # Try to fetch ctrl traces for alignment
            k_ctrl = plane_key(npz, "red_ctrl", seg_id, analysis_plane_idx)
            ctrl_for_align = npz[k_ctrl] if k_ctrl else ch_dict.get('red_ctrl', np.empty((0, seg_len)))
            if ctrl_for_align.size:
                try:
                    align_k = estimate_alignment_offset(ctrl_for_align, valve_pairs, seg_len, rec_freq, MAX_ALIGN_SEC)
                    print(f"  auto-alignment estimated shift: {align_k} frames")
                except Exception as e:
                    print(f"  alignment estimation failed: {e}")
                    align_k = 0

        total_shift = int(MANUAL_SHIFT_FRAMES + align_k)
        if total_shift != 0:
            valve_seg_all, valve_seg_is_open, opens_local, closes_local, valve_pairs = apply_shift_to_events(
                valve_seg_all, valve_seg_is_open, opens_local, closes_local, valve_pairs, total_shift, seg_len
            )
            print(f"  applied event shift: {total_shift} frames (manual: {MANUAL_SHIFT_FRAMES}, auto: {align_k})")

        # Load traces for this plane and segment
        # Overlap rows
        ov_k = overlap_key(npz, seg_id, analysis_plane_idx)
        overlap_rows_global = npz[ov_k] if ov_k in npz else (npz[ov_k] if ov_k else np.array([], dtype=int))

        # Main channels
        key_green = plane_key(npz, "green", seg_id, analysis_plane_idx)
        key_ctrl = plane_key(npz, "red_ctrl", seg_id, analysis_plane_idx)
        if key_green in npz and key_ctrl in npz:
            green_all = npz[key_green]
            ctrl_all = npz[key_ctrl]
        else:
            # Fallback to pre-collected dictionary if keys are absent
            green_all = ch_dict.get('green', np.empty((0, seg_len)))
            ctrl_all = ch_dict.get('red_ctrl', np.empty((0, seg_len)))

        # Dedup across planes if present
        dd_k = dedup_key(npz, seg_id, analysis_plane_idx)
        if dd_k and dd_k in npz:
            dedup_rows_global = np.asarray(npz[dd_k], dtype=int)
        else:
            dedup_rows_global = np.array([], dtype=int)

        if dedup_rows_global.size:
            n_before = green_all.shape[0]
            keep_mask = np.ones(n_before, dtype=bool)
            valid_dedup_rows = dedup_rows_global[dedup_rows_global < n_before]
            if valid_dedup_rows.size < dedup_rows_global.size:
                oob = dedup_rows_global[dedup_rows_global >= n_before]
                print(f"  Warning: {oob.size} dedup indices out of bounds and ignored: {oob}")
            keep_mask[valid_dedup_rows] = False
            green_all = green_all[keep_mask]
            ctrl_all = ctrl_all[keep_mask]

        # Overlap subset if available
        if isinstance(overlap_rows_global, np.ndarray):
            valid_overlap_mask = overlap_rows_global < green_all.shape[0]
            overlap_rows = overlap_rows_global[valid_overlap_mask]
        else:
            overlap_rows = np.array([], dtype=int)

        if overlap_rows.size > 0:
            green_olap = green_all[overlap_rows]
            ctrl_olap = ctrl_all[overlap_rows]
        else:
            green_olap = np.empty((0, seg_len))
            ctrl_olap = np.empty((0, seg_len))

        # Plot and quantify overlap cells if any
        if green_olap.size:
            print("  plotting green-with-red cells only. count:", green_olap.shape[0])

            def _plot_green_with_ctrl(
                    seg_id,
                    green_tr, ctrl_tr,
                    events_all, is_open_flags, opens_local, closes_local, valve_pairs,
                    out_dir,
                    csv_path, analysis_plane_idx, total_shift, map_info
            ):
                responders, diagnostics = detect_responders_green_ctrl(
                    green_tr, ctrl_tr, opens_local, closes_local,
                    rec_freq=rec_freq, baseline_sec=240.0, post_sec=50.0,
                    z_thresh=5.0, min_frames=5 * rec_freq, min_resp_trials=1,
                    pre_window_sec=20.0, max_baseline_slope=0.02
                )

                correlations = diagnostics.get("correlations", np.zeros(green_tr.shape[0]))
                preprocess = diagnostics.get("preprocess_data")
                last_stim_idx = diagnostics.get("last_stim_idx")

                # Write one row per cell
                n_pairs = min(len(opens_local), len(closes_local))
                rows = [
                    [
                        analysis_plane_idx, seg_id, int(i),
                        int(bool(responders[i])) if responders.size else 0,
                        len(opens_local), len(closes_local), n_pairs,
                        int(total_shift), str(map_info.get("mode", ""))
                    ]
                    for i in range(green_tr.shape[0])
                ]
                append_rows_to_csv(csv_path, rows)

                seg_dir = out_dir / f'segment_{seg_id:02d}'
                seg_dir.mkdir(exist_ok=True)

                if preprocess is not None:
                    g_den = preprocess.get("green_denoised", green_tr.astype(float))
                    r_den = preprocess.get("ctrl_denoised")
                    if r_den is None or r_den.size == 0:
                        r_den = np.zeros_like(g_den)
                else:
                    cutoff = 0.02
                    g_lp = lowpass(green_tr, cutoff=cutoff)
                    r_lp = lowpass(ctrl_tr, cutoff=cutoff) if ctrl_tr.size else np.zeros_like(green_tr)
                    g_den = running_mean(bleaching_correction(g_lp, rec_freq=rec_freq), win=10*rec_mean, align="center")
                    r_den = running_mean(bleaching_correction(r_lp, rec_freq=rec_freq), win=10*rec_mean, align="center")

                first_idx = events_all[0] if len(events_all) > 0 else g_den.shape[1]
                start_idx = max(0, first_idx - int(rec_freq * 240.0))
                end_idx = min(first_idx, g_den.shape[1])

                g_norm = zscore(g_den[:, start_idx:end_idx], g_den)
                if r_den.size:
                    r_norm = zscore(r_den[:, start_idx:end_idx], r_den)
                else:
                    r_norm = np.zeros_like(g_norm)

                if g_norm.shape[0] >= 2:
                    cg = sns.clustermap(
                        g_norm, row_cluster=True, col_cluster=False,
                        method='average', metric='euclidean',
                        cmap='Greens', xticklabels=False, yticklabels=False, figsize=(2, 2)
                    )
                    order = cg.dendrogram_row.reordered_ind
                    plt.close(cg.fig)
                else:
                    order = np.arange(g_norm.shape[0])

                g_sorted = g_norm[order[::-1], :]
                r_sorted = r_norm[order[::-1], :]

                fig = plot_sorted_heatmap(g_sorted, 'green', seg_id, events_all, is_open_flags, valve_pairs,
                                          cmap='coolwarm')
                fname = seg_dir / f'green_sharedorder_seg{seg_id}.png'
                fig.savefig(fname, dpi=150)
                plt.close(fig)
                print(f'  saved {fname.relative_to(out_dir.parent)}')

                fig = plot_sorted_heatmap(r_sorted, 'ctrl', seg_id, events_all, is_open_flags, valve_pairs,
                                          cmap='coolwarm')
                fname = seg_dir / f'ctrl_sharedorder_seg{seg_id}.png'
                fig.savefig(fname, dpi=150)
                plt.close(fig)
                print(f'  saved {fname.relative_to(out_dir.parent)}')

                high_neg_data = diagnostics.get("high_neg", {})
                high_pos_data = diagnostics.get("high_pos", {})
                low_corr_data = diagnostics.get("low_corr", {})

                correlation_threshold = 0.5

                for cell in range(green_tr.shape[0]):
                    fig, ax = plt.subplots(figsize=(9, 4))
                    is_sig = bool(responders[cell]) if responders.size else False
                    status = 'SIGNIFICANT' if is_sig else 'n.s.'
                    corr = correlations[cell] if not np.isnan(correlations[cell]) else 0.0

                    if cell in high_neg_data and high_neg_data[cell] is not None:
                        proc = high_neg_data[cell]
                        trace_green = proc.get("traces_norm")
                        if trace_green is not None:
                            trace_green = np.asarray(trace_green)[0]
                        else:
                            trace_green = g_norm[cell]
                        ax.plot(trace_green, lw=1.2, color='green', label='green (single-channel)')
                        title = f'Green-only cell {cell} segment {seg_id} - {status} (high neg corr: {corr:.3f})'

                    elif cell in high_pos_data and high_pos_data[cell] is not None:
                        proc = high_pos_data[cell]
                        trace_green = proc.get("green", {}).get("traces_norm") if proc.get("green") else None
                        trace_red = proc.get("red", {}).get("traces_norm") if proc.get("red") else None
                        if trace_green is not None:
                            trace_green = np.asarray(trace_green)[0]
                        else:
                            trace_green = g_norm[cell]
                        if trace_red is not None:
                            trace_red = np.asarray(trace_red)[0]
                        else:
                            trace_red = r_norm[cell]
                        ax.plot(trace_green, lw=1.0, color='green', label='green (single-channel)')
                        ax.plot(trace_red, lw=0.8, color='red', label='red (single-channel)')
                        title = f'Both channels cell {cell} segment {seg_id} - {status} (high pos corr: {corr:.3f})'

                    elif cell in low_corr_data and low_corr_data[cell] is not None:
                        proc = low_corr_data[cell]
                        trace_green = proc.get("g_norm")
                        if trace_green is None:
                            trace_green = g_norm[cell]
                        ax.plot(trace_green, lw=1.0, color='green', label='green (motion-corrected)')
                        if r_norm.size:
                            ax.plot(r_norm[cell], lw=0.8, color='gray', label='red ctrl')
                        title = f'Motion-corrected cell {cell} segment {seg_id} - {status} (low corr: {corr:.3f})'

                    else:
                        ax.plot(g_norm[cell], lw=1.0, color='green', label='green (motion-corrected)')
                        if r_norm.size:
                            ax.plot(r_norm[cell], lw=0.8, color='gray', label='red ctrl')
                        title = f'Motion-corrected cell {cell} segment {seg_id} - {status} (corr: {corr:.3f})'

                    _draw_all_events_as_lines(ax, events_all, is_open_flags)
                    ax.set_xlabel('Frame (within segment)')
                    ax.set_ylabel('Intensity (z-scored)')
                    ax.set_title(title)
                    ax.legend(frameon=False)
                    plt.tight_layout()
                    fname = seg_dir / f'green_cell{cell:03d}_overlap.png'
                    plt.savefig(fname, dpi=150)
                    plt.close(fig)
                    print(f'  saved {fname.relative_to(out_dir.parent)}')


            _plot_green_with_ctrl(
                seg_id,
                green_olap, ctrl_olap,
                valve_seg_all, valve_seg_is_open, opens_local, closes_local, valve_pairs,
                OUT_DIR,
                csv_path, analysis_plane_idx, total_shift, map_info
            )

        # Plot red channel traces for reference
        if 'red' in ch_dict and ch_dict['red'].size:
            print('  plotting red channel')
            plot_segment(seg_id, ch_dict['red'], valve_seg_all, valve_seg_is_open, 'red', OUT_DIR)

    print(f'\nDone processing Analysis plane {analysis_plane_idx}. Folder: {OUT_DIR}')

print('\nAll Analysis planes processed.')


# ==================================================================
# 8) CSV COMBINE UTIL
# ==================================================================

def _combine_summary_csvs_across_planes(
    base_dir: Path,
    output_name: str = 'responding_cells_summary_all_planes.csv'
) -> None:
    """
    Aggregate per-plane responding_cells_summary.csv files into one totals CSV:
    columns = total_cells, total_responders, percent_responders
    """
    plane_dirs = sorted([
        p for p in base_dir.iterdir()
        if p.is_dir() and p.name.startswith('plots_plane') and p.name.endswith('_v5')
    ])
    if not plane_dirs:
        print('No per-plane summary folders found. Skipping totals CSV.')
        return

    def _to_int_bool(s):
        s = str(s).strip()
        if s in ('1', 'True', 'true', 'YES', 'yes', 'Y', 'y'):
            return 1
        if s in ('0', 'False', 'false', 'NO', 'no', 'N', 'n'):
            return 0
        # try numeric fallback
        try:
            return 1 if float(s) >= 0.5 else 0
        except Exception:
            return 0

    total_cells = 0
    total_resp = 0

    for d in plane_dirs:
        csv_path = d / 'responding_cells_summary.csv'
        if not csv_path.exists():
            continue

        with open(csv_path, newline='') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header:
                continue

            # find the is_responder column by name or fall back to index 3
            try:
                idx_is_resp = header.index('is_responder')
            except ValueError:
                idx_is_resp = 3  # current schema: [plane, segment, cell_index, is_responder, ...]

            rows = list(reader)

        # accumulate
        total_cells += len(rows)
        for r in rows:
            if not r:
                continue
            # guard length
            val = r[idx_is_resp] if idx_is_resp < len(r) else 0
            total_resp += _to_int_bool(val)

    percent = (100.0 * total_resp / total_cells) if total_cells > 0 else 0.0

    out_csv = base_dir / output_name
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['total_cells', 'total_responders', 'percent_responders'])
        w.writerow([int(total_cells), int(total_resp), f'{percent:.2f}'])

    print(f"Totals saved to: {out_csv}  "
          f"(total_cells={total_cells}, total_responders={total_resp}, percent={percent:.2f}%)")


# Run the combiner at the end
_combine_summary_csvs_across_planes(ROI_NPZ_PATH.parent)
