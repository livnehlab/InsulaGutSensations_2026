"""
Revision experiment: insular cortex fiber photometry during opto stimulation.

Conditions: without caspase, bilateral caspase, and a control with the red light far from
the neck. For each pyPhotometry (.ppd) file: detect the stimulation cues, preprocess the
signal (low-pass filter, double-exponential bleaching correction), z-score each trial to
its pre-stimulus baseline, and plot trial heatmaps and mean traces. The response amplitude
(mean z-score in RESPONSE_WINDOW_SEC) is compared across conditions.

Dependency: analysis_code.tools.data_import is data_import.py from pyPhotometry
(Thomas Akam, GPL-3.0, https://github.com/pyPhotometry/code), used to read the .ppd files.
"""

import os
import glob
import pickle
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Qt5Agg")
matplotlib.rcParams["pdf.fonttype"] = 42

import matplotlib.pyplot as plt

from os.path import join
from scipy.signal import butter, filtfilt
from scipy.stats import linregress
from scipy.optimize import curve_fit

from analysis_code.tools import data_import


# ---------------------------------------------------------------------------
# Data locations (set these before running)
# ---------------------------------------------------------------------------
BEHAVIOR_ROOT = r'path\to\behavior_data'  # folder with the behavioral data (Dprime/, Revision_experiments/)


# ============================================================
# User settings
# ============================================================

MAIN_FOLDER = BEHAVIOR_ROOT + r"\Revision_experiments\InsCtx photometry"
SAVE_ROOT = BEHAVIOR_ROOT + r"\Revision_experiments\InsCtx photometry\analysis_results"

# Folder names inside MAIN_FOLDER
CONDITION_FOLDERS = {
    "bilateral_casp": "Bilateral casp",
    "without_casp": "Without casp",
    "red_light_far_from_neck": "without casp control exp with red light far from neck",
}

CONDITION_ORDER = [
    "without_casp",
    "bilateral_casp",
    "red_light_far_from_neck",
]

CONDITION_LABELS = {
    "bilateral_casp": "Bilateral casp",
    "without_casp": "Without casp",
    "red_light_far_from_neck": "Red light far\nfrom neck",
}

# Trial alignment settings
PRE_EVENT_SEC = 5
POST_EVENT_SEC = 20

# Baseline and response windows for conclusion plot
BASELINE_WINDOW_SEC = (-5, 0)
RESPONSE_WINDOW_SEC = (0, 4)
OPTO_WINDOW_SEC = (0, 4)
# Cue detection settings
# These are used by identify_cues.
# They will be converted to frames based on sampling rate.
SPACE_BETWEEN_CUES_SEC = 10
FIRST_SPACE_SEC = 0.5

# Heatmap settings
SORT_TRIALS_BY_RESPONSE = False

# Save options
SAVE_INDIVIDUAL_TRACES = True


# ============================================================
# Cue detection and preprocessing functions
# ============================================================

def identify_cues(signals, sampling_rate):
    """
    Find cue timing in pyPhotometry digital signal.

    sampling_rate is passed in so that space_between_cues and first_space are
    defined in seconds (SPACE_BETWEEN_CUES_SEC, FIRST_SPACE_SEC).
    """

    space_between_cues = int(SPACE_BETWEEN_CUES_SEC * sampling_rate)
    first_space = int(FIRST_SPACE_SEC * sampling_rate)

    signals_frames = []
    current_state = 0
    space = 0

    for i in range(len(signals) - 1):
        if signals_frames:
            space_cues = space_between_cues
        else:
            space_cues = first_space

        if signals[i] == 1 : #and signals[i + 1] != 0
            if current_state == 1:
                pass
            elif space > space_cues:
                signals_frames.append(i)
                current_state = 1
                space = 0
            else:
                current_state = 1
                space += 1
        else:
            if current_state == 1:
                space = 0
                current_state = 0
            else:
                space += 1

    return signals_frames


def find_trial_starts(signal, sampling_rate, minimal_space):
    """
    Alternative cue detection for 2-sec-long TTL pulses (not used by default).
    """

    tolerance = int(0.2 * sampling_rate)
    trial_length_frames = int(2 * sampling_rate)
    long_interval_frames = minimal_space

    min_length = trial_length_frames - tolerance
    max_length = trial_length_frames + tolerance

    sequences = []
    in_sequence = False
    seq_start_idx = None
    seq_length = 0

    for idx, val in enumerate(signal):
        if val == 1:
            if not in_sequence:
                in_sequence = True
                seq_start_idx = idx
                seq_length = 1
            else:
                seq_length += 1
        else:
            if in_sequence:
                sequences.append({
                    "start_idx": seq_start_idx,
                    "length": seq_length,
                    "end_idx": seq_start_idx + seq_length - 1
                })
                in_sequence = False

    if in_sequence:
        sequences.append({
            "start_idx": seq_start_idx,
            "length": seq_length,
            "end_idx": seq_start_idx + seq_length - 1
        })

    trial_starts_indices = []
    previous_seq = None

    for seq in sequences:
        seq_length = seq["length"]
        seq_start_idx = seq["start_idx"]

        if min_length <= seq_length <= max_length:
            trial_starts_indices.append(seq_start_idx)

        elif seq_length > max_length:
            if previous_seq is not None:
                time_since_prev = seq["start_idx"] - previous_seq["end_idx"]

                if previous_seq["length"] > max_length:
                    trial_starts_indices.append(seq_start_idx)

                elif min_length <= previous_seq["length"] <= max_length:
                    if time_since_prev > long_interval_frames:
                        trial_starts_indices.append(seq_start_idx)

            else:
                trial_starts_indices.append(seq_start_idx)

        previous_seq = seq

    return trial_starts_indices


def double_exponential(t, const, amp_fast, amp_slow, tau_slow, tau_multiplier):
    """
    Compute a double exponential function with constant offset.
    """

    tau_fast = tau_slow * tau_multiplier

    return (
        const
        + amp_slow * np.exp(-t / tau_slow)
        + amp_fast * np.exp(-t / tau_fast)
    )


def preprocess_photometry(analog1, analog2, time_seconds, sampling_rate):
    """
    Preprocess photometry signal.

    - low-pass filter
    - double-exponential bleaching correction
    - isosbestic motion estimate (computed, but not subtracted; see below)
    """

    b, a = butter(2, 10, btype="low", fs=sampling_rate)

    dlight_denoised = filtfilt(b, a, analog1)
    iso_denoised = filtfilt(b, a, analog2)

    # Fit double exponential to signal channel
    max_sig = np.percentile(dlight_denoised, 99)
    initial_params = [max_sig / 2, max_sig / 4, max_sig / 4, 3600, 0.1]

    bounds = (
        [0, 0, 0, 600, 0],
        [max_sig, max_sig, max_sig, 36000, 1]
    )

    try:
        dlight_params, _ = curve_fit(
            double_exponential,
            time_seconds,
            dlight_denoised,
            p0=initial_params,
            bounds=bounds,
            maxfev=1000
        )

        dlight_expfit = double_exponential(time_seconds, *dlight_params)

    except Exception as e:
        print(f"Warning: dLight exponential fit failed. Using median baseline. Error: {e}")
        dlight_expfit = np.full_like(dlight_denoised, np.median(dlight_denoised))

    # Fit double exponential to isosbestic/control channel
    max_iso = np.percentile(iso_denoised, 99)
    initial_params_iso = [max_iso / 2, max_iso / 4, max_iso / 4, 3600, 0.1]

    bounds_iso = (
        [0, 0, 0, 600, 0],
        [max_iso, max_iso, max_iso, 36000, 1]
    )

    try:
        iso_params, _ = curve_fit(
            double_exponential,
            time_seconds,
            iso_denoised,
            p0=initial_params_iso,
            bounds=bounds_iso,
            maxfev=1000
        )

        iso_expfit = double_exponential(time_seconds, *iso_params)

    except Exception as e:
        print(f"Warning: iso exponential fit failed. Using median baseline. Error: {e}")
        iso_expfit = np.full_like(iso_denoised, np.median(iso_denoised))

    dlight_detrended = dlight_denoised - dlight_expfit
    iso_detrended = iso_denoised - iso_expfit

    # Motion correction estimate.
    slope, intercept, r_value, p_value, std_err = linregress(
        x=iso_detrended,
        y=dlight_detrended
    )

    dlight_est_motion = intercept + slope * iso_detrended

    # The motion estimate is not subtracted:
    # dlight_corrected = dlight_detrended
    # (motion-subtracted version: dlight_corrected = dlight_detrended - dlight_est_motion)
    dlight_corrected = dlight_detrended

    # Return the bleaching-corrected signal
    return dlight_corrected


def load_photometry_data(path):
    """
    Load pyPhotometry file.

    Returns:
    - activity: preprocessed continuous activity
    - trial_starts: trial start frames
    - data: full pyPhotometry dictionary
    """

    data = data_import.import_ppd(
        path,
        low_pass=None,
        high_pass=None
    )

    analog1 = data["analog_1"]
    analog2 = data["analog_2"]
    time_seconds = data["time"] / 1000
    sampling_rate = data["sampling_rate"]

    activity = preprocess_photometry(
        analog1=analog1,
        analog2=analog2,
        time_seconds=time_seconds,
        sampling_rate=sampling_rate
    )

    trial_starts = identify_cues(
        signals=data["digital_1"],
        sampling_rate=sampling_rate
    )

    return activity, trial_starts, data


# ============================================================
# Trial alignment
# ============================================================

def extract_mouse_name_from_ppd_path(ppd_path):
    """
    Extract mouse name from file name.

    Assumes file names of the form <mouse>_<day>.ppd.
    Example:
    mouse33_day1.ppd -> mouse33_day1
    """

    file_name = os.path.basename(ppd_path)
    mouse_name = os.path.splitext(file_name)[0]

    return mouse_name


def align_activity_to_trials(
        activity,
        trial_starts,
        sampling_rate,
        pre_event_sec=PRE_EVENT_SEC,
        post_event_sec=POST_EVENT_SEC
):
    """
    Create trials x time matrix aligned to trial starts.

    Returns:
    - trial_matrix: trials x time
    - time_axis: seconds relative to trial start
    - valid_trial_starts: cue frames that had enough data around them
    """

    pre_frames = int(pre_event_sec * sampling_rate)
    post_frames = int(post_event_sec * sampling_rate)

    window_length = pre_frames + post_frames

    time_axis = np.arange(-pre_frames, post_frames) / sampling_rate

    trial_matrix = []
    valid_trial_starts = []

    for start_frame in trial_starts:
        start_idx = int(start_frame) - pre_frames
        end_idx = int(start_frame) + post_frames

        if start_idx < 0:
            continue

        if end_idx > len(activity):
            continue

        trial_trace = activity[start_idx:end_idx]

        if len(trial_trace) != window_length:
            continue

        trial_matrix.append(trial_trace)
        valid_trial_starts.append(start_frame)

    if len(trial_matrix) == 0:
        return np.empty((0, window_length)), time_axis, []

    trial_matrix = np.asarray(trial_matrix)

    return trial_matrix, time_axis, valid_trial_starts


def baseline_zscore_trials(
        trial_matrix,
        time_axis,
        baseline_window_sec=BASELINE_WINDOW_SEC
):
    """
    Z-score each trial using its own baseline window.
    """

    baseline_mask = (
        (time_axis >= baseline_window_sec[0]) &
        (time_axis < baseline_window_sec[1])
    )

    if np.sum(baseline_mask) == 0:
        raise ValueError("Baseline window does not overlap time_axis.")

    baseline_mean = np.nanmean(trial_matrix[:, baseline_mask], axis=1, keepdims=True)
    baseline_std = np.nanstd(trial_matrix[:, baseline_mask], axis=1, keepdims=True)

    baseline_std[baseline_std == 0] = np.nan

    trial_matrix_z = (trial_matrix - baseline_mean) / baseline_std

    return trial_matrix_z


def calculate_response_metric(
        trial_matrix_z,
        time_axis,
        response_window_sec=RESPONSE_WINDOW_SEC
):
    """
    Calculate response amplitude for one mouse/file.

    Current metric:
    Mean z-scored activity in RESPONSE_WINDOW_SEC.

    """

    response_mask = (
        (time_axis >= response_window_sec[0]) &
        (time_axis < response_window_sec[1])
    )

    if np.sum(response_mask) == 0:
        return np.nan

    response_per_trial = np.nanmean(trial_matrix_z[:, response_mask], axis=1)
    response_mean = np.nanmean(response_per_trial)

    return response_mean


# ============================================================
# Plotting
# ============================================================

def plot_trial_heatmap(
        trial_matrix_z,
        time_axis,
        condition_label,
        mouse_name,
        save_path,
        sort_trials_by_response=SORT_TRIALS_BY_RESPONSE
):
    """
    Plot heatmap: trials x time.
    """

    os.makedirs(save_path, exist_ok=True)

    matrix_to_plot = np.asarray(trial_matrix_z)

    if sort_trials_by_response and matrix_to_plot.shape[0] > 0:
        response_mask = (
            (time_axis >= RESPONSE_WINDOW_SEC[0]) &
            (time_axis < RESPONSE_WINDOW_SEC[1])
        )

        response_values = np.nanmean(matrix_to_plot[:, response_mask], axis=1)
        sort_order = np.argsort(response_values)
        matrix_to_plot = matrix_to_plot[sort_order, :]

    fig, ax = plt.subplots(figsize=(10, 6))

    im = ax.imshow(
        matrix_to_plot,
        aspect="auto",
        interpolation="nearest",
        extent=[
            time_axis[0],
            time_axis[-1],
            matrix_to_plot.shape[0],
            0
        ],
        cmap="viridis"
    )

    ax.axvline(
        0,
        color="white",
        linestyle="--",
        linewidth=1.5,
        label="Trial start"
    )
    ax.axvspan(
        OPTO_WINDOW_SEC[0],
        OPTO_WINDOW_SEC[1],
        color="red",
        alpha=0.18,
        label="Opto / red light"
    )
    ax.set_title(f"{condition_label} | {mouse_name}")
    ax.set_xlabel("Time from trial start (sec)")
    ax.set_ylabel("Trials")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Activity z-score")

    ax.legend(frameon=False, loc="upper right")

    plt.tight_layout()

    fig.savefig(
        join(save_path, f"heatmap_{mouse_name}.png"),
        dpi=300
    )

    fig.savefig(
        join(save_path, f"heatmap_{mouse_name}.pdf")
    )

    plt.close(fig)


def plot_average_trace(
        trial_matrix_z,
        time_axis,
        condition_label,
        mouse_name,
        save_path
):
    """
    Plot average z-scored activity trace for one mouse/file.
    """

    os.makedirs(save_path, exist_ok=True)

    mean_trace = np.nanmean(trial_matrix_z, axis=0)

    if trial_matrix_z.shape[0] > 1:
        sem_trace = np.nanstd(trial_matrix_z, axis=0, ddof=1) / np.sqrt(trial_matrix_z.shape[0])
    else:
        sem_trace = np.zeros_like(mean_trace)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        time_axis,
        mean_trace,
        linewidth=2,
        label=mouse_name
    )

    ax.fill_between(
        time_axis,
        mean_trace - sem_trace,
        mean_trace + sem_trace,
        alpha=0.25
    )

    ax.axvline(
        0,
        color="black",
        linestyle="--",
        linewidth=1,
        label="Trial start"
    )

    ax.axhline(
        0,
        color="black",
        linestyle=":",
        linewidth=1,
        alpha=0.5
    )
    ax.axvspan(
        OPTO_WINDOW_SEC[0],
        OPTO_WINDOW_SEC[1],
        color="red",
        alpha=0.15,
        label="Opto / red light"
    )
    ax.set_title(f"Mean activity | {condition_label} | {mouse_name}")
    ax.set_xlabel("Time from trial start (sec)")
    ax.set_ylabel("Activity z-score")
    ax.legend(frameon=False)

    plt.tight_layout()

    fig.savefig(
        join(save_path, f"average_trace_{mouse_name}.png"),
        dpi=300
    )

    fig.savefig(
        join(save_path, f"average_trace_{mouse_name}.pdf")
    )

    plt.close(fig)


def plot_group_average_traces(
        all_mouse_summary,
        saved_traces,
        save_root
):
    """
    Plot group average time course per condition.

    Each mouse contributes its own average trace.
    """

    group_trace_path = join(save_root, "group_traces")
    os.makedirs(group_trace_path, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 6))

    for condition_name in CONDITION_ORDER:
        condition_traces = saved_traces.get(condition_name, [])

        if len(condition_traces) == 0:
            continue

        time_axis = condition_traces[0]["time_axis"]

        mouse_mean_traces = np.asarray([
            trace_info["mean_trace"]
            for trace_info in condition_traces
        ])

        group_mean = np.nanmean(mouse_mean_traces, axis=0)

        if mouse_mean_traces.shape[0] > 1:
            group_sem = np.nanstd(mouse_mean_traces, axis=0, ddof=1) / np.sqrt(mouse_mean_traces.shape[0])
        else:
            group_sem = np.zeros_like(group_mean)

        ax.plot(
            time_axis,
            group_mean,
            linewidth=2,
            label=f"{CONDITION_LABELS[condition_name]} (n={mouse_mean_traces.shape[0]})"
        )

        ax.fill_between(
            time_axis,
            group_mean - group_sem,
            group_mean + group_sem,
            alpha=0.2
        )

    ax.axvline(
        0,
        color="black",
        linestyle="--",
        linewidth=1,
        label="Trial start"
    )
    ax.axvspan(
        OPTO_WINDOW_SEC[0],
        OPTO_WINDOW_SEC[1],
        color="red",
        alpha=0.12,
        label="Opto / red light"
    )
    ax.axhline(
        0,
        color="black",
        linestyle=":",
        linewidth=1,
        alpha=0.5
    )

    ax.set_title("Group average photometry response")
    ax.set_xlabel("Time from trial start (sec)")
    ax.set_ylabel("Activity z-score")
    ax.legend(frameon=False)

    plt.tight_layout()

    fig.savefig(
        join(group_trace_path, "group_average_traces.png"),
        dpi=300
    )

    fig.savefig(
        join(group_trace_path, "group_average_traces.pdf")
    )

    plt.close(fig)


def plot_conclusion_bar(
        all_mouse_summary,
        save_root
):
    """
    Conclusion plot comparing the 3 conditions.

    Each dot = one mouse/file.
    Bar = group mean.
    Error bar = SEM.
    """

    conclusion_path = join(save_root, "conclusion")
    os.makedirs(conclusion_path, exist_ok=True)

    df = pd.DataFrame(all_mouse_summary)

    df.to_csv(
        join(conclusion_path, "photometry_response_by_mouse.csv"),
        index=False
    )

    summary_df = (
        df
        .groupby("condition", observed=True)
        .agg(
            mean_response=("response_metric", "mean"),
            sem_response=(
                "response_metric",
                lambda x: (
                    np.nanstd(x, ddof=1) / np.sqrt(np.sum(~np.isnan(x)))
                    if np.sum(~np.isnan(x)) > 1
                    else 0
                )
            ),
            n_mice=("response_metric", lambda x: np.sum(~np.isnan(x))),
            total_trials=("n_trials", "sum")
        )
        .reset_index()
    )

    summary_df.to_csv(
        join(conclusion_path, "photometry_response_group_summary.csv"),
        index=False
    )

    x_positions = np.arange(len(CONDITION_ORDER))

    means = []
    sems = []
    labels = []

    for condition_name in CONDITION_ORDER:
        condition_summary = summary_df[
            summary_df["condition"] == condition_name
        ]

        if len(condition_summary) == 0:
            means.append(np.nan)
            sems.append(0)
        else:
            means.append(condition_summary["mean_response"].iloc[0])
            sems.append(condition_summary["sem_response"].iloc[0])

        labels.append(CONDITION_LABELS[condition_name])

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.bar(
        x_positions,
        means,
        yerr=sems,
        capsize=5,
        alpha=0.6,
        edgecolor="black",
        label="Mean ± SEM",
        zorder=1
    )

    rng = np.random.default_rng(1)

    for condition_idx, condition_name in enumerate(CONDITION_ORDER):
        condition_df = df[df["condition"] == condition_name]

        y_values = condition_df["response_metric"].values

        jitter = rng.uniform(
            low=-0.08,
            high=0.08,
            size=len(y_values)
        )

        ax.scatter(
            np.full(len(y_values), condition_idx) + jitter,
            y_values,
            color="gray",
            edgecolor="black",
            alpha=0.9,
            zorder=3
        )

    for x, mean in zip(x_positions, means):
        if np.isnan(mean):
            label = "NaN"
            y = 0
        else:
            label = f"{mean:.2f}"
            y = mean

        ax.text(
            x,
            y,
            label,
            ha="center",
            va="bottom",
            fontsize=9
        )

    ax.axhline(
        0,
        color="black",
        linestyle=":",
        linewidth=1,
        alpha=0.6
    )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=25, ha="right")

    ax.set_ylabel(
        f"Mean z-scored activity\n{RESPONSE_WINDOW_SEC[0]}-{RESPONSE_WINDOW_SEC[1]} sec"
    )

    ax.set_title("Photometry response comparison across conditions")
    ax.legend(frameon=False)

    plt.tight_layout()

    fig.savefig(
        join(conclusion_path, "photometry_response_conclusion_bar.png"),
        dpi=300
    )

    fig.savefig(
        join(conclusion_path, "photometry_response_conclusion_bar.pdf")
    )

    plt.close(fig)

    print(f"Saved conclusion plots in: {conclusion_path}")


# ============================================================
# Main Analysis
# ============================================================

def analyze_ppd_file(
        ppd_path,
        condition_name,
        condition_label,
        save_root
):
    """
    Analyze one PPD file:
    - load
    - preprocess
    - detect trials
    - align activity
    - z-score
    - create heatmap
    - create individual average trace
    - calculate response metric
    """

    mouse_name = extract_mouse_name_from_ppd_path(ppd_path)

    print(f"Analyzing {condition_label}: {mouse_name}")

    activity, trial_starts, data = load_photometry_data(ppd_path)

    sampling_rate = data["sampling_rate"]

    trial_matrix, time_axis, valid_trial_starts = align_activity_to_trials(
        activity=activity,
        trial_starts=trial_starts,
        sampling_rate=sampling_rate,
        pre_event_sec=PRE_EVENT_SEC,
        post_event_sec=POST_EVENT_SEC
    )

    if trial_matrix.shape[0] == 0:
        print(f"  No valid trials found for {mouse_name}")
        return None, None

    trial_matrix_z = baseline_zscore_trials(
        trial_matrix=trial_matrix,
        time_axis=time_axis,
        baseline_window_sec=BASELINE_WINDOW_SEC
    )

    response_metric = calculate_response_metric(
        trial_matrix_z=trial_matrix_z,
        time_axis=time_axis,
        response_window_sec=RESPONSE_WINDOW_SEC
    )

    condition_save_path = join(save_root, condition_name, mouse_name)
    os.makedirs(condition_save_path, exist_ok=True)

    plot_trial_heatmap(
        trial_matrix_z=trial_matrix_z,
        time_axis=time_axis,
        condition_label=condition_label,
        mouse_name=mouse_name,
        save_path=condition_save_path,
        sort_trials_by_response=SORT_TRIALS_BY_RESPONSE
    )

    if SAVE_INDIVIDUAL_TRACES:
        plot_average_trace(
            trial_matrix_z=trial_matrix_z,
            time_axis=time_axis,
            condition_label=condition_label,
            mouse_name=mouse_name,
            save_path=condition_save_path
        )

    np.save(
        join(condition_save_path, "trial_matrix_z.npy"),
        trial_matrix_z
    )

    np.save(
        join(condition_save_path, "time_axis.npy"),
        time_axis
    )

    with open(join(condition_save_path, "valid_trial_starts.pkl"), "wb") as f:
        pickle.dump(valid_trial_starts, f)

    summary_row = {
        "condition": condition_name,
        "condition_label": condition_label,
        "mouse": mouse_name,
        "ppd_path": ppd_path,
        "sampling_rate": sampling_rate,
        "n_trials": trial_matrix_z.shape[0],
        "response_metric": response_metric,
    }

    mean_trace_info = {
        "condition": condition_name,
        "mouse": mouse_name,
        "time_axis": time_axis,
        "mean_trace": np.nanmean(trial_matrix_z, axis=0),
        "n_trials": trial_matrix_z.shape[0],
    }

    return summary_row, mean_trace_info


def run_analysis():
    """
    Run Analysis on all conditions and all PPD files.
    """

    os.makedirs(SAVE_ROOT, exist_ok=True)

    all_mouse_summary = []

    saved_traces = {
        condition_name: []
        for condition_name in CONDITION_ORDER
    }

    for condition_name in CONDITION_ORDER:
        condition_folder_name = CONDITION_FOLDERS[condition_name]
        condition_label = CONDITION_LABELS[condition_name]

        condition_folder_path = join(MAIN_FOLDER, condition_folder_name)

        if not os.path.exists(condition_folder_path):
            print(f"Condition folder not found: {condition_folder_path}")
            continue

        ppd_files = sorted(glob.glob(join(condition_folder_path, "*.ppd")))

        if len(ppd_files) == 0:
            print(f"No PPD files found in: {condition_folder_path}")
            continue

        for ppd_path in ppd_files:
            summary_row, mean_trace_info = analyze_ppd_file(
                ppd_path=ppd_path,
                condition_name=condition_name,
                condition_label=condition_label,
                save_root=SAVE_ROOT
            )

            if summary_row is None:
                continue

            all_mouse_summary.append(summary_row)
            saved_traces[condition_name].append(mean_trace_info)

    if len(all_mouse_summary) == 0:
        print("No valid data found.")
        return

    all_summary_df = pd.DataFrame(all_mouse_summary)

    all_summary_df.to_csv(
        join(SAVE_ROOT, "all_mouse_photometry_summary.csv"),
        index=False
    )

    plot_group_average_traces(
        all_mouse_summary=all_mouse_summary,
        saved_traces=saved_traces,
        save_root=SAVE_ROOT
    )

    plot_conclusion_bar(
        all_mouse_summary=all_mouse_summary,
        save_root=SAVE_ROOT
    )

    print("Finished photometry Analysis.")


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    run_analysis()