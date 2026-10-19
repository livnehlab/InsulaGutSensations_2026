import os
import glob
import pickle
import numpy as np
import pandas as pd
import scipy.io
import scipy.stats as stats
import matplotlib

matplotlib.use("Qt5Agg")
matplotlib.rcParams["pdf.fonttype"] = 42

import matplotlib.pyplot as plt

from os.path import join
from scipy.stats import norm


# ============================================================
# User settings
# ============================================================

MAIN_FOLDER = r"\\isi.storwis.weizmann.ac.il\Labs\livneh\omerizh\vagal_OxtR_data summary\Revision_experiments\1Hz training"
SAVE_ROOT = r"\\isi.storwis.weizmann.ac.il\Labs\livneh\omerizh\vagal_OxtR_data summary\Revision_experiments\1Hz training\analysis_results3"

mice = ["OI389", "OI390", "OI391", "OI392"]  # change to your mice

photometry_hz = 50
monkey_logic_hz = 1000

pre_opto_sec = 1
pre_odor_no_opto_sec = 5

max_time_sec_for_raster = 20


# ============================================================
# Conditions
# ============================================================

NO_OPTO_CONDITION = 6
HZ1_CONDITION = 12

CONDITION_ORDER = [
    NO_OPTO_CONDITION,
    HZ1_CONDITION,
]

CONDITION_LABELS = {
    NO_OPTO_CONDITION: "No opto",
    HZ1_CONDITION: "1Hz opto",
}


# ============================================================
# Behavioral event codes
# ============================================================

start_cue_code = 23      # odor cue start
stop_cue_code = 24       # odor cue end

start_opto_code = 30     # opto stim start
stop_opto_code = 31      # opto stim end

start_reward_code = 27   # reward, if relevant


# ============================================================
# Original contingency TrialErrors response codes
# ============================================================

MISS_CODE = 2
HIT_CODE = 3
CR_CODE = 4
FA_CODE = 5

RESPONSE_LABELS = {
    MISS_CODE: "Miss",
    HIT_CODE: "Hit",
    CR_CODE: "CR",
    FA_CODE: "FA",
}


# ============================================================
# General helpers
# ============================================================

def natural_day_sort(day_name):
    """
    Sort day folders like day1, day2, day10 in the correct order.
    """
    digits = "".join([c for c in day_name if c.isdigit()])
    if digits == "":
        return 999999
    return int(digits)


def calculate_dprime(hit_rate, false_alarm_rate):
    """
    d' = Z(hit rate) - Z(false alarm rate)
    """
    if np.isnan(hit_rate) or np.isnan(false_alarm_rate):
        return np.nan
    return norm.ppf(hit_rate) - norm.ppf(false_alarm_rate)


# ============================================================
# Lick helpers
# ============================================================

def reduce_hz(plot):
    """
    Downsample MonkeyLogic lick data from 1000 Hz to photometry_hz.
    """
    final_array = []
    reduced_size = int(monkey_logic_hz / photometry_hz)

    for trial in plot:
        reduced_array = []
        for i in range(0, len(trial), reduced_size):
            reduced_data = trial[i: i + reduced_size]
            if sum(reduced_data) > 0:
                reduced_array.append(1)
            else:
                reduced_array.append(0)
        final_array.append(reduced_array)
    return final_array


def find_licks(licks_output):
    """
    Find lick onset bins for each trial.
    """
    trials_licks = []
    for trial_output in licks_output:
        licks = []
        output = np.asarray(trial_output)
        for l in range(len(output)):
            if output[l] >= 1 and (l == 0 or output[l - 1] < 1):
                licks.append(l)
        trials_licks.append(licks)
    return trials_licks


# ============================================================
# Timing extraction
# ============================================================

def cues_time(data, n_cues):
    """
    Find odor cue timing, reward timing, and opto timing in each trial.
    """
    cues_timing = []
    reward_num = []
    rewards_timing = []
    opto_timing = []

    for trial in range(1, n_cues + 1):
        behavioral_codes = data[f"Trial{trial}"]["BehavioralCodes"][0][0][0][0][1]
        codes_timing = data[f"Trial{trial}"]["BehavioralCodes"][0][0][0][0][0]

        start_cue = np.where(behavioral_codes == start_cue_code)[0]
        stop_cue = np.where(behavioral_codes == stop_cue_code)[0]

        cue_start_time = codes_timing[start_cue[0]][0]
        cue_stop_time = codes_timing[stop_cue[0]][0]

        start_opto = np.where(behavioral_codes == start_opto_code)[0]
        stop_opto = np.where(behavioral_codes == stop_opto_code)[0]

        if len(start_opto) > 0 and len(stop_opto) > 0:
            opto_start_time = codes_timing[start_opto[0]][0]
            opto_stop_time = codes_timing[stop_opto[-1]][0]
            opto_timing.append(
                (
                    opto_start_time,
                    opto_stop_time,
                    opto_start_time - cue_start_time,
                    opto_stop_time - cue_start_time
                )
            )
        else:
            opto_timing.append((np.nan, np.nan, np.nan, np.nan))

        if np.any(behavioral_codes == start_reward_code):
            start_reward = np.where(behavioral_codes == start_reward_code)[0]
            reward_num.append(len(start_reward))
            cues_timing.append(
                (
                    cue_start_time,
                    cue_stop_time,
                    codes_timing[start_reward[0]][0],
                    codes_timing[start_reward[-1]][0]
                )
            )
            rewards_timing.append(
                [
                    codes_timing[start_reward[i]][0] - cue_start_time
                    for i in range(len(start_reward))
                ]
            )
        else:
            cues_timing.append((cue_start_time, cue_stop_time))
            reward_num.append(0)
            rewards_timing.append([])

    return cues_timing, reward_num, rewards_timing, opto_timing


def calculate_licks(
        mat_data,
        data_type,
        cues_timing,
        opto_timing,
        pre_opto_sec=1,
        pre_odor_no_opto_sec=5
):
    """
    Rearrange licks by trial and downsample.
    """
    licks = [
        mat_data[f"Trial{i}"][0][0][10][0][0][11][0][0][data_type]
        for i in range(1, len(cues_timing) + 1)
    ]

    trial_start_times_ml = []
    cue_offset_sec = []

    pre_opto_samples_ml = int(pre_opto_sec * monkey_logic_hz)
    pre_odor_no_opto_samples_ml = int(pre_odor_no_opto_sec * monkey_logic_hz)

    for i in range(len(cues_timing)):
        cue_start_ml = int(cues_timing[i][0])
        opto_start_ml = opto_timing[i][0]

        if not np.isnan(opto_start_ml):
            trial_start_ml = max(0, int(opto_start_ml) - pre_opto_samples_ml)
        else:
            trial_start_ml = max(0, cue_start_ml - pre_odor_no_opto_samples_ml)

        trial_start_times_ml.append(trial_start_ml)
        cue_offset_sec.append((cue_start_ml - trial_start_ml) / monkey_logic_hz)

    licks_array = []
    for i in range(0, len(licks) - 1):
        current_trial_start = trial_start_times_ml[i]
        next_trial_start = trial_start_times_ml[i + 1]
        current_trial_array = licks[i][current_trial_start:]
        iti_array = licks[i + 1][:next_trial_start]
        licks_array.append(list(current_trial_array) + list(iti_array))

    licks_output = reduce_hz(licks_array)
    cue_offset_sec = cue_offset_sec[:-1]
    trial_start_times_ml = trial_start_times_ml[:-1]

    return licks_output, cue_offset_sec, trial_start_times_ml


# ============================================================
# Load raw MonkeyLogic data
# ============================================================

def load_full_data_from_mat(path):
    """
    Load data from mat file.
    """
    mat_data = scipy.io.loadmat(path)

    conditions_full = np.asarray(
        mat_data["TrialRecord"][0][0][7][0]
    ).squeeze()

    responses_full = np.asarray(
        mat_data["TrialRecord"][0][0]["TrialErrors"][0]
    ).squeeze()

    min_len = min(len(conditions_full), len(responses_full))
    conditions_full = conditions_full[:min_len]
    responses_full = responses_full[:min_len]

    cues_timing, rewards_number, rewards_times, opto_timing = cues_time(
        mat_data,
        len(conditions_full)
    )

    licks_output, cue_offset_sec, trial_start_times_ml = calculate_licks(
        mat_data=mat_data,
        data_type=0,
        cues_timing=cues_timing,
        opto_timing=opto_timing,
        pre_opto_sec=pre_opto_sec,
        pre_odor_no_opto_sec=pre_odor_no_opto_sec
    )

    n_raster_trials = len(licks_output)

    conditions_raster = conditions_full[:n_raster_trials]
    responses_raster = responses_full[:n_raster_trials]
    cues_timing = cues_timing[:n_raster_trials]
    rewards_number = rewards_number[:n_raster_trials]
    rewards_times = rewards_times[:n_raster_trials]
    opto_timing = opto_timing[:n_raster_trials]
    cue_offset_sec = cue_offset_sec[:n_raster_trials]
    trial_start_times_ml = trial_start_times_ml[:n_raster_trials]

    return {
        "conditions_for_dprime": conditions_full,
        "responses_for_dprime": responses_full,

        "conditions_for_raster": conditions_raster,
        "responses_for_raster": responses_raster,
        "licks_output": licks_output,
        "cues_timing": cues_timing,
        "rewards_number": rewards_number,
        "rewards_times": rewards_times,
        "opto_timing": opto_timing,
        "cue_offset_sec": cue_offset_sec,
        "trial_start_times_ml": trial_start_times_ml,
    }


def load_mouse_day_data(mouse_mat_files):
    """
    Concatenate mat files.
    """
    conditions_dprime_concat = []
    responses_dprime_concat = []

    conditions_raster_concat = []
    responses_raster_concat = []
    licks_concat = []
    cues_timing_concat = []
    rewards_number_concat = []
    rewards_times_concat = []
    opto_timing_concat = []
    cue_offset_sec_concat = []
    trial_start_times_ml_concat = []

    for mat_path in mouse_mat_files:
        data = load_full_data_from_mat(mat_path)
        conditions_dprime_concat.extend(data["conditions_for_dprime"])
        responses_dprime_concat.extend(data["responses_for_dprime"])

        conditions_raster_concat.extend(data["conditions_for_raster"])
        responses_raster_concat.extend(data["responses_for_raster"])
        licks_concat.extend(data["licks_output"])
        cues_timing_concat.extend(data["cues_timing"])
        rewards_number_concat.extend(data["rewards_number"])
        rewards_times_concat.extend(data["rewards_times"])
        opto_timing_concat.extend(data["opto_timing"])
        cue_offset_sec_concat.extend(data["cue_offset_sec"])
        trial_start_times_ml_concat.extend(data["trial_start_times_ml"])

    return {
        "conditions_for_dprime": np.asarray(conditions_dprime_concat),
        "responses_for_dprime": np.asarray(responses_dprime_concat),

        "conditions_for_raster": np.asarray(conditions_raster_concat),
        "responses_for_raster": np.asarray(responses_raster_concat),
        "licks_output": licks_concat,
        "cues_timing": cues_timing_concat,
        "rewards_number": rewards_number_concat,
        "rewards_times": rewards_times_concat,
        "opto_timing": opto_timing_concat,
        "cue_offset_sec": cue_offset_sec_concat,
        "trial_start_times_ml": trial_start_times_ml_concat,
    }


# ============================================================
# Standardized calculation functions
# ============================================================

def calculate_dprime_for_mouse_day(
        conditions_played,
        responses,
        no_opto_condition=NO_OPTO_CONDITION,
        hz1_condition=HZ1_CONDITION
):
    """
    Standardized d' calculation using Hit (3), Miss (2) for Go (condition 6)
    and FA (5), CR (4) for No-Go (condition 12).
    
    Hit Rate = (Hits + 0.5) / (Hits + Misses + 1)
    FA Rate = (FA + 0.5) / (FA + CR + 1)
    """
    conditions_played = np.asarray(conditions_played)
    responses = np.asarray(responses)

    no_opto_trials = np.where(conditions_played == no_opto_condition)[0]
    hz1_trials = np.where(conditions_played == hz1_condition)[0]

    no_opto_responses = responses[no_opto_trials]
    hz1_responses = responses[hz1_trials]

    # Calculate count metrics
    n_hits = np.sum(no_opto_responses == HIT_CODE)
    n_misses = np.sum(no_opto_responses == MISS_CODE)
    go_trials = n_hits + n_misses

    n_false_alarms = np.sum(hz1_responses == FA_CODE)
    n_correct_rejects = np.sum(hz1_responses == CR_CODE)
    nogo_trials = n_false_alarms + n_correct_rejects

    # corrected rates for d' calculation
    hit_rate = (n_hits + 0.5) / (go_trials + 1) if go_trials > 0 else np.nan
    false_alarm_rate = (n_false_alarms + 0.5) / (nogo_trials + 1) if nogo_trials > 0 else np.nan

    d_prime = calculate_dprime(hit_rate, false_alarm_rate)

    # Observed rates (raw percentages)
    raw_hit_rate = n_hits / go_trials if go_trials > 0 else np.nan
    raw_fa_rate = n_false_alarms / nogo_trials if nogo_trials > 0 else np.nan

    return {
        "n_no_opto_trials": go_trials,
        "n_hz1_trials": nogo_trials,
        "n_hits": n_hits,
        "n_misses": n_misses,
        "n_false_alarms": n_false_alarms,
        "n_correct_rejects": n_correct_rejects,
        "hit_rate": hit_rate,
        "false_alarm_rate": false_alarm_rate,
        "raw_hit_rate": raw_hit_rate,
        "raw_fa_rate": raw_fa_rate,
        "d_prime": d_prime,
    }


def calculate_dprime_for_window(resp, conds):
    """
    d' calculator for rolling windows.
    """
    hits = np.sum((resp == HIT_CODE) & (conds == NO_OPTO_CONDITION))
    misses = np.sum((resp == MISS_CODE) & (conds == NO_OPTO_CONDITION))
    fas = np.sum((resp == FA_CODE) & (conds == HZ1_CONDITION))
    crs = np.sum((resp == CR_CODE) & (conds == HZ1_CONDITION))

    go_trials = hits + misses
    nogo_trials = fas + crs

    hit_rate = (hits + 0.5) / (go_trials + 1)
    fa_rate = (fas + 0.5) / (nogo_trials + 1)

    d_prime = norm.ppf(hit_rate) - norm.ppf(fa_rate)
    return d_prime


def calculate_rolling_dprime_session(conditions, responses, window_size=20):
    """
    Extract relevant trials (Condition 6 and 12) and calculate rolling d'.
    """
    mask = (conditions == NO_OPTO_CONDITION) | (conditions == HZ1_CONDITION)
    conds_sub = conditions[mask]
    resp_sub = responses[mask]

    n_trials = len(resp_sub)
    if n_trials < window_size:
        return np.array([])

    rolling_dprimes = []
    for i in range(n_trials - window_size + 1):
        w_resp = resp_sub[i : i + window_size]
        w_cond = conds_sub[i : i + window_size]
        rolling_dprimes.append(calculate_dprime_for_window(w_resp, w_cond))

    return np.array(rolling_dprimes)


def calculate_nan_sem(matrix, axis=0):
    """
    Robust SEM calculation across columns/rows ignoring NaNs.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        n_non_nan = np.sum(~np.isnan(matrix), axis=axis)
        std = np.nanstd(matrix, axis=axis, ddof=1)
        sem = std / np.sqrt(n_non_nan)
        sem[n_non_nan <= 1] = np.nan
        return sem


# ============================================================
# Raster plotting (original logic preserved)
# ============================================================

def plot_lick_raster_by_condition_for_mouse_day(
        mouse,
        day,
        conditions_played,
        responses,
        licks_output,
        cues_timing,
        rewards_times,
        opto_timing,
        cue_offset_sec,
        save_path,
        max_time_sec=20
):
    os.makedirs(save_path, exist_ok=True)
    trials_licks = find_licks(licks_output)

    for condition_code in CONDITION_ORDER:
        trial_indices = np.where(conditions_played == condition_code)[0]
        trial_indices = [
            i for i in trial_indices
            if i < len(trials_licks)
            and i < len(licks_output)
            and i < len(cues_timing)
            and i < len(rewards_times)
            and i < len(opto_timing)
            and i < len(cue_offset_sec)
            and i < len(responses)
        ]

        if len(trial_indices) == 0:
            continue

        lick_times_by_trial = [
            (np.asarray(trials_licks[i]) / photometry_hz) - cue_offset_sec[i]
            for i in trial_indices
        ]

        fig, ax = plt.subplots(figsize=(12, 7))
        ax.eventplot(
            lick_times_by_trial,
            orientation="horizontal",
            lineoffsets=np.arange(len(trial_indices)),
            linelengths=0.8,
            linewidths=0.8,
            colors="black"
        )

        ax.axvline(0, linestyle="-", color="black", alpha=0.8, label="Odor onset")

        odor_ends = []
        for trial_i in trial_indices:
            cue_start = cues_timing[trial_i][0]
            cue_end = cues_timing[trial_i][1]
            odor_ends.append((cue_end - cue_start) / monkey_logic_hz)
        odor_end_sec = np.nanmedian(odor_ends)
        ax.axvspan(0, odor_end_sec, alpha=0.2, color="gray", label="Odor cue")

        opto_label_added = False
        opto_windows_sec = []
        for row_i, trial_i in enumerate(trial_indices):
            opto_start_rel = opto_timing[trial_i][2]
            opto_end_rel = opto_timing[trial_i][3]
            if not np.isnan(opto_start_rel) and not np.isnan(opto_end_rel):
                opto_start_sec = opto_start_rel / monkey_logic_hz
                opto_end_sec = opto_end_rel / monkey_logic_hz
                opto_windows_sec.append((opto_start_sec, opto_end_sec))
                ax.hlines(
                    y=row_i,
                    xmin=opto_start_sec,
                    xmax=opto_end_sec,
                    linewidth=3,
                    color="red",
                    alpha=0.7,
                    label="1Hz opto" if not opto_label_added else None
                )
                opto_label_added = True

        if len(opto_windows_sec) > 0:
            opto_windows_sec = np.asarray(opto_windows_sec)
            median_opto_start = np.nanmedian(opto_windows_sec[:, 0])
            median_opto_end = np.nanmedian(opto_windows_sec[:, 1])
            median_opto_duration = median_opto_end - median_opto_start
            ax.axvline(median_opto_start, linestyle="--", color="red", alpha=0.8, label="Median opto start")
            ax.axvline(median_opto_end, linestyle="--", color="darkred", alpha=0.8, label="Median opto end")
            ax.axvspan(median_opto_start, median_opto_end, color="red", alpha=0.08)
            ax.text(x=(median_opto_start+median_opto_end)/2, y=-0.5, s=f"Opto: {median_opto_duration:.2f} sec", ha='center', va='bottom', color='red')

        reward_label_added = False
        for row_i, trial_i in enumerate(trial_indices):
            trial_reward_times = rewards_times[trial_i]
            if len(trial_reward_times) > 0:
                reward_times_sec = np.asarray(trial_reward_times) / monkey_logic_hz
                ax.scatter(reward_times_sec, np.full(len(reward_times_sec), row_i), marker="|", s=80, color="green", label="Reward" if not reward_label_added else None)
                reward_label_added = True

        min_time_sec = min(-cue_offset_sec[i] for i in trial_indices)
        ax.set_xlim(min_time_sec, max_time_sec if max_time_sec is not None else 20)
        ax.set_ylim(len(trial_indices), -1)
        ax.set_yticks(np.arange(len(trial_indices)))

        y_tick_labels = [f"T{trial_i + 1} | {int(responses[trial_i])} {RESPONSE_LABELS.get(int(responses[trial_i]), 'Unknown')}" for trial_i in trial_indices]
        ax.set_yticklabels(y_tick_labels, fontsize=8)
        ax.set_title(f"{mouse} | {day} | {CONDITION_LABELS[condition_code]}")
        ax.set_xlabel("Time from odor cue onset (sec)")
        ax.set_ylabel("Trial | Response")

        handles, labels = ax.get_legend_handles_labels()
        unique_labels = dict(zip(labels, handles))
        if len(unique_labels) > 0:
            ax.legend(unique_labels.values(), unique_labels.keys(), frameon=False, loc="upper right")

        plt.tight_layout()
        safe_day = str(day).replace("\\", "_").replace("/", "_")
        file_base = f"lick_raster_{mouse}_{safe_day}_condition_{condition_code}"
        fig.savefig(join(save_path, f"{file_base}.png"), dpi=300)
        fig.savefig(join(save_path, f"{file_base}.pdf"))
        plt.close(fig)


# ============================================================
# Main calculation across days and mice
# ============================================================

def calculate_dprime_across_days(
        main_folder,
        save_root,
        mice
):
    """
    Read all day folders and calculate metrics per mouse per day.
    
    Supports:
    - Splits day1 into day1 (1st half) at day_number=1.0 and day1 (2nd half) at day_number=1.5.
    - Day number sorting.
    """
    os.makedirs(save_root, exist_ok=True)

    day_folders = [
        folder for folder in os.listdir(main_folder)
        if os.path.isdir(join(main_folder, folder))
    ]
    day_folders = sorted(day_folders, key=natural_day_sort)

    all_results = []

    for day_folder in day_folders:
        day_path = join(main_folder, day_folder)
        day_number = natural_day_sort(day_folder)

        print(f"\nProcessing {day_folder}")

        for mouse in mice:
            mouse_mat_files = glob.glob(join(day_path, f"*{mouse}*.mat"))
            if len(mouse_mat_files) == 0:
                print(f"  {mouse}: no .mat files found")
                continue

            mouse_mat_files = sorted(mouse_mat_files)
            mouse_day_data = load_mouse_day_data(mouse_mat_files)

            # Raster plots (run on standard whole-session data)
            raster_save_path = join(save_root, "rasters", day_folder, mouse)
            plot_lick_raster_by_condition_for_mouse_day(
                mouse=mouse,
                day=day_folder,
                conditions_played=mouse_day_data["conditions_for_raster"],
                responses=mouse_day_data["responses_for_raster"],
                licks_output=mouse_day_data["licks_output"],
                cues_timing=mouse_day_data["cues_timing"],
                rewards_times=mouse_day_data["rewards_times"],
                opto_timing=mouse_day_data["opto_timing"],
                cue_offset_sec=mouse_day_data["cue_offset_sec"],
                save_path=raster_save_path,
                max_time_sec=max_time_sec_for_raster
            )

            # Split Day 1 into two halves
            if day_folder.lower() == "day1":
                conds_full = mouse_day_data["conditions_for_dprime"]
                resp_full = mouse_day_data["responses_for_dprime"]
                T = len(conds_full)
                split_idx = T // 2

                # Half 1 (plotted at day_number = 1.0)
                dprime_res_1 = calculate_dprime_for_mouse_day(conds_full[:split_idx], resp_full[:split_idx])
                row_1 = {
                    "mouse": mouse,
                    "day": "day1 (1st half)",
                    "day_number": 1.0,
                    **dprime_res_1
                }
                all_results.append(row_1)

                # Half 2 (plotted at day_number = 1.5)
                dprime_res_2 = calculate_dprime_for_mouse_day(conds_full[split_idx:], resp_full[split_idx:])
                row_2 = {
                    "mouse": mouse,
                    "day": "day1 (2nd half)",
                    "day_number": 1.5,
                    **dprime_res_2
                }
                all_results.append(row_2)

                print(f"  {mouse} (day1 split): 1st half d'={dprime_res_1['d_prime']:.3f}, 2nd half d'={dprime_res_2['d_prime']:.3f}")

            else:
                # Normal day calculation
                dprime_result = calculate_dprime_for_mouse_day(
                    conditions_played=mouse_day_data["conditions_for_dprime"],
                    responses=mouse_day_data["responses_for_dprime"],
                    no_opto_condition=NO_OPTO_CONDITION,
                    hz1_condition=HZ1_CONDITION
                )
                row = {
                    "mouse": mouse,
                    "day": day_folder,
                    "day_number": float(day_number),
                    **dprime_result
                }
                all_results.append(row)
                print(f"  {mouse}: d'={dprime_result['d_prime']:.3f}, hit_rate={dprime_result['raw_hit_rate']:.3f}, FA_rate={dprime_result['raw_fa_rate']:.3f}")

    if len(all_results) == 0:
        print("No data found.")
        return None

    dprime_df = pd.DataFrame(all_results)
    dprime_df.to_csv(join(save_root, "dprime_by_mouse_day.csv"), index=False)

    summary_df = (
        dprime_df
        .groupby(["day", "day_number"], observed=True)
        .agg(
            mean_d_prime=("d_prime", "mean"),
            sem_d_prime=(
                "d_prime",
                lambda x: (
                    np.nanstd(x, ddof=1) / np.sqrt(np.sum(~np.isnan(x)))
                    if np.sum(~np.isnan(x)) > 1
                    else 0
                )
            ),
            mean_hit_rate=("raw_hit_rate", "mean"),
            sem_hit_rate=(
                "raw_hit_rate",
                lambda x: (
                    np.nanstd(x, ddof=1) / np.sqrt(np.sum(~np.isnan(x)))
                    if np.sum(~np.isnan(x)) > 1
                    else 0
                )
            ),
            mean_fa_rate=("raw_fa_rate", "mean"),
            sem_fa_rate=(
                "raw_fa_rate",
                lambda x: (
                    np.nanstd(x, ddof=1) / np.sqrt(np.sum(~np.isnan(x)))
                    if np.sum(~np.isnan(x)) > 1
                    else 0
                )
            ),
            n_mice=("d_prime", lambda x: np.sum(~np.isnan(x))),
            total_no_opto_trials=("n_no_opto_trials", "sum"),
            total_hz1_trials=("n_hz1_trials", "sum"),
            total_hits=("n_hits", "sum"),
            total_false_alarms=("n_false_alarms", "sum")
        )
        .reset_index()
        .sort_values("day_number")
    )
    summary_df.to_csv(join(save_root, "dprime_group_summary_by_day.csv"), index=False)

    return dprime_df, summary_df


# ============================================================
# Plotting functions across days
# ============================================================

def plot_metric_across_days(
        dprime_df,
        summary_df,
        save_root,
        metric_name='d_prime'
):
    """
    Plot a behavioral metric (d', hit_rate, fa_rate) across days.
    """
    os.makedirs(save_root, exist_ok=True)

    if metric_name == 'd_prime':
        y_col = 'd_prime'
        mean_col = 'mean_d_prime'
        sem_col = 'sem_d_prime'
        ylabel = "Performance (d')"
        title = "Performance (d') across days"
        filename = "dprime_across_days_1hz"
        hline_val = 0
        ylim = None
    elif metric_name == 'hit_rate':
        y_col = 'raw_hit_rate'
        mean_col = 'mean_hit_rate'
        sem_col = 'sem_hit_rate'
        ylabel = "Hit Rate"
        title = "Hit Rate across days"
        filename = "hit_rate_across_days_1hz"
        hline_val = None
        ylim = (-0.05, 1.05)
    elif metric_name == 'fa_rate':
        y_col = 'raw_fa_rate'
        mean_col = 'mean_fa_rate'
        sem_col = 'sem_fa_rate'
        ylabel = "False Alarm Rate"
        title = "False Alarm Rate across days"
        filename = "fa_rate_across_days_1hz"
        hline_val = None
        ylim = (-0.05, 1.05)
    else:
        raise ValueError(f"Unknown metric_name: {metric_name}")

    fig, ax = plt.subplots(figsize=(8, 5))

    # Create equally spaced x positions for each session/day
    summary_df = summary_df.sort_values("day_number").copy()
    summary_df["x_pos"] = np.arange(len(summary_df))

    day_to_x = dict(zip(summary_df["day"], summary_df["x_pos"]))

    # Individual mice
    for mouse in sorted(dprime_df["mouse"].unique()):
        mouse_df = dprime_df[dprime_df["mouse"] == mouse].copy()
        mouse_df = mouse_df.sort_values("day_number")
        mouse_df["x_pos"] = mouse_df["day"].map(day_to_x)

        ax.plot(
            mouse_df["x_pos"],
            mouse_df[y_col],
            marker="o",
            linewidth=1,
            alpha=0.35,
            color="gray",
            zorder=1
        )

    # Group mean ± SEM
    ax.errorbar(
        summary_df["x_pos"],
        summary_df[mean_col],
        yerr=summary_df[sem_col],
        marker="o",
        linewidth=3,
        capsize=4,
        color="#0f4c5c",
        label="Mean ± SEM",
        zorder=3
    )

    if hline_val is not None:
        ax.axhline(hline_val, linestyle="--", linewidth=1, color="black", alpha=0.4)

    ax.set_xlabel("Session / day")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if ylim is not None:
        ax.set_ylim(ylim)

    ax.set_xticks(summary_df["x_pos"])
    ax.set_xticklabels(summary_df["day"], rotation=25, ha='right')

    ax.legend(frameon=False)
    plt.tight_layout()
    fig.savefig(join(save_root, f"{filename}.png"), dpi=300)
    fig.savefig(join(save_root, f"{filename}.pdf"))
    plt.close(fig)

    print(f"Saved {metric_name} cross-days plot in: {save_root}")


# ============================================================
# Rolling d' comparison (First vs Last Day)
# ============================================================

def plot_rolling_dprime_first_vs_last_day(
        main_folder,
        save_root,
        mice,
        window_size=20
):
    """
    Load data from the first and last days of training.
    Compute the rolling d' within those sessions.
    Plot both rolling curves as separate lines on the same plot.
    """
    os.makedirs(save_root, exist_ok=True)

    day_folders = [
        folder for folder in os.listdir(main_folder)
        if os.path.isdir(join(main_folder, folder))
    ]
    day_folders = sorted(day_folders, key=natural_day_sort)

    if len(day_folders) < 2:
        print("Not enough days to generate first vs last day rolling d' plot.")
        return

    first_day = day_folders[0]
    last_day = day_folders[5]

    def get_rolling_matrix_for_day(day_folder):
        day_path = join(main_folder, day_folder)
        all_rollings = []
        for mouse in mice:
            mouse_mat_files = glob.glob(join(day_path, f"*{mouse}*.mat"))
            if len(mouse_mat_files) == 0:
                continue
            mouse_mat_files = sorted(mouse_mat_files)
            mouse_day_data = load_mouse_day_data(mouse_mat_files)
            
            rolling_dprime = calculate_rolling_dprime_session(
                mouse_day_data["conditions_for_dprime"],
                mouse_day_data["responses_for_dprime"],
                window_size=window_size
            )
            if len(rolling_dprime) > 0:
                all_rollings.append(rolling_dprime)
        
        if len(all_rollings) == 0:
            return None, 0
            
        max_len = max(len(arr) for arr in all_rollings)
        padded_rolling = []
        for arr in all_rollings:
            if len(arr) < max_len:
                padded = np.pad(arr, (0, max_len - len(arr)), mode='constant', constant_values=np.nan)
            else:
                padded = arr
            padded_rolling.append(padded)
        return np.vstack(padded_rolling), max_len

    # Calculate for first and last days
    matrix_first, len_first = get_rolling_matrix_for_day(first_day)
    matrix_last, len_last = get_rolling_matrix_for_day(last_day)

    fig, ax = plt.subplots(figsize=(8, 5))

    # Center alignment offset (for window size = 20, center is trial 10, offset = 9)
    center_offset = window_size // 2

    # Plot first day rolling curve
    if matrix_first is not None:
        mean_first = np.nanmean(matrix_first, axis=0)
        sem_first = calculate_nan_sem(matrix_first, axis=0)
        x_first = np.arange(1, len_first + 1) + (center_offset - 1)
        
        ax.plot(x_first, mean_first, color='gray', linewidth=2.5, label=f"First day ({first_day})")
        ax.fill_between(x_first, mean_first - sem_first, mean_first + sem_first, color='gray', alpha=0.15)

    # Plot last day rolling curve
    if matrix_last is not None:
        mean_last = np.nanmean(matrix_last, axis=0)
        sem_last = calculate_nan_sem(matrix_last, axis=0)
        x_last = np.arange(1, len_last + 1) + (center_offset - 1)
        
        ax.plot(x_last, mean_last, color='#0f4c5c', linewidth=2.5, label=f"Last day ({last_day})")
        ax.fill_between(x_last, mean_last - sem_last, mean_last + sem_last, color='#0f4c5c', alpha=0.15)

    ax.axhline(0, color='black', linestyle='--', alpha=0.3)
    ax.set_xlabel(f"Trial of session (center of {window_size}-trial window)", fontsize=11)
    ax.set_ylabel("Performance (d')", fontsize=11)
    ax.set_title("Within-Session d' Learning Dynamics: First vs Last Day", fontsize=12, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(frameon=False)

    plt.tight_layout()
    fig.savefig(join(save_root, "rolling_dprime_first_vs_last_day.png"), dpi=300)
    fig.savefig(join(save_root, "rolling_dprime_first_vs_last_day.pdf"))
    plt.close(fig)

    print(f"Saved rolling comparison plot (first vs last day) in: {save_root}")


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    result = calculate_dprime_across_days(
        main_folder=MAIN_FOLDER,
        save_root=SAVE_ROOT,
        mice=mice
    )

    if result is not None:
        dprime_df, summary_df = result

        # 1. Plot standard metrics across days
        plot_metric_across_days(dprime_df, summary_df, SAVE_ROOT, metric_name='d_prime')
        plot_metric_across_days(dprime_df, summary_df, SAVE_ROOT, metric_name='hit_rate')
        plot_metric_across_days(dprime_df, summary_df, SAVE_ROOT, metric_name='fa_rate')

        # 2. Plot rolling d' comparison between first and last day
        plot_rolling_dprime_first_vs_last_day(
            main_folder=MAIN_FOLDER,
            save_root=SAVE_ROOT,
            mice=mice,
            window_size=20
        )
