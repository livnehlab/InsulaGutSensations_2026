import os
import re
import glob
import numpy as np
import pandas as pd
import scipy.io
import matplotlib

matplotlib.use("Qt5Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
import matplotlib.pyplot as plt
from os.path import join
from scipy.stats import norm

# ============================================================
# User settings
# ============================================================
MAIN_FOLDER = r"Z:\omerizh\vagal_OxtR_data summary\Dprime"
SAVE_ROOT = r"Z:\omerizh\vagal_OxtR_data summary\Dprime\analysis_results_groups3"
# Define mice by groups
mice_groups = {
    "detection": ["OI151", "OI152", "OI192", "OI208", "OI212", "OI230", "OI240", "OI241", "OI242"],
    "controls": ["OI118", "OI119", "OI120", "SS37", "SS38"]
}
# mice_groups = {
#     "reversal": ["OI151", "OI152", "OI192", "OI208", "OI212", "OI230", "OI240", "OI241", "OI242"]
# }
photometry_hz = 50
monkey_logic_hz = 1000
pre_opto_sec = 1
pre_odor_no_opto_sec = 5
max_time_sec_for_raster = 20
max_consecutive_hit_no_lick = 7

# ============================================================
# Behavioral event codes
# ============================================================
start_cue_code = 23  # odor cue start
stop_cue_code = 24  # odor cue end
start_opto_code = 30  # opto stim start
stop_opto_code = 31  # opto stim end
start_reward_code = 27  # reward, if relevant

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
def natural_sort_key(s):
    """
    Sort strings containing numbers naturally.
    Useful for sorting file names like 'run1.mat', 'run2.mat', 'run10.mat' correctly.
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]


def natural_day_sort(day_name):
    """
    Sort day folders like day1, day2, day10 in the correct order.
    """
    digits = "".join([c for c in day_name if c.isdigit()])
    if digits == "":
        return 999999
    return int(digits)


def corrected_rate(n_events, n_trials):
    """
    Log-linear correction to avoid infinite d' when rate is exactly 0 or 1.
    rate = (n_events + 0.5) / (n_trials + 1)
    """
    if n_trials == 0:
        return np.nan

    return (n_events + 0.5) / (n_trials + 1)


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


def has_lick_after_odor(trial_licks, cue_offset_sec):
    """
    Return True if the mouse licked after odor onset in this trial.
    """
    cue_bin = int(np.floor(cue_offset_sec * photometry_hz))
    trial_licks = np.asarray(trial_licks)
    if cue_bin >= len(trial_licks):
        return False
    return np.any(trial_licks[cue_bin:] > 0)


def find_cutoff_after_consecutive_no_lick(
        responses,
        licks_output,
        cue_offset_sec,
        max_consecutive=7,
        recovery_hit_streak=5
):
    """
    Find the first streak of `max_consecutive` Go trials with no lick,
    unless the mouse later recovers with `recovery_hit_streak`
    consecutive Hits among Go trials. CR and FA trials are ignored,
    while a Miss breaks the recovery streak.
    """
    consecutive_count = 0
    streak_start_idx = None
    n_trials = min(len(responses), len(licks_output), len(cue_offset_sec))

    for trial_i in range(n_trials):
        response_code = responses[trial_i]

        # We only care about the "no opto" condition (Hits and Misses).
        if response_code not in [HIT_CODE, MISS_CODE]:
            continue

        no_lick = not has_lick_after_odor(
            licks_output[trial_i],
            cue_offset_sec[trial_i]
        )

        if no_lick:
            if consecutive_count == 0:
                streak_start_idx = trial_i
            consecutive_count += 1

            # Check if we hit the 7-miss streak
            if consecutive_count >= max_consecutive:

                # Look for 5 consecutive Hits among later Go trials.
                # CR and FA trials are ignored; a Miss resets the streak.
                hit_streak = 0
                recovered = False
                for future_i in range(trial_i + 1, n_trials):
                    future_response = responses[future_i]

                    if future_response == HIT_CODE:
                        hit_streak += 1
                        if hit_streak >= recovery_hit_streak:
                            recovered = True
                            break
                    elif future_response == MISS_CODE:
                        hit_streak = 0

                if recovered:
                    # This no-lick streak was temporary. Continue looking
                    # for a later stopping point.
                    consecutive_count = 0
                    streak_start_idx = None
                else:
                    return streak_start_idx
        else:
            # The mouse licked on a Go trial (Hit), reset the Miss counter
            consecutive_count = 0
            streak_start_idx = None

    return None



def trim_mouse_day_after_stopped_licking(mouse_day_data, mouse=None, day=None):
    """
    Keep only trials before the mouse stopped licking.
    """
    # Cut after 7 no-lick Go trials unless followed by 5 Hits in a row.
    cutoff_idx = find_cutoff_after_consecutive_no_lick(
        responses=mouse_day_data["responses_for_raster"],
        licks_output=mouse_day_data["licks_output"],
        cue_offset_sec=mouse_day_data["cue_offset_sec"],
        max_consecutive=7,
        recovery_hit_streak=5
    )

    if cutoff_idx is None:
        return mouse_day_data

    label = ""
    if mouse is not None and day is not None:
        label = f"{mouse} {day}: "

    print(
        f"    {label}cutting trials from trial {cutoff_idx + 1} onward "
        f"because there was a streak of 7 no-lick Go trials without "
        f"a later recovery streak of 5 Hits."
    )

    trimmed_data = {}
    for key, value in mouse_day_data.items():
        trimmed_data[key] = value[:cutoff_idx]

    return trimmed_data

# ============================================================
# MonkeyLogic timing extraction
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
    Rearrange licks by trial and downsample to photometry_hz.
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
    for i in range(len(licks)):
        current_trial_start = trial_start_times_ml[i]
        current_trial_array = licks[i][current_trial_start:]

        # For every trial except the last one, append the beginning of the
        # next MonkeyLogic trial up to its analysis start. This preserves the
        # complete interval between the current and next analysis trials.
        if i < len(licks) - 1:
            next_trial_start = trial_start_times_ml[i + 1]
            iti_array = licks[i + 1][:next_trial_start]
            trial_array = list(current_trial_array) + list(iti_array)
        else:
            # Keep the last trial as well. It has no following trial from
            # which to append an ITI, but its own data are sufficient for
            # lick detection after odor onset.
            trial_array = list(current_trial_array)

        licks_array.append(trial_array)

    licks_output = reduce_hz(licks_array)
    return licks_output, cue_offset_sec, trial_start_times_ml


# ============================================================
# Load one .mat file
# ============================================================
def load_full_data_from_mat(path):
    """
    Load condition codes, TrialErrors responses, licks, and event timing
    from one MonkeyLogic .mat file.
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

    expected_trials = len(responses_full)
    aligned_lengths = {
        "conditions": len(conditions_full),
        "responses": len(responses_full),
        "licks": len(licks_output),
        "cue timing": len(cues_timing),
        "reward timing": len(rewards_times),
        "opto timing": len(opto_timing),
        "cue offsets": len(cue_offset_sec),
        "trial starts": len(trial_start_times_ml),
    }
    if any(length != expected_trials for length in aligned_lengths.values()):
        raise ValueError(
            f"Trial arrays are not aligned in {path}: {aligned_lengths}"
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


# ============================================================
# Load all files for one mouse on one day
# ============================================================
def load_mouse_day_data(mouse_mat_files):
    """
    Load and concatenate all .mat files for one mouse on one day.
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
# d' calculation
# ============================================================
def calculate_dprime_for_mouse_day(
        responses
):
    """
    Calculate d' for one mouse on one day based entirely on response codes.
    """
    responses = np.asarray(responses)
    n_hits = np.sum(responses == HIT_CODE)
    n_misses = np.sum(responses == MISS_CODE)
    n_false_alarms = np.sum(responses == FA_CODE)
    n_correct_rejects = np.sum(responses == CR_CODE)

    n_no_opto_trials = n_hits + n_misses
    n_hz1_trials = n_false_alarms + n_correct_rejects

    hit_rate = corrected_rate(n_hits, n_no_opto_trials)
    false_alarm_rate = corrected_rate(n_false_alarms, n_hz1_trials)
    d_prime = calculate_dprime(hit_rate, false_alarm_rate)
    return {
        "n_no_opto_trials": n_no_opto_trials,
        "n_hz1_trials": n_hz1_trials,
        "n_hits": n_hits,
        "n_misses": n_misses,
        "n_false_alarms": n_false_alarms,
        "n_correct_rejects": n_correct_rejects,
        "hit_rate": hit_rate,
        "false_alarm_rate": false_alarm_rate,
        "d_prime": d_prime,
    }


# ============================================================
# Raster plotting
# ============================================================
def plot_lick_raster_by_condition_for_mouse_day(
        mouse,
        day,
        responses,
        licks_output,
        cues_timing,
        rewards_times,
        opto_timing,
        cue_offset_sec,
        save_path,
        max_time_sec=20
):
    """
    Plot lick rasters for one mouse on one day, separating plots purely by response code types.
    """
    os.makedirs(save_path, exist_ok=True)
    trials_licks = find_licks(licks_output)

    # Infer condition types strictly from the response codes
    go_trials = np.where(np.isin(responses, [HIT_CODE, MISS_CODE]))[0]
    nogo_trials = np.where(np.isin(responses, [FA_CODE, CR_CODE]))[0]

    plot_conditions = [
        ("No opto", go_trials),
        ("1Hz opto", nogo_trials)
    ]

    for condition_label, trial_indices in plot_conditions:
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
            print(f"  {mouse} {day}: no trials for {condition_label}")
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
                    y=row_i, xmin=opto_start_sec, xmax=opto_end_sec,
                    linewidth=3, color="red", alpha=0.7,
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
            ax.text(
                x=(median_opto_start + median_opto_end) / 2, y=-0.5,
                s=f"Opto: {median_opto_duration:.2f} sec",
                ha="center", va="bottom", color="red"
            )
        reward_label_added = False
        for row_i, trial_i in enumerate(trial_indices):
            trial_reward_times = rewards_times[trial_i]
            if len(trial_reward_times) > 0:
                reward_times_sec = np.asarray(trial_reward_times) / monkey_logic_hz
                ax.scatter(
                    reward_times_sec, np.full(len(reward_times_sec), row_i),
                    marker="|", s=80, color="green",
                    label="Reward" if not reward_label_added else None
                )
                reward_label_added = True
        min_time_sec = min(-cue_offset_sec[i] for i in trial_indices)
        if max_time_sec is not None:
            ax.set_xlim(min_time_sec, max_time_sec)
        else:
            longest_trial_sec = max(
                len(licks_output[i]) / photometry_hz - cue_offset_sec[i]
                for i in trial_indices
            )
            ax.set_xlim(min_time_sec, longest_trial_sec)
        ax.set_ylim(len(trial_indices), -1)
        ax.set_yticks(np.arange(len(trial_indices)))
        y_tick_labels = []
        for trial_i in trial_indices:
            response_code = int(responses[trial_i])
            response_label = RESPONSE_LABELS.get(response_code, "Unknown")
            y_tick_labels.append(f"T{trial_i + 1} | {response_code} {response_label}")
        ax.set_yticklabels(y_tick_labels, fontsize=8)
        ax.set_title(f"{mouse} | {day} | {condition_label}")
        ax.set_xlabel("Time from odor cue onset (sec)")
        ax.set_ylabel("Trial | Response")
        handles, labels = ax.get_legend_handles_labels()
        unique_labels = dict(zip(labels, handles))
        if len(unique_labels) > 0:
            ax.legend(unique_labels.values(), unique_labels.keys(), frameon=False, loc="upper right")
        plt.tight_layout()
        safe_day = str(day).replace("\\", "_").replace("/", "_")
        file_base = f"lick_raster_{mouse}_{safe_day}_{condition_label.replace(' ', '_')}"
        fig.savefig(join(save_path, f"{file_base}.png"), dpi=300)
        fig.savefig(join(save_path, f"{file_base}.pdf"))
        plt.close(fig)


# ============================================================
# Main calculation across days and mice groups
# ============================================================
def calculate_dprime_across_days(
        main_folder,
        save_root,
        mice_groups
):
    """
    Read all day folders and calculate metrics per mouse per day.
    Splits day 1 into two halves automatically.
    """
    os.makedirs(save_root, exist_ok=True)
    all_results = []
    for group_name, mice_list in mice_groups.items():
        group_path = join(main_folder, group_name)
        if not os.path.exists(group_path):
            print(f"Group folder not found: {group_path}")
            continue
        print(f"\nProcessing Group: {group_name}")
        day_folders = [
            folder for folder in os.listdir(group_path)
            if os.path.isdir(join(group_path, folder))
        ]
        day_folders = sorted(day_folders, key=natural_day_sort)
        for day_folder in day_folders:
            day_path = join(group_path, day_folder)
            day_number = natural_day_sort(day_folder)
            print(f"  Day: {day_folder}")
            for mouse in mice_list:
                # Find all files for this mouse (will match run1, run2, etc.)
                mouse_mat_files = glob.glob(join(day_path, f"*{mouse}*.mat"))
                if len(mouse_mat_files) == 0:
                    print(f"    {mouse}: no .mat files found")
                    continue
                # Use natural sort key to ensure run2 goes before run10, etc.
                mouse_mat_files = sorted(mouse_mat_files, key=natural_sort_key)

                # Load the raw data
                mouse_day_data = load_mouse_day_data(mouse_mat_files)

                # Raster plots for the FULL day (before trimming)
                raster_save_path = join(save_root, "rasters", day_folder, group_name, mouse)
                plot_lick_raster_by_condition_for_mouse_day(
                    mouse=mouse,
                    day=day_folder,
                    responses=mouse_day_data["responses_for_raster"],
                    licks_output=mouse_day_data["licks_output"],
                    cues_timing=mouse_day_data["cues_timing"],
                    rewards_times=mouse_day_data["rewards_times"],
                    opto_timing=mouse_day_data["opto_timing"],
                    cue_offset_sec=mouse_day_data["cue_offset_sec"],
                    save_path=raster_save_path,
                    max_time_sec=max_time_sec_for_raster
                )

                # Trim the session based on consecutive no-licks
                mouse_day_data = trim_mouse_day_after_stopped_licking(
                    mouse_day_data=mouse_day_data,
                    mouse=mouse,
                    day=day_folder
                )

                # Split Day 1 into first 0.5 and second 0.5
                if day_number == 1:
                    total_trials = len(mouse_day_data["responses_for_dprime"])
                    half_idx = total_trials // 2

                    # First Half
                    dprime_result_1 = calculate_dprime_for_mouse_day(
                        responses=mouse_day_data["responses_for_dprime"][:half_idx]
                    )
                    row_1 = {
                        "mouse": mouse,
                        "group": group_name,
                        "day": day_folder + "_part1",
                        "day_number": 1.1,
                        **dprime_result_1
                    }
                    all_results.append(row_1)

                    # Second Half
                    dprime_result_2 = calculate_dprime_for_mouse_day(
                        responses=mouse_day_data["responses_for_dprime"][half_idx:]
                    )
                    row_2 = {
                        "mouse": mouse,
                        "group": group_name,
                        "day": day_folder + "_part2",
                        "day_number": 1.2,
                        **dprime_result_2
                    }
                    all_results.append(row_2)

                    print(
                        f"    {mouse} (part 1): trials d'={dprime_result_1['d_prime']:.3f}, HR={dprime_result_1['hit_rate']:.3f}, FA={dprime_result_1['false_alarm_rate']:.3f}"
                    )
                    print(
                        f"    {mouse} (part 2): d'={dprime_result_2['d_prime']:.3f}, HR={dprime_result_2['hit_rate']:.3f}, FA={dprime_result_2['false_alarm_rate']:.3f}"
                    )
                else:
                    # Regular entire day calculation
                    dprime_result = calculate_dprime_for_mouse_day(
                        responses=mouse_day_data["responses_for_dprime"]
                    )

                    row = {
                        "mouse": mouse,
                        "group": group_name,
                        "day": day_folder,
                        "day_number": float(day_number),
                        **dprime_result
                    }

                    all_results.append(row)

                    print(
                        f"    {mouse}: d'={dprime_result['d_prime']:.3f}, "
                        f"HR={dprime_result['hit_rate']:.3f}, "
                        f"FA={dprime_result['false_alarm_rate']:.3f}, "
                        f"n_no_opto={dprime_result['n_no_opto_trials']}, "
                        f"n_1hz={dprime_result['n_hz1_trials']}"
                    )
    if len(all_results) == 0:
        print("No data found.")
        return None
    dprime_df = pd.DataFrame(all_results)
    print(sorted(dprime_df["day_number"].unique()))
    dprime_df.to_csv(join(save_root, "dprime_by_mouse_day_groups.csv"), index=False)

    summary_df = (
        dprime_df
        .groupby(["group", "day", "day_number"], observed=True)
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
            mean_hit_rate=("hit_rate", "mean"),
            sem_hit_rate=(
                "hit_rate",
                lambda x: (
                    np.nanstd(x, ddof=1) / np.sqrt(np.sum(~np.isnan(x)))
                    if np.sum(~np.isnan(x)) > 1
                    else 0
                )
            ),
            mean_false_alarm_rate=("false_alarm_rate", "mean"),
            sem_false_alarm_rate=(
                "false_alarm_rate",
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
        .sort_values(["group", "day_number"])
    )
    summary_df.to_csv(join(save_root, "dprime_group_summary_by_day_groups.csv"), index=False)
    return dprime_df, summary_df

def plot_day05_vs_last_session_dprime(dprime_df, save_root):
    """
    Compare d' in session 0.5 vs the last available session for each mouse.
    """
    dprime_df = dprime_df[dprime_df["group"] == "detection"].copy()
    rows = []

    for mouse in dprime_df["mouse"].unique():

        mouse_df = dprime_df[dprime_df["mouse"] == mouse].copy()
        mouse_df = mouse_df.sort_values("day_number")

        # Find session 0.5
        day05 = mouse_df[np.isclose(mouse_df["day_number"], 1.1)]

        if len(day05) == 0:
            continue

        day05_dprime = day05["d_prime"].iloc[0]

        # Last session
        last_dprime = mouse_df.iloc[-1]["d_prime"]

        rows.append({
            "mouse": mouse,
            "group": mouse_df.iloc[0]["group"],
            "day05_dprime": day05_dprime,
            "last_dprime": last_dprime
        })

    if len(rows) == 0:
        print("No day 0.5 data found")
        return

    compare_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(6, 6))

    for _, row in compare_df.iterrows():
        ax.plot(
            [0, 1],
            [row["day05_dprime"], row["last_dprime"]],
            marker="o",
            alpha=0.5,
            linewidth=1
        )

    mean_day05 = compare_df["day05_dprime"].mean()
    mean_last = compare_df["last_dprime"].mean()

    sem_day05 = (
        compare_df["day05_dprime"].std(ddof=1)
        / np.sqrt(len(compare_df))
    )
    sem_last = (
        compare_df["last_dprime"].std(ddof=1)
        / np.sqrt(len(compare_df))
    )

    ax.errorbar(
        [0, 1],
        [mean_day05, mean_last],
        yerr=[sem_day05, sem_last],
        color="black",
        linewidth=3,
        marker="o",
        capsize=5
    )

    ax.set_xticks([0, 1])
    ax.set_ylim([0, 5.1])
    ax.set_xticklabels(["0.5", "Last session"])
    ax.set_ylabel("d'")
    ax.set_title("d' : Session 0.5 vs Last Session")

    plt.tight_layout()

    fig.savefig(
        join(save_root, "dprime_day05_vs_last_session.png"),
        dpi=300
    )
    fig.savefig(
        join(save_root, "dprime_day05_vs_last_session.pdf")
    )

    plt.close(fig)

# ============================================================
# Generic Plotting Function
# ============================================================
def plot_metric_across_days(
        df,
        summary_df,
        metric_col,
        mean_col,
        sem_col,
        ylim,
        ylabel,
        title,
        filename_suffix,
        save_root,
        hline_y=None
):
    """
    Plot a specified metric (e.g. d', hit rate, FA rate) across days for multiple groups.
    """
    os.makedirs(save_root, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))

    group_colors = {
        "detection": "red",
        "controls": "black"
    }
    all_day_numbers = sorted(summary_df["day_number"].unique())
    day_map = dict(zip(summary_df["day_number"], summary_df["day"]))

    day_idx_map = {dn: i for i, dn in enumerate(all_day_numbers)}
    for mouse in sorted(df["mouse"].unique()):
        mouse_df = df[df["mouse"] == mouse].copy()
        mouse_df = mouse_df.sort_values("day_number")

        x_positions = [day_idx_map[dn] for dn in mouse_df["day_number"]]

        group = mouse_df["group"].iloc[0]
        color = group_colors.get(group, "gray")

        ax.plot(
            x_positions, mouse_df[metric_col],
            marker="o", linewidth=1, alpha=0.35, color=color, zorder=1
        )

    for group_name in sorted(summary_df["group"].unique()):
        group_summary = summary_df[summary_df["group"] == group_name].copy()
        group_summary = group_summary.sort_values("day_number")

        x_positions = [day_idx_map[dn] for dn in group_summary["day_number"]]

        color = group_colors.get(group_name, "gray")

        ax.errorbar(
            x_positions, group_summary[mean_col],
            yerr=group_summary[sem_col],
            marker="o", linewidth=3, capsize=4, color=color,
            label=group_name.capitalize(), zorder=3
        )
    if hline_y is not None:
        ax.axhline(hline_y, linestyle="--", linewidth=1, color="black", alpha=0.4)

    ax.set_xlabel("Session / day")
    ax.set_ylabel(ylabel)
    if isinstance(ylim, tuple) and len(ylim) == 2:
        ax.set_ylim(ylim)
    ax.set_title(title)

    all_day_numbers = sorted(summary_df["day_number"].unique())
    day_map = dict(zip(summary_df["day_number"], summary_df["day"]))

    ax.set_xticks(range(len(all_day_numbers)))
    ax.set_xticklabels([day_map[dn] for dn in all_day_numbers], rotation=45)
    ax.legend(frameon=False)
    plt.tight_layout()

    fig.savefig(join(save_root, f"{filename_suffix}.png"), dpi=300)
    fig.savefig(join(save_root, f"{filename_suffix}.pdf"))
    plt.close(fig)
    print(f"Saved {title} plot in: {save_root}")


# ============================================================
# Run
# ============================================================
if __name__ == "__main__":
    result = calculate_dprime_across_days(
        main_folder=MAIN_FOLDER,
        save_root=SAVE_ROOT,
        mice_groups=mice_groups
    )
    if result is not None:
        dprime_df, summary_df = result

        # 1. d-prime plot
        plot_metric_across_days(
            df=dprime_df,
            summary_df=summary_df,
            metric_col="d_prime",
            mean_col="mean_d_prime",
            sem_col="sem_d_prime",
            ylim=(-1, 5),
            ylabel="Performance (d')",
            title="Discriminability across days: Detection vs Control",
            filename_suffix="dprime_across_days_compare",
            save_root=SAVE_ROOT,
            hline_y=0
        )

        # 2. Hit rate plot
        plot_metric_across_days(
            df=dprime_df,
            summary_df=summary_df,
            metric_col="hit_rate",
            mean_col="mean_hit_rate",
            sem_col="sem_hit_rate",
            ylim=(-0.05, 1.05),
            ylabel="Hit Rate",
            title="Hit Rate across days: Detection vs Control",
            filename_suffix="hit_rate_across_days_compare",
            save_root=SAVE_ROOT,
            hline_y=None
        )
        # 3. False alarm rate plot
        plot_metric_across_days(
            df=dprime_df,
            summary_df=summary_df,
            metric_col="false_alarm_rate",
            mean_col="mean_false_alarm_rate",
            sem_col="sem_false_alarm_rate",
            ylim=(-0.05, 1.05),
            ylabel="False Alarm Rate",
            title="False Alarm Rate across days: Detection vs Control",
            filename_suffix="fa_rate_across_days_compare",
            save_root=SAVE_ROOT,
            hline_y=None
        )

        plot_day05_vs_last_session_dprime(
            dprime_df=dprime_df,
            save_root=SAVE_ROOT
        )