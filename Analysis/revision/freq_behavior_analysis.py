"""
Revision experiment: number of opto pulses (60 pulses / 4 s at 15 Hz, 15 pulses / 1 s at
15 Hz, 4 pulses at 15 Hz, 1 pulse) vs. no opto.

For each mouse, loads the MonkeyLogic files and computes the response rate of each
condition (Hit rate for no opto, FA rate for the opto conditions), with response-rate,
d' and hit / FA comparisons between the first and last thirds of the sessions across mice.
"""

import os
import glob
import pickle
import numpy as np
import pandas as pd
import scipy.io
import matplotlib
import scipy.stats as stats

matplotlib.use('Qt5Agg')
matplotlib.rcParams['pdf.fonttype'] = 42

import matplotlib.pyplot as plt

from os.path import join
from scipy.stats import wilcoxon, ttest_rel


# ---------------------------------------------------------------------------
# Data locations (set these before running)
# ---------------------------------------------------------------------------
BEHAVIOR_ROOT = r'path\to\behavior_data'  # folder with the behavioral data (Dprime/, Revision_experiments/)


# ============================================================
# User settings
# ============================================================

mice = ['mouse22', 'mouse23', 'mouse24', 'mouse25', 'mouse26', 'mouse27']  # mice to analyze

MAIN_FOLDER = BEHAVIOR_ROOT + r'\Revision_experiments\Detection freq change (60 to 1 pulses)'
SAVE_ROOT = BEHAVIOR_ROOT + r'\Revision_experiments\Detection freq change (60 to 1 pulses)\new_pulse_experiment_licks_data3'

photometry_hz = 50
monkey_logic_hz = 1000

pre_opto_sec = 1
pre_odor_no_opto_sec = 5

max_time_sec_for_raster = 20


# ============================================================
# Behavioral event codes
# ============================================================

start_cue_code = 23      # odor cue start
stop_cue_code = 24       # odor cue end

start_opto_code = 30     # opto stim start
stop_opto_code = 31      # opto stim end

start_reward_code = 27   # reward, if relevant


# ============================================================
# Condition codes
# ============================================================

NO_OPTO = 1
PULSES_60_4SEC_15HZ = 2
PULSES_15_1SEC_15HZ = 3
PULSES_4_15HZ = 4
PULSE_1 = 5

CONDITION_CODES = {
    "no_opto": NO_OPTO,
    "pulses_60_4sec_15hz": PULSES_60_4SEC_15HZ,
    "pulses_15_1sec_15hz": PULSES_15_1SEC_15HZ,
    "pulses_4_15hz": PULSES_4_15HZ,
    "pulse_1": PULSE_1,
}

CONDITION_LABELS = {
    "no_opto": "No opto",
    "pulses_60_4sec_15hz": "60 pulses\n4 sec 15Hz",
    "pulses_15_1sec_15hz": "15 pulses\n1 sec 15Hz",
    "pulses_4_15hz": "4 pulses\n15Hz",
    "pulse_1": "1 pulse",
}

CONDITION_ORDER = [
    "no_opto",
    "pulses_60_4sec_15hz",
    "pulses_15_1sec_15hz",
    "pulses_4_15hz",
    "pulse_1",
]

REFERENCE_CONDITION = "pulses_60_4sec_15hz"


# ============================================================
# Response codes
# ============================================================

RESPONSE_LABELS = {
    2: "Miss",
    3: "Hit",
    4: "CR",
    5: "FA",
}

# Metric for the conclusion plot:
# No opto: Hit rate
# Opto stimulation conditions: FA rate
TARGET_RESPONSE_BY_CONDITION = {
    "no_opto": 3,
    "pulses_60_4sec_15hz": 5,
    "pulses_15_1sec_15hz": 5,
    "pulses_4_15hz": 5,
    "pulse_1": 5,
}

TARGET_RESPONSE_LABEL_BY_CONDITION = {
    "no_opto": "Hit",
    "pulses_60_4sec_15hz": "FA",
    "pulses_15_1sec_15hz": "FA",
    "pulses_4_15hz": "FA",
    "pulse_1": "FA",
}


# ============================================================
# Basic helpers
# ============================================================

def reduce_hz(plot):
    """Downsample MonkeyLogic 1000 Hz data to photometry_hz."""
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
    """Find lick onset bins for each trial."""
    trials_licks = []

    for trial_output in licks_output:
        licks = []
        output = np.asarray(trial_output)

        for l in range(len(output)):
            if output[l] >= 1 and (l == 0 or output[l - 1] < 1):
                licks.append(l)

        trials_licks.append(licks)

    return trials_licks


def find_conditions(conditions_played):
    """Return trial indices for each condition."""
    conditions_played = np.asarray(conditions_played)

    condition_indices = {
        condition_name: np.where(conditions_played == condition_code)[0]
        for condition_name, condition_code in CONDITION_CODES.items()
    }

    return condition_indices


def get_licks_data(condition_indices, licks_output):
    """Save lick onset data by condition."""
    trials_licks = find_licks(licks_output)

    all_trials = pd.DataFrame(trials_licks)

    trials_licks_by_condition = {
        condition_name: pd.DataFrame([trials_licks[i] for i in indices if i < len(trials_licks)])
        for condition_name, indices in condition_indices.items()
    }

    raw_licks_by_condition = {
        condition_name: [licks_output[i] for i in indices if i < len(licks_output)]
        for condition_name, indices in condition_indices.items()
    }

    return all_trials, trials_licks_by_condition, raw_licks_by_condition


# ============================================================
# Load and align MonkeyLogic data
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
        behavioral_codes = data[f'Trial{trial}']['BehavioralCodes'][0][0][0][0][1]
        codes_timing = data[f'Trial{trial}']['BehavioralCodes'][0][0][0][0][0]

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

        if start_reward_code in behavioral_codes:
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
        mat_data[f'Trial{i}'][0][0][10][0][0][11][0][0][data_type]
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

    # The last trial is not used (its ITI ends at the next trial start)
    for i in range(0, len(licks) - 1):
        current_trial_start = trial_start_times_ml[i]
        next_trial_start = trial_start_times_ml[i + 1]

        current_trial_array = licks[i][current_trial_start:]
        iti_array = licks[i + 1][:next_trial_start]

        licks_array.append(list(current_trial_array) + list(iti_array))

    reshaped_trials = reduce_hz(licks_array)

    cue_offset_sec = cue_offset_sec[:-1]
    trial_start_times_ml = trial_start_times_ml[:-1]

    return reshaped_trials, cue_offset_sec, trial_start_times_ml


def load_mat_file(path):
    """Load MonkeyLogic file."""
    mat_data = scipy.io.loadmat(path)

    cues = mat_data['TrialRecord'][0][0][7][0]

    responses = np.asarray(
        mat_data['TrialRecord'][0][0]['TrialErrors'][0]
    ).squeeze()

    cues_timing, rewards, rewards_time, opto_timing = cues_time(
        mat_data,
        len(cues)
    )

    licks_all, cue_offset_sec, trial_start_times_ml = calculate_licks(
        mat_data,
        0,
        cues_timing,
        opto_timing,
        pre_opto_sec=pre_opto_sec,
        pre_odor_no_opto_sec=pre_odor_no_opto_sec
    )

    return (
        cues,
        responses,
        licks_all,
        cues_timing,
        rewards,
        rewards_time,
        opto_timing,
        cue_offset_sec,
        trial_start_times_ml
    )


# ============================================================
# Raster plotting
# ============================================================

def plot_lick_raster_by_condition(
        licks_output,
        condition_indices,
        cues_timing,
        rewards_times,
        opto_timing,
        cue_offset_sec,
        responses,
        save_path,
        photometry_hz=50,
        monkey_logic_hz=1000,
        max_time_sec=None
):
    raster_save_path = join(save_path, "lick_rasters")
    os.makedirs(raster_save_path, exist_ok=True)

    trials_licks = find_licks(licks_output)

    for condition_name in CONDITION_ORDER:
        if condition_name not in condition_indices:
            continue

        trial_indices = list(condition_indices[condition_name])
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
            print(f"No trials found for condition: {condition_name}")
            continue

        lick_times_by_trial = [
            (np.asarray(trials_licks[i]) / photometry_hz) - cue_offset_sec[i]
            for i in trial_indices
        ]

        fig, ax = plt.subplots(figsize=(12, 7))
        ax.eventplot(
            lick_times_by_trial,
            orientation='horizontal',
            lineoffsets=np.arange(len(trial_indices)),
            linelengths=0.8,
            linewidths=0.8,
            colors='black'
        )

        ax.axvline(0, linestyle='-', color='black', alpha=0.8, label='Odor onset')

        odor_ends = []
        for trial_i in trial_indices:
            cue_start = cues_timing[trial_i][0]
            cue_end = cues_timing[trial_i][1]
            odor_ends.append((cue_end - cue_start) / monkey_logic_hz)
        odor_end_sec = np.nanmedian(odor_ends)
        ax.axvspan(0, odor_end_sec, alpha=0.2, color='gray', label='Odor cue')

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
                    color='red',
                    alpha=0.7,
                    label='Opto stim' if not opto_label_added else None
                )
                opto_label_added = True

        if len(opto_windows_sec) > 0:
            opto_windows_sec = np.asarray(opto_windows_sec)
            median_opto_start = np.nanmedian(opto_windows_sec[:, 0])
            median_opto_end = np.nanmedian(opto_windows_sec[:, 1])
            median_opto_duration = median_opto_end - median_opto_start
            ax.axvline(median_opto_start, linestyle='--', color='red', alpha=0.8, label='Median opto start')
            ax.axvline(median_opto_end, linestyle='--', color='darkred', alpha=0.8, label='Median opto end')
            ax.axvspan(median_opto_start, median_opto_end, color='red', alpha=0.08)
            ax.text(x=(median_opto_start + median_opto_end) / 2, y=-0.5, s=f'Opto: {median_opto_duration:.2f} sec', ha='center', va='bottom', color='red')

        reward_label_added = False
        for row_i, trial_i in enumerate(trial_indices):
            trial_reward_times = rewards_times[trial_i]
            if len(trial_reward_times) > 0:
                reward_times_sec = np.asarray(trial_reward_times) / monkey_logic_hz
                ax.scatter(reward_times_sec, np.full(len(reward_times_sec), row_i), marker='|', s=80, color='green', label='Reward' if not reward_label_added else None)
                reward_label_added = True

        min_time_sec = min(-cue_offset_sec[i] for i in trial_indices)
        ax.set_xlim(min_time_sec, max_time_sec if max_time_sec is not None else 20)
        ax.set_ylim(len(trial_indices), -1)
        ax.set_yticks(np.arange(len(trial_indices)))

        response_tick_labels = [f"T{trial_i + 1} | {int(responses[trial_i])} {RESPONSE_LABELS.get(int(responses[trial_i]), 'Unknown')}" for trial_i in trial_indices]
        ax.set_yticklabels(response_tick_labels, fontsize=8)
        ax.set_title(f"Lick raster - {CONDITION_LABELS[condition_name].replace(chr(10), ' ')}")
        ax.set_xlabel("Time from odor cue onset (sec)")
        ax.set_ylabel("Trial | Response")

        handles, labels = ax.get_legend_handles_labels()
        unique_labels = dict(zip(labels, handles))
        if len(unique_labels) > 0:
            ax.legend(unique_labels.values(), unique_labels.keys(), frameon=False, loc='upper right')

        plt.tight_layout()
        fig.savefig(join(raster_save_path, f"lick_raster_{condition_name}.png"), dpi=300)
        fig.savefig(join(raster_save_path, f"lick_raster_{condition_name}.pdf"))
        plt.close(fig)


# ============================================================
# Per-mouse processing
# ============================================================

def process_mouse_data(mouse, main_folder, save_root):
    save_path = join(save_root, mouse)
    os.makedirs(save_path, exist_ok=True)

    print(f"Processing data for mouse {mouse}")

    try:
        paths_ml = glob.glob(join(main_folder, f'*{mouse}*.mat'))
        if not paths_ml:
            print(f"No .mat file found for mouse {mouse} in {main_folder}")
            return
        if len(paths_ml) > 1:
            print(f"More than one .mat file found for mouse {mouse}: Using the first.")
        path = paths_ml[0]

        (
            conditions_played,
            responses,
            licks_output,
            cues_timing,
            rewards_number,
            rewards_times,
            opto_timing,
            cue_offset_sec,
            trial_start_times_ml
        ) = load_mat_file(path)

        conditions_played = conditions_played[:-1]
        responses = responses[:-1]
        cues_timing = cues_timing[:-1]
        rewards_number = rewards_number[:-1]
        rewards_times = rewards_times[:-1]
        opto_timing = opto_timing[:-1]

        condition_indices = find_conditions(conditions_played)

        print(f"\nCondition trial counts for {mouse}:")
        for condition_name, indices in condition_indices.items():
            print(condition_name, len(indices))

        all_trials, trials_licks_by_condition, raw_licks_by_condition = get_licks_data(
            condition_indices,
            licks_output
        )

        with open(join(save_path, 'all_trials_licks.pkl'), 'wb') as f:
            pickle.dump(all_trials, f)
        with open(join(save_path, 'trials_licks_by_condition.pkl'), 'wb') as f:
            pickle.dump(trials_licks_by_condition, f)
        with open(join(save_path, 'raw_licks_by_condition.pkl'), 'wb') as f:
            pickle.dump(raw_licks_by_condition, f)
        with open(join(save_path, 'condition_indices.pkl'), 'wb') as f:
            pickle.dump(condition_indices, f)
        with open(join(save_path, 'licking_data.pkl'), 'wb') as f:
            pickle.dump(licks_output, f)

        np.save(join(save_path, 'conditions_played.npy'), conditions_played)
        np.save(join(save_path, 'responses.npy'), responses)

        with open(join(save_path, 'cues_timing.pkl'), 'wb') as f:
            pickle.dump(cues_timing, f)
        with open(join(save_path, 'rewards_number.pkl'), 'wb') as f:
            pickle.dump(rewards_number, f)
        with open(join(save_path, 'rewards_times.pkl'), 'wb') as f:
            pickle.dump(rewards_times, f)
        with open(join(save_path, 'opto_timing.pkl'), 'wb') as f:
            pickle.dump(opto_timing, f)
        with open(join(save_path, 'cue_offset_sec.pkl'), 'wb') as f:
            pickle.dump(cue_offset_sec, f)
        with open(join(save_path, 'trial_start_times_ml.pkl'), 'wb') as f:
            pickle.dump(trial_start_times_ml, f)

        plot_lick_raster_by_condition(
            licks_output=licks_output,
            condition_indices=condition_indices,
            cues_timing=cues_timing,
            rewards_times=rewards_times,
            opto_timing=opto_timing,
            cue_offset_sec=cue_offset_sec,
            responses=responses,
            save_path=save_path,
            photometry_hz=photometry_hz,
            monkey_logic_hz=monkey_logic_hz,
            max_time_sec=max_time_sec_for_raster
        )

        print(f"Finished processing {mouse}. Files saved in: {save_path}")

    except Exception as e:
        print(f"An error occurred while processing mouse {mouse}: {e}")


# ============================================================
# Response-rate Analysis (Original functionality)
# ============================================================

def calculate_response_rate_for_mouse(
        responses,
        condition_indices
):
    results = []
    responses = np.asarray(responses)

    for condition_name in CONDITION_ORDER:
        if condition_name not in condition_indices:
            continue

        trial_indices = list(condition_indices[condition_name])
        trial_indices = [i for i in trial_indices if i < len(responses)]

        target_response = TARGET_RESPONSE_BY_CONDITION[condition_name]
        target_response_label = TARGET_RESPONSE_LABEL_BY_CONDITION[condition_name]

        if len(trial_indices) == 0:
            results.append({
                "condition": condition_name,
                "condition_label": CONDITION_LABELS[condition_name],
                "target_response": target_response,
                "target_response_label": target_response_label,
                "n_trials": 0,
                "n_target_response": np.nan,
                "response_rate": np.nan,
                "response_rate_percent": np.nan,
                "n_miss": np.nan,
                "n_hit": np.nan,
                "n_correct_reject": np.nan,
                "n_false_alarm": np.nan
            })
            continue

        condition_responses = responses[trial_indices]
        n_trials = len(condition_responses)

        n_miss = np.sum(condition_responses == 2)
        n_hit = np.sum(condition_responses == 3)
        n_correct_reject = np.sum(condition_responses == 4)
        n_false_alarm = np.sum(condition_responses == 5)

        n_target_response = np.sum(condition_responses == target_response)
        response_rate = n_target_response / n_trials
        response_rate_percent = response_rate * 100

        results.append({
            "condition": condition_name,
            "condition_label": CONDITION_LABELS[condition_name],
            "target_response": target_response,
            "target_response_label": target_response_label,
            "n_trials": n_trials,
            "n_target_response": n_target_response,
            "response_rate": response_rate,
            "response_rate_percent": response_rate_percent,
            "n_miss": n_miss,
            "n_hit": n_hit,
            "n_correct_reject": n_correct_reject,
            "n_false_alarm": n_false_alarm
        })

    return pd.DataFrame(results)


def holm_bonferroni(p_values):
    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.full_like(p_values, np.nan)
    valid = ~np.isnan(p_values)
    p_valid = p_values[valid]

    if len(p_valid) == 0:
        return adjusted

    order = np.argsort(p_valid)
    sorted_p = p_valid[order]
    m = len(sorted_p)

    sorted_adjusted = np.array([(m - i) * sorted_p[i] for i in range(m)])
    sorted_adjusted = np.maximum.accumulate(sorted_adjusted)
    sorted_adjusted = np.minimum(sorted_adjusted, 1)

    adjusted_valid = np.empty_like(sorted_adjusted)
    adjusted_valid[order] = sorted_adjusted
    adjusted[valid] = adjusted_valid
    return adjusted


def compare_conditions_to_reference(
        plot_df,
        group_save_path,
        reference_condition=REFERENCE_CONDITION,
        value_col="response_rate_percent",
        test="wilcoxon",
        alpha=0.05
):
    df = plot_df.copy()
    df["condition"] = df["condition"].astype(str)

    pivot_df = df.pivot_table(
        index="mouse",
        columns="condition",
        values=value_col,
        aggfunc="mean"
    )

    if reference_condition not in pivot_df.columns:
        raise ValueError(
            f"Reference condition '{reference_condition}' was not found. "
            f"Available conditions: {list(pivot_df.columns)}"
        )

    results = []
    for condition_name in CONDITION_ORDER:
        if condition_name == reference_condition:
            continue

        if condition_name not in pivot_df.columns:
            results.append({
                "condition": condition_name,
                "condition_label": CONDITION_LABELS[condition_name],
                "reference_condition": reference_condition,
                "reference_label": CONDITION_LABELS[reference_condition],
                "test": test,
                "n_mice": 0,
                "mean_condition": np.nan,
                "mean_reference": np.nan,
                "mean_difference": np.nan,
                "statistic": np.nan,
                "p_value": np.nan
            })
            continue

        paired_df = pivot_df[[condition_name, reference_condition]].dropna()
        condition_values = paired_df[condition_name].values
        reference_values = paired_df[reference_condition].values
        n_mice = len(paired_df)

        if n_mice < 2:
            statistic = np.nan
            p_value = np.nan
        else:
            differences = condition_values - reference_values
            if test == "wilcoxon":
                if np.allclose(differences, 0):
                    statistic = 0
                    p_value = 1.0
                else:
                    statistic, p_value = wilcoxon(
                        condition_values,
                        reference_values,
                        alternative="two-sided",
                        zero_method="wilcox"
                    )
            elif test == "ttest":
                statistic, p_value = ttest_rel(
                    condition_values,
                    reference_values,
                    nan_policy="omit"
                )
            else:
                raise ValueError("test must be either 'wilcoxon' or 'ttest'")

        results.append({
            "condition": condition_name,
            "condition_label": CONDITION_LABELS[condition_name],
            "reference_condition": reference_condition,
            "reference_label": CONDITION_LABELS[reference_condition],
            "test": test,
            "n_mice": n_mice,
            "mean_condition": np.nanmean(condition_values) if n_mice > 0 else np.nan,
            "mean_reference": np.nanmean(reference_values) if n_mice > 0 else np.nan,
            "mean_difference": np.nanmean(condition_values - reference_values) if n_mice > 0 else np.nan,
            "statistic": statistic,
            "p_value": p_value
        })

    stats_df = pd.DataFrame(results)
    stats_df["p_adj_holm"] = holm_bonferroni(stats_df["p_value"].values)
    stats_df["significant"] = stats_df["p_adj_holm"] < alpha

    def p_to_stars(p):
        if np.isnan(p):
            return "n/a"
        elif p < 0.001:
            return "***"
        elif p < 0.01:
            return "**"
        elif p < 0.05:
            return "*"
        else:
            return "ns"

    stats_df["stars"] = stats_df["p_adj_holm"].apply(p_to_stars)
    stats_df.to_csv(
        join(group_save_path, f"stats_vs_{reference_condition}_{test}.csv"),
        index=False
    )
    return stats_df, pivot_df


def add_significance_annotations_vs_reference(
        ax,
        stats_df,
        reference_condition=REFERENCE_CONDITION,
        y_start=104,
        y_step=6
):
    if reference_condition not in CONDITION_ORDER:
        return

    reference_x = CONDITION_ORDER.index(reference_condition)
    current_y = y_start

    for _, row in stats_df.iterrows():
        if not row["significant"]:
            continue

        condition_name = row["condition"]
        if condition_name not in CONDITION_ORDER:
            continue

        condition_x = CONDITION_ORDER.index(condition_name)
        x_left = min(condition_x, reference_x)
        x_right = max(condition_x, reference_x)

        ax.plot(
            [x_left, x_left, x_right, x_right],
            [current_y, current_y + 1.5, current_y + 1.5, current_y],
            color="black",
            linewidth=1
        )
        ax.text(
            (x_left + x_right) / 2,
            current_y + 1.8,
            row["stars"],
            ha="center",
            va="bottom",
            fontsize=10
        )
        current_y += y_step

    ax.set_ylim(0, max(110, current_y + 4))


def plot_response_rate_from_responses_across_mice(
        mice,
        save_root,
        group_save_path,
        reference_condition=REFERENCE_CONDITION,
        test="wilcoxon",
        alpha=0.05
):
    os.makedirs(group_save_path, exist_ok=True)
    all_mouse_results = []

    for mouse in mice:
        mouse_path = join(save_root, mouse)
        if not os.path.exists(mouse_path):
            continue

        try:
            responses = np.load(join(mouse_path, 'responses.npy'))
            with open(join(mouse_path, 'condition_indices.pkl'), 'rb') as f:
                condition_indices = pickle.load(f)
        except FileNotFoundError:
            continue

        mouse_df = calculate_response_rate_for_mouse(
            responses=responses,
            condition_indices=condition_indices
        )
        mouse_df["mouse"] = mouse
        all_mouse_results.append(mouse_df)

    if len(all_mouse_results) == 0:
        return

    all_mouse_results = pd.concat(all_mouse_results, ignore_index=True)
    all_mouse_results.to_csv(join(group_save_path, "response_rate_by_mouse.csv"), index=False)

    plot_df = all_mouse_results[all_mouse_results["condition"].isin(CONDITION_ORDER)].copy()
    plot_df["condition"] = pd.Categorical(plot_df["condition"], categories=CONDITION_ORDER, ordered=True)
    plot_df = plot_df.sort_values("condition")

    stats_df, pivot_df = compare_conditions_to_reference(
        plot_df=plot_df,
        group_save_path=group_save_path,
        reference_condition=reference_condition,
        value_col="response_rate_percent",
        test=test,
        alpha=alpha
    )

    summary_df = (
        plot_df
        .groupby("condition", observed=True)
        .agg(
            mean_response_rate_percent=("response_rate_percent", "mean"),
            sem_response_rate_percent=(
                "response_rate_percent",
                lambda x: (
                    np.nanstd(x, ddof=1) / np.sqrt(np.sum(~np.isnan(x)))
                    if np.sum(~np.isnan(x)) > 1
                    else 0
                )
            )
        )
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    x_positions = np.arange(len(CONDITION_ORDER))
    means = []
    sems = []
    labels = []

    for condition_name in CONDITION_ORDER:
        condition_summary = summary_df[summary_df["condition"] == condition_name]
        if len(condition_summary) == 0:
            means.append(np.nan)
            sems.append(0)
        else:
            means.append(condition_summary["mean_response_rate_percent"].iloc[0])
            sems.append(condition_summary["sem_response_rate_percent"].iloc[0])

        labels.append(f"{CONDITION_LABELS[condition_name]}\n({TARGET_RESPONSE_LABEL_BY_CONDITION[condition_name]})")

    ax.bar(x_positions, means, yerr=sems, capsize=5, alpha=0.6, edgecolor='black', label='Mean ± SEM')

    # Connecting lines for each mouse
    for mouse in plot_df["mouse"].unique():
        mouse_df = plot_df[plot_df["mouse"] == mouse].copy()
        mouse_df["condition"] = pd.Categorical(mouse_df["condition"], categories=CONDITION_ORDER, ordered=True)
        mouse_df = mouse_df.sort_values("condition")

        x_mouse = []
        y_mouse = []
        for condition_name in CONDITION_ORDER:
            condition_row = mouse_df[mouse_df["condition"] == condition_name]
            if len(condition_row) == 0:
                continue
            value = condition_row["response_rate_percent"].iloc[0]
            if np.isnan(value):
                continue
            x_mouse.append(CONDITION_ORDER.index(condition_name))
            y_mouse.append(value)

        if len(x_mouse) > 1:
            ax.plot(x_mouse, y_mouse, color="gray", alpha=0.4, linewidth=1, zorder=2)

    # Individual dots
    for condition_idx, condition_name in enumerate(CONDITION_ORDER):
        condition_mouse_df = plot_df[plot_df["condition"] == condition_name]
        y_values = condition_mouse_df["response_rate_percent"].values
        jitter = np.random.uniform(low=-0.08, high=0.08, size=len(y_values))
        ax.scatter(np.full(len(y_values), condition_idx) + jitter, y_values, color='gray', edgecolor='black', alpha=0.9, zorder=3)

    # Value labels
    for x, mean in zip(x_positions, means):
        label = "NaN" if np.isnan(mean) else f"{mean:.1f}%"
        y = 1 if np.isnan(mean) else mean + 2
        ax.text(x, y, label, ha='center', va='bottom', fontsize=9)

    add_significance_annotations_vs_reference(ax=ax, stats_df=stats_df, reference_condition=reference_condition)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.set_ylabel("Response rate (%)")
    ax.set_xlabel("Condition")
    ax.set_title(f"Response rate across mice\nCompared to {CONDITION_LABELS[reference_condition].replace(chr(10), ' ')}")
    ax.legend(frameon=False)

    plt.tight_layout()
    fig.savefig(join(group_save_path, "response_rate_from_responses_across_mice.png"), dpi=300)
    fig.savefig(join(group_save_path, "response_rate_from_responses_across_mice.pdf"))
    plt.close(fig)


# ============================================================
# Response-rate comparison (First 1/3 vs Last 1/3 of trials)
# ============================================================

def calculate_thirds_response_rate_for_mouse(
        responses,
        condition_indices
):
    """
    Calculate response rate for the first 1/3 and last 1/3 of trials
    for each condition for a single mouse.
    """
    results = []
    responses = np.asarray(responses)

    for condition_name in CONDITION_ORDER:
        if condition_name not in condition_indices:
            continue

        trial_indices = list(condition_indices[condition_name])
        trial_indices = [i for i in trial_indices if i < len(responses)]
        n_trials = len(trial_indices)

        target_response = TARGET_RESPONSE_BY_CONDITION[condition_name]
        target_response_label = TARGET_RESPONSE_LABEL_BY_CONDITION[condition_name]

        # Require at least 3 trials to divide into thirds
        if n_trials < 3:
            results.append({
                "condition": condition_name,
                "condition_label": CONDITION_LABELS[condition_name],
                "target_response": target_response,
                "target_response_label": target_response_label,
                "n_trials_total": n_trials,
                "n_trials_third": 0,
                "first_rate_percent": np.nan,
                "last_rate_percent": np.nan
            })
            continue

        n_third = n_trials // 3
        first_trials = trial_indices[:n_third]
        last_trials = trial_indices[-n_third:]

        first_responses = responses[first_trials]
        last_responses = responses[last_trials]

        first_hits = np.sum(first_responses == target_response)
        first_rate_percent = (first_hits / len(first_responses)) * 100

        last_hits = np.sum(last_responses == target_response)
        last_rate_percent = (last_hits / len(last_responses)) * 100

        results.append({
            "condition": condition_name,
            "condition_label": CONDITION_LABELS[condition_name],
            "target_response": target_response,
            "target_response_label": target_response_label,
            "n_trials_total": n_trials,
            "n_trials_third": n_third,
            "first_rate_percent": first_rate_percent,
            "last_rate_percent": last_rate_percent
        })

    return pd.DataFrame(results)


def plot_thirds_response_rate_comparison_across_mice(
        mice,
        save_root,
        group_save_path
):
    """
    Plot response rates comparing the first 1/3 to the last 1/3 of trials.
    Clustered bar plot: 2 bars per condition (First 1/3 and Last 1/3).
    Individual mouse values are shown as dots connected by lines.
    """
    os.makedirs(group_save_path, exist_ok=True)
    all_mouse_results = []

    for mouse in mice:
        mouse_path = join(save_root, mouse)
        if not os.path.exists(mouse_path):
            continue

        try:
            responses = np.load(join(mouse_path, 'responses.npy'))
            with open(join(mouse_path, 'condition_indices.pkl'), 'rb') as f:
                condition_indices = pickle.load(f)
        except FileNotFoundError:
            continue

        mouse_df = calculate_thirds_response_rate_for_mouse(
            responses=responses,
            condition_indices=condition_indices
        )
        mouse_df["mouse"] = mouse
        all_mouse_results.append(mouse_df)

    if len(all_mouse_results) == 0:
        print("No mouse data found. Could not create thirds comparison plot.")
        return

    all_mouse_results = pd.concat(all_mouse_results, ignore_index=True)
    all_mouse_results.to_csv(join(group_save_path, "response_rate_thirds_by_mouse.csv"), index=False)

    summary_df = (
        all_mouse_results
        .groupby("condition", observed=True)
        .agg(
            mean_first=("first_rate_percent", "mean"),
            sem_first=(
                "first_rate_percent",
                lambda x: (
                    np.nanstd(x, ddof=1) / np.sqrt(np.sum(~np.isnan(x)))
                    if np.sum(~np.isnan(x)) > 1
                    else 0
                )
            ),
            mean_last=("last_rate_percent", "mean"),
            sem_last=(
                "last_rate_percent",
                lambda x: (
                    np.nanstd(x, ddof=1) / np.sqrt(np.sum(~np.isnan(x)))
                    if np.sum(~np.isnan(x)) > 1
                    else 0
                )
            )
        )
        .reset_index()
    )
    summary_df.to_csv(join(group_save_path, "response_rate_thirds_group_summary.csv"), index=False)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 7))

    width = 0.35
    x_positions = np.arange(len(CONDITION_ORDER))

    means_first = []
    sems_first = []
    means_last = []
    sems_last = []
    labels = []

    for condition_name in CONDITION_ORDER:
        cond_sum = summary_df[summary_df["condition"] == condition_name]
        if len(cond_sum) == 0:
            means_first.append(np.nan)
            sems_first.append(0)
            means_last.append(np.nan)
            sems_last.append(0)
        else:
            means_first.append(cond_sum["mean_first"].iloc[0])
            sems_first.append(cond_sum["sem_first"].iloc[0])
            means_last.append(cond_sum["mean_last"].iloc[0])
            sems_last.append(cond_sum["sem_last"].iloc[0])
        labels.append(f"{CONDITION_LABELS[condition_name]}\n({TARGET_RESPONSE_LABEL_BY_CONDITION[condition_name]})")

    # Clustered Bars
    ax.bar(x_positions - width/2, means_first, width=width, yerr=sems_first, capsize=4,
           color='#a2d2ff', edgecolor='black', alpha=0.7, label='First 1/3')
    ax.bar(x_positions + width/2, means_last, width=width, yerr=sems_last, capsize=4,
           color='#0f4c5c', edgecolor='black', alpha=0.8, label='Last 1/3')

    # Individual mice dots and pairing lines
    rng = np.random.default_rng(100)
    for mouse in all_mouse_results["mouse"].unique():
        mouse_df = all_mouse_results[all_mouse_results["mouse"] == mouse]
        
        for k, condition_name in enumerate(CONDITION_ORDER):
            cond_row = mouse_df[mouse_df["condition"] == condition_name]
            if len(cond_row) == 0:
                continue
                
            y_first = cond_row["first_rate_percent"].iloc[0]
            y_last = cond_row["last_rate_percent"].iloc[0]
            
            if np.isnan(y_first) or np.isnan(y_last):
                continue
                
            jitter = rng.uniform(-0.06, 0.06)
            x_f = k - width/2 + jitter
            x_l = k + width/2 + jitter
            
            # Connect the dots for the same mouse
            ax.plot([x_f, x_l], [y_first, y_last], color='gray', alpha=0.35, linewidth=0.8, zorder=2)
            
            # Draw individual dots
            ax.scatter(x_f, y_first, color='gray', edgecolor='black', s=45, alpha=0.7, zorder=3)
            ax.scatter(x_l, y_last, color='gray', edgecolor='black', s=45, alpha=0.7, zorder=3)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=25, ha='right')
    ax.set_ylabel("Response rate (%)", fontsize=11)
    ax.set_xlabel("Condition", fontsize=11)
    ax.set_title("Response Rate Comparison: First 1/3 vs Last 1/3 of Session", fontsize=13, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.legend(frameon=False, loc='upper right')

    plt.tight_layout()
    fig.savefig(join(group_save_path, "response_rate_thirds_comparison.png"), dpi=300)
    fig.savefig(join(group_save_path, "response_rate_thirds_comparison.pdf"))
    plt.close(fig)

    print(f"Saved thirds comparison plot in: {group_save_path}")


def calculate_thirds_dprime_for_mouse(
        responses,
        condition_indices
):
    results = []
    responses = np.asarray(responses)

    if "no_opto" not in condition_indices:
        return pd.DataFrame(results)

    go_trials = list(condition_indices["no_opto"])
    go_trials = [i for i in go_trials if i < len(responses)]

    if len(go_trials) < 3:
        return pd.DataFrame(results)

    n_third_go = len(go_trials) // 3
    first_go = go_trials[:n_third_go]
    last_go = go_trials[-n_third_go:]

    first_go_resp = responses[first_go]
    last_go_resp = responses[last_go]

    def calc_dprime(go_resp, nogo_resp):
        hits = np.sum(go_resp == 3)
        misses = np.sum(go_resp == 2)
        fas = np.sum(nogo_resp == 5)
        crs = np.sum(nogo_resp == 4)

        hr = (hits + 0.5) / (hits + misses + 1)
        far = (fas + 0.5) / (fas + crs + 1)

        return stats.norm.ppf(hr) - stats.norm.ppf(far)

    for condition_name in CONDITION_ORDER:
        if condition_name == "no_opto":
            continue

        if condition_name not in condition_indices:
            continue

        nogo_trials = list(condition_indices[condition_name])
        nogo_trials = [i for i in nogo_trials if i < len(responses)]

        if len(nogo_trials) < 3:
            continue

        n_third_nogo = len(nogo_trials) // 3
        first_nogo = nogo_trials[:n_third_nogo]
        last_nogo = nogo_trials[-n_third_nogo:]

        first_nogo_resp = responses[first_nogo]
        last_nogo_resp = responses[last_nogo]

        dprime_first = calc_dprime(first_go_resp, first_nogo_resp)
        dprime_last = calc_dprime(last_go_resp, last_nogo_resp)

        results.append({
            "condition": condition_name,
            "dprime_first": dprime_first,
            "dprime_last": dprime_last
        })

    return pd.DataFrame(results)


def plot_thirds_dprime_comparison_across_mice(
        mice,
        save_root,
        group_save_path
):
    os.makedirs(group_save_path, exist_ok=True)
    all_mouse_results = []

    for mouse in mice:
        mouse_path = join(save_root, mouse)
        if not os.path.exists(mouse_path):
            continue

        try:
            responses = np.load(join(mouse_path, 'responses.npy'))
            with open(join(mouse_path, 'condition_indices.pkl'), 'rb') as f:
                condition_indices = pickle.load(f)
        except FileNotFoundError:
            continue

        mouse_df = calculate_thirds_dprime_for_mouse(
            responses=responses,
            condition_indices=condition_indices
        )
        if len(mouse_df) > 0:
            mouse_df["mouse"] = mouse
            all_mouse_results.append(mouse_df)

    if len(all_mouse_results) == 0:
        print("No mouse data found. Could not create thirds d' comparison plot.")
        return

    df = pd.concat(all_mouse_results, ignore_index=True)

    summary_df = (
        df
        .groupby("condition", observed=True)
        .agg(
            mean_first=("dprime_first", "mean"),
            sem_first=("dprime_first", lambda x: np.nanstd(x, ddof=1) / np.sqrt(np.sum(~np.isnan(x))) if np.sum(~np.isnan(x)) > 1 else 0),
            mean_last=("dprime_last", "mean"),
            sem_last=("dprime_last", lambda x: np.nanstd(x, ddof=1) / np.sqrt(np.sum(~np.isnan(x))) if np.sum(~np.isnan(x)) > 1 else 0)
        )
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    delay_conditions = [c for c in CONDITION_ORDER if c != "no_opto"]
    x_positions = np.arange(len(delay_conditions))
    width = 0.35

    means_first = []
    sems_first = []
    means_last = []
    sems_last = []
    labels = []

    for cond in delay_conditions:
        cond_sum = summary_df[summary_df["condition"] == cond]
        if len(cond_sum) == 0:
            means_first.append(np.nan)
            sems_first.append(0)
            means_last.append(np.nan)
            sems_last.append(0)
        else:
            means_first.append(cond_sum["mean_first"].iloc[0])
            sems_first.append(cond_sum["sem_first"].iloc[0])
            means_last.append(cond_sum["mean_last"].iloc[0])
            sems_last.append(cond_sum["sem_last"].iloc[0])
        labels.append(f"{CONDITION_LABELS[cond]}\n({TARGET_RESPONSE_LABEL_BY_CONDITION[cond]})")

    ax.bar(x_positions - width/2, means_first, width=width, yerr=sems_first, capsize=4,
           color='#a2d2ff', edgecolor='black', alpha=0.7, label='First 1/3')
    ax.bar(x_positions + width/2, means_last, width=width, yerr=sems_last, capsize=4,
           color='#0f4c5c', edgecolor='black', alpha=0.8, label='Last 1/3')

    rng = np.random.default_rng(100)

    for mouse in df["mouse"].unique():
        mdf = df[df["mouse"] == mouse]
        for i, cond in enumerate(delay_conditions):
            r = mdf[mdf["condition"] == cond]
            if len(r) == 0:
                continue
            v_first = r["dprime_first"].iloc[0]
            v_last = r["dprime_last"].iloc[0]

            jitter = rng.uniform(-0.06, 0.06)
            x1 = i - width/2 + jitter
            x2 = i + width/2 + jitter

            ax.plot([x1, x2], [v_first, v_last], color="gray", alpha=0.4, linewidth=1, zorder=2)
            ax.scatter(x1, v_first, color="gray", edgecolor="black", alpha=0.9, zorder=3)
            ax.scatter(x2, v_last, color="gray", edgecolor="black", alpha=0.9, zorder=3)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Performance (d')")
    ax.set_title("d' First 1/3 vs Last 1/3")
    ax.legend(frameon=False)
    ax.axhline(0, color='black', linestyle=':', alpha=0.3)

    plt.tight_layout()
    fig.savefig(join(group_save_path, "thirds_dprime_comparison.png"), dpi=300)
    fig.savefig(join(group_save_path, "thirds_dprime_comparison.pdf"))
    plt.close(fig)


def plot_thirds_hit_fa_comparison_across_mice(
        mice,
        save_root,
        group_save_path
):
    os.makedirs(group_save_path, exist_ok=True)
    all_mouse_results = []

    for mouse in mice:
        mouse_path = join(save_root, mouse)
        if not os.path.exists(mouse_path):
            continue

        try:
            responses = np.load(join(mouse_path, 'responses.npy'))
            with open(join(mouse_path, 'condition_indices.pkl'), 'rb') as f:
                condition_indices = pickle.load(f)
        except FileNotFoundError:
            continue

        mouse_df = calculate_thirds_response_rate_for_mouse(
            responses=responses,
            condition_indices=condition_indices
        )
        if len(mouse_df) > 0:
            mouse_df["mouse"] = mouse
            all_mouse_results.append(mouse_df)

    if len(all_mouse_results) == 0:
        print("No mouse data found. Could not create thirds Hit/FA comparison plot.")
        return

    df = pd.concat(all_mouse_results, ignore_index=True)

    summary_df = (
        df
        .groupby("condition", observed=True)
        .agg(
            mean_first=("first_rate_percent", "mean"),
            sem_first=("first_rate_percent", lambda x: np.nanstd(x, ddof=1) / np.sqrt(np.sum(~np.isnan(x))) if np.sum(~np.isnan(x)) > 1 else 0),
            mean_last=("last_rate_percent", "mean"),
            sem_last=("last_rate_percent", lambda x: np.nanstd(x, ddof=1) / np.sqrt(np.sum(~np.isnan(x))) if np.sum(~np.isnan(x)) > 1 else 0)
        )
        .reset_index()
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [1, 5]})
    width = 0.35
    rng = np.random.default_rng(100)

    # LEFT PLOT: Hit rate (no_opto)
    hit_cond = "no_opto"
    hit_sum = summary_df[summary_df["condition"] == hit_cond]
    if len(hit_sum) > 0:
        mf = hit_sum["mean_first"].iloc[0]
        sf = hit_sum["sem_first"].iloc[0]
        ml = hit_sum["mean_last"].iloc[0]
        sl = hit_sum["sem_last"].iloc[0]

        ax1.bar([0 - width/2], [mf], width=width, yerr=[sf], capsize=4, color='#a2d2ff', edgecolor='black', label='First 1/3')
        ax1.bar([0 + width/2], [ml], width=width, yerr=[sl], capsize=4, color='#0f4c5c', edgecolor='black', label='Last 1/3')

        for mouse in df["mouse"].unique():
            mr = df[(df["mouse"] == mouse) & (df["condition"] == hit_cond)]
            if len(mr) == 0: continue
            v_first = mr["first_rate_percent"].iloc[0]
            v_last = mr["last_rate_percent"].iloc[0]
            
            jitter = rng.uniform(-0.06, 0.06)
            x1 = -width/2 + jitter
            x2 = width/2 + jitter
            
            ax1.plot([x1, x2], [v_first, v_last], color="gray", alpha=0.4, linewidth=1, zorder=2)
            ax1.scatter(x1, v_first, color="gray", edgecolor="black", alpha=0.9, zorder=3)
            ax1.scatter(x2, v_last, color="gray", edgecolor="black", alpha=0.9, zorder=3)

        ax1.set_xticks([0])
        ax1.set_xticklabels([CONDITION_LABELS[hit_cond]])
        ax1.set_ylabel("Hit Rate (%)")
        ax1.set_title("Hit Rate")
        ax1.set_ylim(-5, 105)

    # RIGHT PLOT: FA rate (delay conditions)
    delay_conditions = [c for c in CONDITION_ORDER if c != "no_opto"]
    x_positions = np.arange(len(delay_conditions))

    means_first = []
    sems_first = []
    means_last = []
    sems_last = []
    labels = []

    for cond in delay_conditions:
        cond_sum = summary_df[summary_df["condition"] == cond]
        if len(cond_sum) == 0:
            means_first.append(np.nan)
            sems_first.append(0)
            means_last.append(np.nan)
            sems_last.append(0)
        else:
            means_first.append(cond_sum["mean_first"].iloc[0])
            sems_first.append(cond_sum["sem_first"].iloc[0])
            means_last.append(cond_sum["mean_last"].iloc[0])
            sems_last.append(cond_sum["sem_last"].iloc[0])
        labels.append(CONDITION_LABELS[cond])

    ax2.bar(x_positions - width/2, means_first, width=width, yerr=sems_first, capsize=4, color='#a2d2ff', edgecolor='black')
    ax2.bar(x_positions + width/2, means_last, width=width, yerr=sems_last, capsize=4, color='#0f4c5c', edgecolor='black')

    for mouse in df["mouse"].unique():
        for i, cond in enumerate(delay_conditions):
            mr = df[(df["mouse"] == mouse) & (df["condition"] == cond)]
            if len(mr) == 0: continue
            v_first = mr["first_rate_percent"].iloc[0]
            v_last = mr["last_rate_percent"].iloc[0]
            
            jitter = rng.uniform(-0.06, 0.06)
            x1 = i - width/2 + jitter
            x2 = i + width/2 + jitter
            
            ax2.plot([x1, x2], [v_first, v_last], color="gray", alpha=0.4, linewidth=1, zorder=2)
            ax2.scatter(x1, v_first, color="gray", edgecolor="black", alpha=0.9, zorder=3)
            ax2.scatter(x2, v_last, color="gray", edgecolor="black", alpha=0.9, zorder=3)

    ax2.set_xticks(x_positions)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("FA Rate (%)")
    ax2.set_title("FA Rate")
    ax2.set_ylim(-5, 105)

    fig.legend(['First 1/3', 'Last 1/3'], loc='upper right', bbox_to_anchor=(0.95, 0.95), frameon=False)
    plt.tight_layout()
    fig.savefig(join(group_save_path, "thirds_hit_fa_comparison.png"), dpi=300)
    fig.savefig(join(group_save_path, "thirds_hit_fa_comparison.pdf"))
    plt.close(fig)


# ============================================================
# Run
# ============================================================

if __name__ == '__main__':

    for mouse in mice:
        process_mouse_data(
            mouse=mouse,
            main_folder=MAIN_FOLDER,
            save_root=SAVE_ROOT
        )

    GROUP_SAVE_PATH = join(SAVE_ROOT, "group_summary")

    plot_response_rate_from_responses_across_mice(
        mice=mice,
        save_root=SAVE_ROOT,
        group_save_path=GROUP_SAVE_PATH,
        reference_condition=REFERENCE_CONDITION,
        test="wilcoxon",
        alpha=0.05
    )

    plot_thirds_response_rate_comparison_across_mice(
        mice=mice,
        save_root=SAVE_ROOT,
        group_save_path=GROUP_SAVE_PATH
    )

    plot_thirds_dprime_comparison_across_mice(
        mice=mice,
        save_root=SAVE_ROOT,
        group_save_path=GROUP_SAVE_PATH
    )

    plot_thirds_hit_fa_comparison_across_mice(
        mice=mice,
        save_root=SAVE_ROOT,
        group_save_path=GROUP_SAVE_PATH
    )