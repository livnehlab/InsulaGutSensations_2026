"""
Revision experiment: detection to reversal.

In the reversal sessions the contingency is reversed (15 Hz opto: reward upon lick;
no opto: no reward), and the response codes are relabeled accordingly. For each mouse,
loads the MonkeyLogic files and computes the response rate of each condition, rolling
metrics within the sessions, first vs. last third comparisons, and overall d' and
hit / FA comparisons across mice.
"""

import os
import glob
import pickle
import numpy as np
import pandas as pd
import scipy.io
import scipy.stats as stats
import matplotlib

matplotlib.use('Qt5Agg')
matplotlib.rcParams['pdf.fonttype'] = 42

import matplotlib.pyplot as plt

from os.path import join


# ---------------------------------------------------------------------------
# Data locations (set these before running)
# ---------------------------------------------------------------------------
BEHAVIOR_ROOT = r'path\to\behavior_data'  # folder with the behavioral data (Dprime/, Revision_experiments/)


# ============================================================
# User settings
# ============================================================

mice = ['mouse22', 'mouse23', 'mouse25', 'mouse28', 'mouse29', 'mouse30']  # mice to analyze

MAIN_FOLDER = BEHAVIOR_ROOT + r'\Revision_experiments\Detection to reversal'
SAVE_ROOT = BEHAVIOR_ROOT + r'\Revision_experiments\Detection to reversal\processed_data5'
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

REVERSAL_15HZ_REWARD = 1
REVERSAL_NO_OPTO_NO_REWARD = 2
HZ15_NO_REWARD = 3
NO_OPTO_REWARD = 4

CONDITION_CODES = {
    "reversal_15hz_reward": REVERSAL_15HZ_REWARD,
    "reversal_no_opto_no_reward": REVERSAL_NO_OPTO_NO_REWARD,
    "hz15_no_reward": HZ15_NO_REWARD,
    "no_opto_reward": NO_OPTO_REWARD,
}

# Plot order:
# first pair = original
# second pair = reversal
CONDITION_ORDER = [
    "hz15_no_reward",
    "no_opto_reward",
    "reversal_15hz_reward",
    "reversal_no_opto_no_reward",
]

CONDITION_LABELS = {
    "reversal_15hz_reward": "15Hz\nreward upon lick",
    "reversal_no_opto_no_reward": "No opto\nno reward",
    "hz15_no_reward": "15Hz\nno reward",
    "no_opto_reward": "No opto\nreward upon lick",
}

# Creates visual space between the original pair and reversal pair
CONDITION_X_POSITIONS = {
    "no_opto_reward": 0,
    "hz15_no_reward": 1,
    "reversal_15hz_reward": 3,
    "reversal_no_opto_no_reward": 4,
}


# ============================================================
# Response-code interpretation
# ============================================================

# Original response-code meaning:
# 2 = Miss
# 3 = Hit
# 4 = CR
# 5 = FA
STANDARD_RESPONSE_LABELS = {
    2: "Miss",
    3: "Hit",
    4: "CR",
    5: "FA",
}

# Reversal response-code meaning:
# 2 = CR
# 3 = FA
# 4 = Miss
# 5 = Hit
REVERSAL_RESPONSE_LABELS = {
    2: "CR",
    3: "FA",
    4: "Miss",
    5: "Hit",
}

REVERSAL_CONDITIONS = {
    "reversal_15hz_reward",
    "reversal_no_opto_no_reward",
}

INVERSE_CONDITION_CODES = {
    REVERSAL_15HZ_REWARD: "reversal_15hz_reward",
    REVERSAL_NO_OPTO_NO_REWARD: "reversal_no_opto_no_reward",
    HZ15_NO_REWARD: "hz15_no_reward",
    NO_OPTO_REWARD: "no_opto_reward",
}


def get_response_label(response_code, condition_name):
    """
    Return the correct response label depending on whether the trial
    belongs to a reversal condition or an original condition.
    """

    response_code = int(response_code)

    if condition_name in REVERSAL_CONDITIONS:
        return REVERSAL_RESPONSE_LABELS.get(response_code, "Unknown")

    return STANDARD_RESPONSE_LABELS.get(response_code, "Unknown")


# ============================================================
# Target response codes for licking response rate
# ============================================================

# Goal of the conclusion plot:
# Plot "licking response rate" in each condition.
#
# Reversal:
# Condition 1: Reversal 15Hz, reward upon licking
#   Opto requires licking.
#   In reversal coding, Hit = code 5.
#
# Condition 2: Reversal No opto, no reward
#   No opto requires no licking.
#   In reversal coding, FA = code 3.
#
# Original:
# Condition 3: 15Hz, no reward
#   Opto requires no licking.
#   In original coding, FA = code 5.
#
# Condition 4: No opto, reward upon licking
#   No opto requires licking.
#   In original coding, Hit = code 3.

TARGET_RESPONSE_BY_CONDITION = {
    "reversal_15hz_reward": 5,          # reversal Hit
    "reversal_no_opto_no_reward": 3,    # reversal FA
    "hz15_no_reward": 5,                # original FA
    "no_opto_reward": 3,                # original Hit
}

TARGET_RESPONSE_LABEL_BY_CONDITION = {
    "reversal_15hz_reward": "Lick",
    "reversal_no_opto_no_reward": "Lick",
    "hz15_no_reward": "Lick",
    "no_opto_reward": "Lick",
}


# ============================================================
# Basic helpers
# ============================================================

def reduce_hz(plot):
    """Downsample MonkeyLogic 1000 Hz lick data to photometry_hz."""

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
        condition_name: pd.DataFrame(
            [trials_licks[i] for i in indices if i < len(trials_licks)]
        )
        for condition_name, indices in condition_indices.items()
    }

    raw_licks_by_condition = {
        condition_name: [
            licks_output[i] for i in indices if i < len(licks_output)
        ]
        for condition_name, indices in condition_indices.items()
    }

    return all_trials, trials_licks_by_condition, raw_licks_by_condition


# ============================================================
# Load and align MonkeyLogic data
# ============================================================

def cues_time(data, n_cues):
    """
    Find odor cue timing, reward timing, and opto timing in each trial.

    opto_timing per trial:
    (
        opto_start_time,
        opto_stop_time,
        opto_start_relative_to_odor,
        opto_stop_relative_to_odor
    )
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

    Trial start:
    - If there is opto: 1 sec before first 30
    - If there is no opto: 5 sec before odor cue onset, code 23

    Returns:
    - reshaped_trials
    - cue_offset_sec
    - trial_start_times_ml
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

    # Keep one fewer trial, as in the original extraction logic
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
    """
    Plot lick rasters separately for each condition.

    X axis:
    - 0 = odor cue onset
    - negative values = time before odor
    - positive values = time after odor

    Y axis:
    - each row is one trial
    - label includes trial number, raw TrialErrors code, and condition-specific label
    """

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

        ax.axvline(
            0,
            linestyle='-',
            color='black',
            alpha=0.8,
            label='Odor onset'
        )

        odor_ends = []

        for trial_i in trial_indices:
            cue_start = cues_timing[trial_i][0]
            cue_end = cues_timing[trial_i][1]
            odor_ends.append((cue_end - cue_start) / monkey_logic_hz)

        odor_end_sec = np.nanmedian(odor_ends)

        ax.axvspan(
            0,
            odor_end_sec,
            alpha=0.2,
            color='gray',
            label='Odor cue'
        )

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

            ax.axvline(
                median_opto_start,
                linestyle='--',
                color='red',
                alpha=0.8,
                label='Median opto start'
            )

            ax.axvline(
                median_opto_end,
                linestyle='--',
                color='darkred',
                alpha=0.8,
                label='Median opto end'
            )

            ax.axvspan(
                median_opto_start,
                median_opto_end,
                color='red',
                alpha=0.08
            )

            ax.text(
                x=(median_opto_start + median_opto_end) / 2,
                y=-0.5,
                s=f'Opto: {median_opto_duration:.2f} sec',
                ha='center',
                va='bottom',
                color='red'
            )

        reward_label_added = False

        for row_i, trial_i in enumerate(trial_indices):
            trial_reward_times = rewards_times[trial_i]

            if len(trial_reward_times) > 0:
                reward_times_sec = np.asarray(trial_reward_times) / monkey_logic_hz

                ax.scatter(
                    reward_times_sec,
                    np.full(len(reward_times_sec), row_i),
                    marker='|',
                    s=80,
                    color='green',
                    label='Reward' if not reward_label_added else None
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

        ax.set_title(f"Lick raster - {CONDITION_LABELS[condition_name].replace(chr(10), ' ')}")
        ax.set_xlabel("Time from odor cue onset (sec)")
        ax.set_ylabel("Trial | Response")

        ax.set_ylim(len(trial_indices), -1)

        ax.set_yticks(np.arange(len(trial_indices)))

        response_tick_labels = []

        for trial_i in trial_indices:
            response_code = int(responses[trial_i])
            response_name = get_response_label(response_code, condition_name)

            response_tick_labels.append(
                f"T{trial_i + 1} | {response_code} {response_name}"
            )

        ax.set_yticklabels(response_tick_labels, fontsize=8)

        handles, labels = ax.get_legend_handles_labels()
        unique_labels = dict(zip(labels, handles))

        if len(unique_labels) > 0:
            ax.legend(
                unique_labels.values(),
                unique_labels.keys(),
                frameon=False,
                loc='upper right'
            )

        plt.tight_layout()

        fig.savefig(
            join(raster_save_path, f"lick_raster_{condition_name}.png"),
            dpi=300
        )

        fig.savefig(
            join(raster_save_path, f"lick_raster_{condition_name}.pdf")
        )

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
            print(f"More than one .mat file found for mouse {mouse}:")
            for p in paths_ml:
                print(p)
            print("Using the first file only.")

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

        # licks_output is one trial shorter because of the extraction logic
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
# Response-rate Analysis
# ============================================================

def calculate_response_rate_for_mouse(
        responses,
        condition_indices
):
    """
    Calculate licking response rate per condition using TrialErrors.

    Response-code meanings differ between reversal and original conditions.

    Reversal:
    2 = CR
    3 = FA
    4 = Miss
    5 = Hit

    Original:
    2 = Miss
    3 = Hit
    4 = CR
    5 = FA

    Licking response rate:
    - Reversal 15Hz reward: Hit = code 5
    - Reversal no opto no reward: FA = code 3
    - Original 15Hz no reward: FA = code 5
    - Original no opto reward: Hit = code 3
    """

    results = []

    responses = np.asarray(responses)

    for condition_name in CONDITION_ORDER:
        if condition_name not in condition_indices:
            continue

        trial_indices = list(condition_indices[condition_name])

        trial_indices = [
            i for i in trial_indices
            if i < len(responses)
        ]

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

        response_labels = np.asarray([
            get_response_label(response_code, condition_name)
            for response_code in condition_responses
        ])

        n_trials = len(condition_responses)

        n_miss = np.sum(response_labels == "Miss")
        n_hit = np.sum(response_labels == "Hit")
        n_correct_reject = np.sum(response_labels == "CR")
        n_false_alarm = np.sum(response_labels == "FA")

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


def plot_response_rate_reversal_original_across_mice(
        mice,
        save_root,
        group_save_path
):
    """
    Plot licking response rate across mice for the reversal/original experiment.

    Plot layout:
    - first pair: reversal conditions
    - second pair: original conditions

    Each mouse:
    - one dot per condition
    - connecting line across all 4 conditions
    """

    os.makedirs(group_save_path, exist_ok=True)

    all_mouse_results = []

    for mouse in mice:
        mouse_path = join(save_root, mouse)

        if not os.path.exists(mouse_path):
            print(f"Skipping {mouse}: folder not found")
            continue

        try:
            responses = np.load(join(mouse_path, 'responses.npy'))

            with open(join(mouse_path, 'condition_indices.pkl'), 'rb') as f:
                condition_indices = pickle.load(f)

        except FileNotFoundError as e:
            print(f"Skipping {mouse}: missing file {e}")
            continue

        mouse_df = calculate_response_rate_for_mouse(
            responses=responses,
            condition_indices=condition_indices
        )

        mouse_df["mouse"] = mouse
        all_mouse_results.append(mouse_df)

    if len(all_mouse_results) == 0:
        print("No mouse data found. Could not create response-rate plot.")
        return

    plot_df = pd.concat(all_mouse_results, ignore_index=True)

    plot_df.to_csv(
        join(group_save_path, "licking_response_rate_by_mouse.csv"),
        index=False
    )

    plot_df = plot_df[
        plot_df["condition"].isin(CONDITION_ORDER)
    ].copy()

    plot_df["condition"] = pd.Categorical(
        plot_df["condition"],
        categories=CONDITION_ORDER,
        ordered=True
    )

    plot_df = plot_df.sort_values("condition")

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
            ),
            n_mice=("response_rate_percent", lambda x: np.sum(~np.isnan(x))),
            total_trials=("n_trials", "sum"),
            total_target_responses=("n_target_response", "sum"),
            total_hits=("n_hit", "sum"),
            total_correct_rejects=("n_correct_reject", "sum"),
            total_false_alarms=("n_false_alarm", "sum")
        )
        .reset_index()
    )

    summary_df.to_csv(
        join(group_save_path, "licking_response_rate_group_summary.csv"),
        index=False
    )

    fig, ax = plt.subplots(figsize=(8, 6))

    x_positions = [
        CONDITION_X_POSITIONS[condition_name]
        for condition_name in CONDITION_ORDER
    ]

    means = []
    sems = []
    x_tick_labels = []

    for condition_name in CONDITION_ORDER:
        condition_summary = summary_df[
            summary_df["condition"] == condition_name
        ]

        if len(condition_summary) == 0:
            means.append(np.nan)
            sems.append(0)
        else:
            means.append(
                condition_summary["mean_response_rate_percent"].iloc[0]
            )
            sems.append(
                condition_summary["sem_response_rate_percent"].iloc[0]
            )

        x_tick_labels.append(CONDITION_LABELS[condition_name])

    ax.bar(
        x_positions,
        means,
        yerr=sems,
        capsize=5,
        width=0.75,
        alpha=0.6,
        edgecolor='black',
        label='Mean ± SEM',
        zorder=1
    )

    # Deterministic jitter so each mouse line connects to its own dots
    rng = np.random.default_rng(1)

    mouse_condition_x = {}

    for mouse in plot_df["mouse"].unique():
        mouse_condition_x[mouse] = {}

        for condition_name in CONDITION_ORDER:
            base_x = CONDITION_X_POSITIONS[condition_name]
            mouse_condition_x[mouse][condition_name] = base_x + rng.uniform(-0.08, 0.08)

    # Connecting lines for each mouse across all 4 conditions
    for mouse in plot_df["mouse"].unique():
        mouse_df = plot_df[plot_df["mouse"] == mouse].copy()

        mouse_df["condition"] = pd.Categorical(
            mouse_df["condition"],
            categories=CONDITION_ORDER,
            ordered=True
        )

        mouse_df = mouse_df.sort_values("condition")

        x_mouse = []
        y_mouse = []

        for condition_name in CONDITION_ORDER:
            condition_row = mouse_df[
                mouse_df["condition"] == condition_name
            ]

            if len(condition_row) == 0:
                continue

            value = condition_row["response_rate_percent"].iloc[0]

            if np.isnan(value):
                continue

            x_mouse.append(mouse_condition_x[mouse][condition_name])
            y_mouse.append(value)

        if len(x_mouse) > 1:
            ax.plot(
                x_mouse,
                y_mouse,
                color="gray",
                alpha=0.4,
                linewidth=1,
                zorder=2
            )

    # Individual mouse dots
    for condition_name in CONDITION_ORDER:
        condition_mouse_df = plot_df[
            plot_df["condition"] == condition_name
        ]

        for _, row in condition_mouse_df.iterrows():
            mouse = row["mouse"]
            y_value = row["response_rate_percent"]

            if np.isnan(y_value):
                continue

            ax.scatter(
                mouse_condition_x[mouse][condition_name],
                y_value,
                color='gray',
                edgecolor='black',
                alpha=0.9,
                zorder=3
            )

    # Value labels above bars
    for x, mean in zip(x_positions, means):
        if np.isnan(mean):
            label = "NaN"
            y = 1
        else:
            label = f"{mean:.1f}%"
            y = mean + 2

        ax.text(
            x,
            y,
            label,
            ha='center',
            va='bottom',
            fontsize=9
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_tick_labels, rotation=0, ha='center')

    # Group labels under x axis
    ax.text(
        0.5,
        -0.22,
        "Original",
        ha='center',
        va='top',
        transform=ax.get_xaxis_transform(),
        fontsize=11,
        fontweight='bold'
    )

    ax.text(
        3.5,
        -0.22,
        "Reversal",
        ha='center',
        va='top',
        transform=ax.get_xaxis_transform(),
        fontsize=11,
        fontweight='bold'
    )
    # Visual separator between the two pairs
    ax.axvline(
        2,
        color='black',
        linestyle=':',
        alpha=0.3,
        linewidth=1
    )

    ax.set_xlim(-0.75, 4.75)
    ax.set_ylim(0, 100)

    ax.set_ylabel("Licking response rate (%)")
    ax.set_xlabel("")
    ax.set_title("Licking response rate across mice: reversal vs original")

    ax.legend(frameon=False, loc='upper right')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.28)

    fig.savefig(
        join(group_save_path, "licking_response_rate_reversal_original_across_mice.png"),
        dpi=300
    )

    fig.savefig(
        join(group_save_path, "licking_response_rate_reversal_original_across_mice.pdf")
    )

    plt.close(fig)

    print(f"Saved reversal/original licking response-rate plot in: {group_save_path}")


# ============================================================
# Behavioral metric calculations (Decoded Responses)
# ============================================================

def decode_responses(responses_subset, conditions_subset):
    """
    Decode response codes into 'Hit', 'Miss', 'CR', or 'FA'
    based on the condition of each trial (reversal vs original).
    """
    decoded = []
    for r, c in zip(responses_subset, conditions_subset):
        cond_name = INVERSE_CONDITION_CODES.get(c, "Unknown")
        decoded.append(get_response_label(r, cond_name))
    return np.array(decoded)


def calculate_dprime_decoded(decoded_labels):
    """
    Calculate d' for decoded response labels using the Hautus correction.
    """
    if len(decoded_labels) == 0:
        return np.nan

    hits = np.sum(decoded_labels == "Hit")
    misses = np.sum(decoded_labels == "Miss")
    crs = np.sum(decoded_labels == "CR")
    fas = np.sum(decoded_labels == "FA")

    go_trials = hits + misses
    nogo_trials = fas + crs

    hit_rate = (hits + 0.5) / (go_trials + 1)
    fa_rate = (fas + 0.5) / (nogo_trials + 1)

    d_prime = stats.norm.ppf(hit_rate) - stats.norm.ppf(fa_rate)
    return d_prime


def calculate_observed_hit_rate_decoded(decoded_labels):
    """
    Calculate Hit rate for decoded labels: Hits / (Hits + Misses)
    """
    hits = np.sum(decoded_labels == "Hit")
    misses = np.sum(decoded_labels == "Miss")
    go_trials = hits + misses
    return hits / go_trials if go_trials > 0 else np.nan


def calculate_observed_fa_rate_decoded(decoded_labels):
    """
    Calculate FA rate for decoded labels: FAs / (FAs + CRs)
    """
    fas = np.sum(decoded_labels == "FA")
    crs = np.sum(decoded_labels == "CR")
    nogo_trials = fas + crs
    return fas / nogo_trials if nogo_trials > 0 else np.nan


def calculate_rolling_metric(responses_subset, conditions_subset, window_size, metric_type):
    """
    Calculate rolling metric values by decoding trials in a sliding window.
    """
    n_trials = len(responses_subset)
    if n_trials < window_size:
        return np.array([])

    rolling_vals = []
    for i in range(n_trials - window_size + 1):
        w_resp = responses_subset[i : i + window_size]
        w_cond = conditions_subset[i : i + window_size]
        decoded = decode_responses(w_resp, w_cond)
        
        if metric_type == 'dprime':
            val = calculate_dprime_decoded(decoded)
        elif metric_type == 'hit_rate':
            val = calculate_observed_hit_rate_decoded(decoded)
        elif metric_type == 'fa_rate':
            val = calculate_observed_fa_rate_decoded(decoded)
        else:
            val = np.nan
            
        rolling_vals.append(val)
        
    return np.array(rolling_vals)


def calculate_nan_sem(matrix, axis=0):
    """
    Robust SEM calculation across columns/rows ignoring NaNs.
    For columns with 0 or 1 valid entry, SEM is returned as NaN.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        n_non_nan = np.sum(~np.isnan(matrix), axis=axis)
        std = np.nanstd(matrix, axis=axis, ddof=1)
        sem = std / np.sqrt(n_non_nan)
        sem[n_non_nan <= 1] = np.nan
        return sem


# ============================================================
# Reversal group plotting function
# ============================================================

def plot_metric_analysis_across_mice(
        mice,
        save_root,
        group_save_path,
        metric_name='dprime',
        window_size=20
):
    """
    Unified group plotting function for reversal analysis:
    - Left: Dot + SEM and individual mice dots for the original contingency.
    - Right: Line plot of rolling metric (+ SEM shaded area) for the reversal contingency.
    
    Supports: 'dprime', 'hit_rate', 'fa_rate'
    Uses centered alignment with full integer X values (offset = window_size // 2 - 1).
    """
    os.makedirs(group_save_path, exist_ok=True)

    # Configure metric parameters
    if metric_name == 'dprime':
        single_calc = calculate_dprime_decoded
        ylabel = "Performance (d')"
        title_suffix = "Original vs Reversal Performance (d')"
        filename = "reversal_dprime_analysis_across_mice"
        ylim = None
        hline_val = 0
    elif metric_name == 'hit_rate':
        single_calc = calculate_observed_hit_rate_decoded
        ylabel = "Hit Rate"
        title_suffix = "Original vs Reversal Hit Rate"
        filename = "reversal_hit_rate_analysis_across_mice"
        ylim = (-0.05, 1.05)
        hline_val = None
    elif metric_name == 'fa_rate':
        single_calc = calculate_observed_fa_rate_decoded
        ylabel = "False Alarm Rate"
        title_suffix = "Original vs Reversal False Alarm Rate"
        filename = "reversal_fa_rate_analysis_across_mice"
        ylim = (-0.05, 1.05)
        hline_val = None
    else:
        raise ValueError(f"Unknown metric_name: {metric_name}")

    all_orig_vals = []
    all_rolling_vals = []
    mouse_names = []

    for mouse in mice:
        mouse_path = join(save_root, mouse)
        if not os.path.exists(mouse_path):
            print(f"Skipping {mouse} for {metric_name} Analysis: folder not found")
            continue

        try:
            responses = np.load(join(mouse_path, 'responses.npy'))
            conditions_played = np.load(join(mouse_path, 'conditions_played.npy'))
        except FileNotFoundError as e:
            print(f"Skipping {mouse} for {metric_name} Analysis: missing files: {e}")
            continue

        # 1. Original contingency trials (HZ15_NO_REWARD & NO_OPTO_REWARD)
        original_mask = (conditions_played == HZ15_NO_REWARD) | (conditions_played == NO_OPTO_REWARD)
        orig_decoded = decode_responses(responses[original_mask], conditions_played[original_mask])
        orig_val = single_calc(orig_decoded)

        # 2. Reversal contingency trials (REVERSAL_15HZ_REWARD & REVERSAL_NO_OPTO_NO_REWARD)
        reversal_mask = (conditions_played == REVERSAL_15HZ_REWARD) | (conditions_played == REVERSAL_NO_OPTO_NO_REWARD)
        rolling_val = calculate_rolling_metric(
            responses[reversal_mask],
            conditions_played[reversal_mask],
            window_size,
            metric_name
        )

        all_orig_vals.append(orig_val)
        all_rolling_vals.append(rolling_val)
        mouse_names.append(mouse)

    if len(all_orig_vals) == 0:
        print(f"No mouse data found. Could not create {metric_name} plot.")
        return

    # Export original values to CSV
    orig_df = pd.DataFrame({
        "mouse": mouse_names,
        f"original_{metric_name}": all_orig_vals
    })
    orig_df.to_csv(join(group_save_path, f"reversal_{metric_name}_original_by_mouse.csv"), index=False)

    # Padding rolling values with NaNs
    max_roll_len = max(len(arr) for arr in all_rolling_vals) if len(all_rolling_vals) > 0 else 0

    padded_rolling = []
    for arr in all_rolling_vals:
        if len(arr) < max_roll_len:
            padded = np.pad(arr, (0, max_roll_len - len(arr)), mode='constant', constant_values=np.nan)
        else:
            padded = arr
        padded_rolling.append(padded)

    if max_roll_len > 0:
        roll_matrix = np.vstack(padded_rolling)
        # Export rolling values to CSV
        roll_df = pd.DataFrame(
            roll_matrix,
            index=mouse_names,
            columns=[f"window_{i+1}" for i in range(max_roll_len)]
        )
        roll_df.to_csv(join(group_save_path, f"reversal_{metric_name}_rolling_new_odor_by_mouse.csv"))
    else:
        roll_matrix = np.empty((len(all_orig_vals), 0))

    # Calculate summaries
    all_orig_vals = np.asarray(all_orig_vals)
    mean_orig = np.nanmean(all_orig_vals)
    sem_orig = np.nanstd(all_orig_vals, ddof=1) / np.sqrt(np.sum(~np.isnan(all_orig_vals))) if np.sum(~np.isnan(all_orig_vals)) > 1 else 0

    mean_roll = np.nanmean(roll_matrix, axis=0) if max_roll_len > 0 else np.array([])
    sem_roll = calculate_nan_sem(roll_matrix, axis=0) if max_roll_len > 0 else np.array([])

    # Plot
    fig = plt.figure(figsize=(11, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 3.5], wspace=0.15)

    ax_orig = fig.add_subplot(gs[0])
    ax_roll = fig.add_subplot(gs[1], sharey=ax_orig)

    # --- Left panel (Original Contingency) ---
    rng = np.random.default_rng(42)
    jitter = rng.uniform(-0.12, 0.12, size=len(all_orig_vals))
    x_orig = np.zeros(len(all_orig_vals)) + jitter

    # Plot individual mouse dots
    ax_orig.scatter(
        x_orig,
        all_orig_vals,
        color='gray',
        edgecolor='black',
        alpha=0.6,
        s=60,
        zorder=3,
        label='Individual mice'
    )

    # Plot average across mice + SEM
    ax_orig.errorbar(
        0,
        mean_orig,
        yerr=sem_orig,
        fmt='o',
        color='#0f4c5c',
        ecolor='#0f4c5c',
        elinewidth=2.5,
        capsize=5,
        markersize=10,
        zorder=4,
        label='Average ± SEM'
    )

    ax_orig.set_xlim(-0.5, 0.5)
    ax_orig.set_xticks([0])
    ax_orig.set_xticklabels(['Original'], fontsize=11)
    ax_orig.set_ylabel(ylabel, fontsize=12)
    if ylim is not None:
        ax_orig.set_ylim(ylim)
    if hline_val is not None:
        ax_orig.axhline(hline_val, color='gray', linestyle='--', alpha=0.5, zorder=1)
    ax_orig.spines['top'].set_visible(False)
    ax_orig.spines['right'].set_visible(False)
    ax_orig.legend(frameon=False, loc='upper right', fontsize=9)

    # --- Right panel (Reversal Rolling) ---
    if max_roll_len > 0:
        # Centered alignment: for window size = 20, first point represents trials 1-20 centered at Trial 10 (index 1 + 9)
        x_roll = np.arange(1, max_roll_len + 1) + (window_size // 2 - 1)

        # Plot individual mice lines as subtle grey traces
        for i, arr in enumerate(padded_rolling):
            ax_roll.plot(
                x_roll,
                arr,
                color='gray',
                alpha=0.2,
                linewidth=1,
                zorder=1
            )

        # Plot group average line
        ax_roll.plot(
            x_roll,
            mean_roll,
            color='#0f4c5c',
            linewidth=2.5,
            zorder=3,
            label='Average'
        )

        # Plot shaded SEM area
        ax_roll.fill_between(
            x_roll,
            mean_roll - sem_roll,
            mean_roll + sem_roll,
            color='#0f4c5c',
            alpha=0.15,
            zorder=2,
            label='± SEM'
        )

    if hline_val is not None:
        ax_roll.axhline(hline_val, color='gray', linestyle='--', alpha=0.5, zorder=1)
    ax_roll.set_xlabel(f"Trial of reversal phase (center of {window_size}-trial window)", fontsize=11)
    ax_roll.set_title(f"Reversal {ylabel.lower()} learning curve", fontsize=12)
    ax_roll.spines['top'].set_visible(False)
    ax_roll.spines['right'].set_visible(False)
    ax_roll.legend(frameon=False, loc='upper left', fontsize=9)

    # Remove y ticks for shared y axis
    plt.setp(ax_roll.get_yticklabels(), visible=False)
    ax_roll.yaxis.tick_left()

    # Title and cleanup
    fig.suptitle(title_suffix, fontsize=14, y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)

    fig.savefig(join(group_save_path, f"{filename}.png"), dpi=300)
    fig.savefig(join(group_save_path, f"{filename}.pdf"))
    plt.close(fig)

    print(f"Saved {metric_name} Analysis plots in: {group_save_path}")

def plot_hit_fa_thirds_comparison(mice, save_root, group_save_path):
    """
    Plots Hit Rate and FA Rate.
    Left Panel: Original trials.
    Right Panel: Reversal trials (First 1/3 vs Last 1/3).
    """
    os.makedirs(group_save_path, exist_ok=True)
    
    records = []
    
    for mouse in mice:
        mouse_path = join(save_root, mouse)
        if not os.path.exists(mouse_path):
            continue
            
        try:
            responses = np.load(join(mouse_path, 'responses.npy'))
            conditions_played = np.load(join(mouse_path, 'conditions_played.npy'))
        except FileNotFoundError:
            continue
            
        # 1. Original
        orig_mask = (conditions_played == HZ15_NO_REWARD) | (conditions_played == NO_OPTO_REWARD)
        orig_resp = responses[orig_mask]
        orig_cond = conditions_played[orig_mask]
        orig_decoded = decode_responses(orig_resp, orig_cond)
        
        orig_hit_rate = calculate_observed_hit_rate_decoded(orig_decoded)
        orig_fa_rate = calculate_observed_fa_rate_decoded(orig_decoded)
        
        # 2. Reversal
        rev_mask = (conditions_played == REVERSAL_15HZ_REWARD) | (conditions_played == REVERSAL_NO_OPTO_NO_REWARD)
        rev_resp = responses[rev_mask]
        rev_cond = conditions_played[rev_mask]
        rev_decoded = decode_responses(rev_resp, rev_cond)
        
        n_rev = len(rev_decoded)
        if n_rev > 0:
            third_idx = n_rev // 3
            first_third_decoded = rev_decoded[:third_idx]
            last_third_decoded = rev_decoded[-third_idx:] if third_idx > 0 else []
            
            first_hit_rate = calculate_observed_hit_rate_decoded(first_third_decoded)
            first_fa_rate = calculate_observed_fa_rate_decoded(first_third_decoded)
            
            last_hit_rate = calculate_observed_hit_rate_decoded(last_third_decoded)
            last_fa_rate = calculate_observed_fa_rate_decoded(last_third_decoded)
        else:
            first_hit_rate = np.nan
            first_fa_rate = np.nan
            last_hit_rate = np.nan
            last_fa_rate = np.nan
            
        records.append({
            'mouse': mouse,
            'orig_hit': orig_hit_rate,
            'orig_fa': orig_fa_rate,
            'first_hit': first_hit_rate,
            'first_fa': first_fa_rate,
            'last_hit': last_hit_rate,
            'last_fa': last_fa_rate
        })
        
    df = pd.DataFrame(records)
    if len(df) == 0:
        print("No mouse data found for Hit/FA thirds comparison.")
        return
        
    df.to_csv(join(group_save_path, "reversal_hit_fa_rates_thirds_by_mouse.csv"), index=False)
        
    fig, ax = plt.subplots(figsize=(10, 6))
    
    rng = np.random.default_rng(42)
    jitter = rng.uniform(-0.05, 0.05, size=len(df))
    
    hit_color = '#2ca02c' # green
    fa_color = '#d62728'  # red
    
    x_orig_hit = np.zeros(len(df)) + jitter
    x_orig_fa = np.ones(len(df)) + jitter
    
    x_first_hit = np.zeros(len(df)) + 2.5 + jitter
    x_first_fa = np.ones(len(df)) + 2.5 + jitter
    
    x_last_hit = np.zeros(len(df)) + 5.0 + jitter
    x_last_fa = np.ones(len(df)) + 5.0 + jitter
    
    for i, row in df.iterrows():
        # Original
        ax.plot([x_orig_hit[i], x_orig_fa[i]], [row['orig_hit'], row['orig_fa']], color='gray', alpha=0.4)
        ax.scatter(x_orig_hit[i], row['orig_hit'], color='gray', alpha=0.6, s=30)
        ax.scatter(x_orig_fa[i], row['orig_fa'], color='gray', alpha=0.6, s=30)
        
        # First 1/3
        ax.plot([x_first_hit[i], x_first_fa[i]], [row['first_hit'], row['first_fa']], color='gray', alpha=0.4)
        ax.scatter(x_first_hit[i], row['first_hit'], color='gray', alpha=0.6, s=30)
        ax.scatter(x_first_fa[i], row['first_fa'], color='gray', alpha=0.6, s=30)
        
        # Last 1/3
        ax.plot([x_last_hit[i], x_last_fa[i]], [row['last_hit'], row['last_fa']], color='gray', alpha=0.4)
        ax.scatter(x_last_hit[i], row['last_hit'], color='gray', alpha=0.6, s=30)
        ax.scatter(x_last_fa[i], row['last_fa'], color='gray', alpha=0.6, s=30)
        
    ax.errorbar(0, df['orig_hit'].mean(), yerr=df['orig_hit'].sem(), fmt='o', color=hit_color, markersize=10, capsize=5, label='Hit Rate')
    ax.errorbar(1, df['orig_fa'].mean(), yerr=df['orig_fa'].sem(), fmt='o', color=fa_color, markersize=10, capsize=5, label='FA Rate')
    
    ax.errorbar(2.5, df['first_hit'].mean(), yerr=df['first_hit'].sem(), fmt='o', color=hit_color, markersize=10, capsize=5)
    ax.errorbar(3.5, df['first_fa'].mean(), yerr=df['first_fa'].sem(), fmt='o', color=fa_color, markersize=10, capsize=5)
    
    ax.errorbar(5.0, df['last_hit'].mean(), yerr=df['last_hit'].sem(), fmt='o', color=hit_color, markersize=10, capsize=5)
    ax.errorbar(6.0, df['last_fa'].mean(), yerr=df['last_fa'].sem(), fmt='o', color=fa_color, markersize=10, capsize=5)
    
    ax.set_xticks([0, 1, 2.5, 3.5, 5.0, 6.0])
    ax.set_xticklabels(['Hit Rate', 'FA Rate', 'Hit Rate', 'FA Rate', 'Hit Rate', 'FA Rate'])
    
    # Group labels
    ax.text(0.5, -0.15, "Original", ha='center', va='top', transform=ax.get_xaxis_transform(), fontsize=11, fontweight='bold')
    ax.text(3.0, -0.15, "First 1/3", ha='center', va='top', transform=ax.get_xaxis_transform(), fontsize=11, fontweight='bold')
    ax.text(5.5, -0.15, "Last 1/3", ha='center', va='top', transform=ax.get_xaxis_transform(), fontsize=11, fontweight='bold')
    
    ax.set_title('Reversal Hit/FA Rates Comparison')
    ax.set_ylabel('Rate')
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(-0.5, 6.5)
    
    # Visual separators
    ax.axvline(1.75, color='black', linestyle=':', alpha=0.3)
    ax.axvline(4.25, color='black', linestyle=':', alpha=0.3)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(frameon=False, loc='upper right', fontsize=9)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    fig.savefig(join(group_save_path, "reversal_hit_fa_rates_thirds_comparison.png"), dpi=300)
    fig.savefig(join(group_save_path, "reversal_hit_fa_rates_thirds_comparison.pdf"))
    plt.close(fig)
    print(f"Saved Reversal Hit/FA Thirds Comparison plot in: {group_save_path}")


def plot_overall_hit_fa_comparison(mice, save_root, group_save_path):
    """
    Plots Hit Rate and FA Rate overall.
    Left Panel: Original trials.
    Right Panel: Reversal trials.
    """
    os.makedirs(group_save_path, exist_ok=True)
    
    records = []
    
    for mouse in mice:
        mouse_path = join(save_root, mouse)
        if not os.path.exists(mouse_path):
            continue
            
        try:
            responses = np.load(join(mouse_path, 'responses.npy'))
            conditions_played = np.load(join(mouse_path, 'conditions_played.npy'))
        except FileNotFoundError:
            continue
            
        # 1. Original
        orig_mask = (conditions_played == HZ15_NO_REWARD) | (conditions_played == NO_OPTO_REWARD)
        orig_resp = responses[orig_mask]
        orig_cond = conditions_played[orig_mask]
        orig_decoded = decode_responses(orig_resp, orig_cond)
        
        orig_hit_rate = calculate_observed_hit_rate_decoded(orig_decoded)
        orig_fa_rate = calculate_observed_fa_rate_decoded(orig_decoded)
        
        # 2. Reversal
        rev_mask = (conditions_played == REVERSAL_15HZ_REWARD) | (conditions_played == REVERSAL_NO_OPTO_NO_REWARD)
        rev_resp = responses[rev_mask]
        rev_cond = conditions_played[rev_mask]
        rev_decoded = decode_responses(rev_resp, rev_cond)
        
        rev_hit_rate = calculate_observed_hit_rate_decoded(rev_decoded)
        rev_fa_rate = calculate_observed_fa_rate_decoded(rev_decoded)
            
        records.append({
            'mouse': mouse,
            'orig_hit': orig_hit_rate,
            'orig_fa': orig_fa_rate,
            'rev_hit': rev_hit_rate,
            'rev_fa': rev_fa_rate
        })
        
    df = pd.DataFrame(records)
    if len(df) == 0:
        print("No mouse data found for Overall Hit/FA comparison.")
        return
        
    df.to_csv(join(group_save_path, "reversal_hit_fa_rates_overall_by_mouse.csv"), index=False)
        
    fig, ax = plt.subplots(figsize=(8, 6))
    
    rng = np.random.default_rng(42)
    jitter = rng.uniform(-0.05, 0.05, size=len(df))
    
    hit_color = '#2ca02c' # green
    fa_color = '#d62728'  # red
    
    x_orig_hit = np.zeros(len(df)) + jitter
    x_orig_fa = np.ones(len(df)) + jitter
    
    x_rev_hit = np.zeros(len(df)) + 2.5 + jitter
    x_rev_fa = np.ones(len(df)) + 2.5 + jitter
    
    for i, row in df.iterrows():
        # Original
        ax.plot([x_orig_hit[i], x_orig_fa[i]], [row['orig_hit'], row['orig_fa']], color='gray', alpha=0.4)
        ax.scatter(x_orig_hit[i], row['orig_hit'], color='gray', alpha=0.6, s=30)
        ax.scatter(x_orig_fa[i], row['orig_fa'], color='gray', alpha=0.6, s=30)
        
        # Reversal
        ax.plot([x_rev_hit[i], x_rev_fa[i]], [row['rev_hit'], row['rev_fa']], color='gray', alpha=0.4)
        ax.scatter(x_rev_hit[i], row['rev_hit'], color='gray', alpha=0.6, s=30)
        ax.scatter(x_rev_fa[i], row['rev_fa'], color='gray', alpha=0.6, s=30)
        
    ax.errorbar(0, df['orig_hit'].mean(), yerr=df['orig_hit'].sem(), fmt='o', color=hit_color, markersize=10, capsize=5, label='Hit Rate')
    ax.errorbar(1, df['orig_fa'].mean(), yerr=df['orig_fa'].sem(), fmt='o', color=fa_color, markersize=10, capsize=5, label='FA Rate')
    
    ax.errorbar(2.5, df['rev_hit'].mean(), yerr=df['rev_hit'].sem(), fmt='o', color=hit_color, markersize=10, capsize=5)
    ax.errorbar(3.5, df['rev_fa'].mean(), yerr=df['rev_fa'].sem(), fmt='o', color=fa_color, markersize=10, capsize=5)
    
    ax.set_xticks([0, 1, 2.5, 3.5])
    ax.set_xticklabels(['Hit Rate', 'FA Rate', 'Hit Rate', 'FA Rate'])
    
    # Group labels
    ax.text(0.5, -0.15, "Original", ha='center', va='top', transform=ax.get_xaxis_transform(), fontsize=11, fontweight='bold')
    ax.text(3.0, -0.15, "Reversal", ha='center', va='top', transform=ax.get_xaxis_transform(), fontsize=11, fontweight='bold')
    
    ax.set_title('Overall Hit/FA Rates Comparison')
    ax.set_ylabel('Rate')
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(-0.5, 4.0)
    
    # Visual separators
    ax.axvline(1.75, color='black', linestyle=':', alpha=0.3)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(frameon=False, loc='upper right', fontsize=9)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    fig.savefig(join(group_save_path, "reversal_hit_fa_rates_overall_comparison.png"), dpi=300)
    fig.savefig(join(group_save_path, "reversal_hit_fa_rates_overall_comparison.pdf"))
    plt.close(fig)
    print(f"Saved Reversal Overall Hit/FA Comparison plot in: {group_save_path}")


def plot_overall_dprime_comparison(mice, save_root, group_save_path):
    """
    Plots overall d' for Original vs Reversal.
    """
    os.makedirs(group_save_path, exist_ok=True)
    
    records = []
    
    for mouse in mice:
        mouse_path = join(save_root, mouse)
        if not os.path.exists(mouse_path):
            continue
            
        try:
            responses = np.load(join(mouse_path, 'responses.npy'))
            conditions_played = np.load(join(mouse_path, 'conditions_played.npy'))
        except FileNotFoundError:
            continue
            
        # 1. Original
        orig_mask = (conditions_played == HZ15_NO_REWARD) | (conditions_played == NO_OPTO_REWARD)
        orig_resp = responses[orig_mask]
        orig_cond = conditions_played[orig_mask]
        orig_decoded = decode_responses(orig_resp, orig_cond)
        
        orig_dprime = calculate_dprime_decoded(orig_decoded)
        
        # 2. Reversal
        rev_mask = (conditions_played == REVERSAL_15HZ_REWARD) | (conditions_played == REVERSAL_NO_OPTO_NO_REWARD)
        rev_resp = responses[rev_mask]
        rev_cond = conditions_played[rev_mask]
        rev_decoded = decode_responses(rev_resp, rev_cond)
        
        rev_dprime = calculate_dprime_decoded(rev_decoded)
            
        records.append({
            'mouse': mouse,
            'orig_dprime': orig_dprime,
            'rev_dprime': rev_dprime
        })
        
    df = pd.DataFrame(records)
    if len(df) == 0:
        print("No mouse data found for Overall d' comparison.")
        return
        
    df.to_csv(join(group_save_path, "reversal_dprime_overall_by_mouse.csv"), index=False)

    fig, ax = plt.subplots(figsize=(6, 6))

    rng = np.random.default_rng(42)
    jitter = rng.uniform(-0.05, 0.05, size=len(df))

    dprime_color = '#1f77b4'

    x_orig = np.zeros(len(df)) + jitter
    x_rev = np.ones(len(df)) + jitter

    # Individual mice
    for i, row in df.iterrows():
        ax.plot(
            [x_orig[i], x_rev[i]],
            [row['orig_dprime'], row['rev_dprime']],
            color='gray',
            alpha=0.4
        )
        ax.scatter(
            x_orig[i],
            row['orig_dprime'],
            color='gray',
            alpha=0.6,
            s=30
        )
        ax.scatter(
            x_rev[i],
            row['rev_dprime'],
            color='gray',
            alpha=0.6,
            s=30
        )

    # Group means ± SEM
    ax.errorbar(
        0,
        df['orig_dprime'].mean(),
        yerr=df['orig_dprime'].sem(),
        fmt='o',
        color=dprime_color,
        markersize=10,
        capsize=5,
        label="Mean d'"
    )

    ax.errorbar(
        1,
        df['rev_dprime'].mean(),
        yerr=df['rev_dprime'].sem(),
        fmt='o',
        color=dprime_color,
        markersize=10,
        capsize=5
    )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Original', 'Reversal'], fontsize=12)

    ax.set_title("Overall d' Comparison")
    ax.set_ylabel("Performance (d')")
    ax.set_xlim(-0.5, 1.5)

    ax.axhline(0, color='black', linestyle=':', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    fig.savefig(join(group_save_path, "reversal_dprime_overall_comparison.png"), dpi=300)
    fig.savefig(join(group_save_path, "reversal_dprime_overall_comparison.pdf"))
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

    plot_response_rate_reversal_original_across_mice(
        mice=mice,
        save_root=SAVE_ROOT,
        group_save_path=GROUP_SAVE_PATH
    )

    plot_metric_analysis_across_mice(
        mice=mice,
        save_root=SAVE_ROOT,
        group_save_path=GROUP_SAVE_PATH,
        metric_name='dprime',
        window_size=20
    )

    plot_metric_analysis_across_mice(
        mice=mice,
        save_root=SAVE_ROOT,
        group_save_path=GROUP_SAVE_PATH,
        metric_name='hit_rate',
        window_size=20
    )

    plot_metric_analysis_across_mice(
        mice=mice,
        save_root=SAVE_ROOT,
        group_save_path=GROUP_SAVE_PATH,
        metric_name='fa_rate',
        window_size=20
    )
    
    plot_hit_fa_thirds_comparison(
        mice=mice,
        save_root=SAVE_ROOT,
        group_save_path=GROUP_SAVE_PATH
    )

    plot_overall_hit_fa_comparison(
        mice=mice,
        save_root=SAVE_ROOT,
        group_save_path=GROUP_SAVE_PATH
    )

    plot_overall_dprime_comparison(
        mice=mice,
        save_root=SAVE_ROOT,
        group_save_path=GROUP_SAVE_PATH
    )

