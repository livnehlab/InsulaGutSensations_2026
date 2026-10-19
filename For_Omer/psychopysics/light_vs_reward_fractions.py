import os
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.gridspec as gridspec
from os.path import join
from matplotlib.colors import TwoSlopeNorm
import seaborn as sns
import matplotlib.patches as mpatches
from scipy.stats import kruskal
import matplotlib.patches as patches
from scipy.stats import gaussian_kde
import random
from scipy.stats import pearsonr
from scipy.stats import sem
matplotlib.use('Qt5Agg')
matplotlib.rcParams['pdf.fonttype'] = 42


###############################################################################
# Helper functions
###############################################################################
def add_axvline_opto_seconds(ax, pre_trial_length, rec_hz):
    """
    Draw vertical lines (in seconds) on a plot, marking:
      - Distractor light (darkorange)
      - Opto stimulation (yellow)
      - Odor (blue)
    """
    ax.axvline(x=pre_trial_length / rec_hz, color='darkorange', linestyle='--', linewidth=1, label='Distractor light')
    ax.axvline(x=(pre_trial_length + 9 * rec_hz) / rec_hz, color='darkorange', linestyle='--', linewidth=1)
    ax.axvline(x=(pre_trial_length + 1 * rec_hz) / rec_hz, color='yellow', linestyle='--', linewidth=1,
               label='Opto stim')
    ax.axvline(x=(pre_trial_length + 1 * rec_hz + 4 * rec_hz) / rec_hz, color='yellow', linestyle='--', linewidth=1)
    ax.axvline(x=(pre_trial_length + 5 * rec_hz) / rec_hz, color='blue', linestyle='--', linewidth=1, label='Odor')
    ax.axvline(x=(pre_trial_length + 5 * rec_hz + 2 * rec_hz) / rec_hz, color='blue', linestyle='--', linewidth=1)


def add_axvline_opto_frames(ax, pre_trial_length, rec_hz):
    """
    Draw vertical lines on a plot whose x-axis is in frame units.
    The lines mark the same events, but now the axis is in frames (not seconds).
    """
    ax.axvline(x=pre_trial_length, color='darkorange', linestyle='--', linewidth=1, label='Distractor light')
    ax.axvline(x=(pre_trial_length + 9 * rec_hz), color='darkorange', linestyle='--', linewidth=1)
    ax.axvline(x=(pre_trial_length + 1 * rec_hz), color='yellow', linestyle='--', linewidth=1, label='Opto stim')
    ax.axvline(x=(pre_trial_length + 1 * rec_hz + 4 * rec_hz), color='yellow', linestyle='--', linewidth=1)
    ax.axvline(x=(pre_trial_length + 5 * rec_hz), color='blue', linestyle='--', linewidth=1, label='Odor')
    ax.axvline(x=(pre_trial_length + 5 * rec_hz + 2 * rec_hz), color='blue', linestyle='--', linewidth=1)
    ax.legend(loc='upper right')


def build_freq_indices(conditions_played, frequency_map, target_freqs):
    """
    Construct a dictionary mapping each frequency in frequency_map to the list of trial indices
    where that frequency (via condition analysis_code) appeared.

    Also filters out frequency 0 if present (keeping only frequencies in target_freqs).
    """
    freq_indices = {}
    for code, freq in frequency_map.items():
        inds = [i for i, x in enumerate(conditions_played) if x == code]
        if freq in freq_indices:
            freq_indices[freq].extend(inds)
        else:
            freq_indices[freq] = inds

    # Keep only target frequencies
    freq_indices = {freq: inds for freq, inds in freq_indices.items() if freq in target_freqs}
    return freq_indices


def load_dff_run_list_sub(dff_file):
    """Load the normalized dF/F data from pickle."""
    with open(dff_file, 'rb') as f:
        return pickle.load(f)


def load_event_data(load_loc, target_freqs):
    """
    For each frequency in target_freqs, load:
      - all_light_averages_<freq>.pkl
      - res<freq>.pkl (all light vs. baseline)
      - res_all_light_vs_lightHit_<freq>.pkl (all light vs. hit)
    Return dicts: event_data_by_freq, results_by_freq, results_vs_hit_by_freq.
    """
    # load reward and lick on event data
    with open(join(load_loc, 'res_reward_vs_lightHit.pkl'), 'rb') as f:
        res_reward_vs_lightHit = pickle.load(f)
    with open(join(load_loc, 'res_lick_on_vs_lightHit.pkl'), 'rb') as f:
        res_lick_on_vs_lightHit = pickle.load(f)
    with open(join(load_loc, 'auROC_results_hit.pkl'), 'rb') as f:
        auROC_results_hit = pickle.load(f)

    # load light event data
    event_data_by_freq = {}
    results_by_freq = {}
    results_vs_hit_by_freq = {}

    for freq in target_freqs:
        freq_str = str(freq)
        file_path = os.path.join(load_loc, f'res{freq_str}.pkl') # all light
        with open(file_path, 'rb') as f:
            event_data_by_freq[freq] = pickle.load(f)

        # Load result dictionaries
        res_file = os.path.join(load_loc, f'res{freq_str}.pkl')
        res_vs_hit_file = os.path.join(load_loc, f'res_CR_vs_lightHit_{freq_str}.pkl') # all light (named "CR" by mistake)
        with open(res_file, 'rb') as f:
            res = pickle.load(f)
        with open(res_vs_hit_file, 'rb') as f:
            res_vs_hit = pickle.load(f)

        results_by_freq[freq] = res
        results_vs_hit_by_freq[freq] = res_vs_hit

    return event_data_by_freq, results_by_freq, results_vs_hit_by_freq, res_reward_vs_lightHit, res_lick_on_vs_lightHit, auROC_results_hit


def build_freq_matrix(n_neurons, target_freqs, results_by_freq, results_vs_hit_by_freq):
    """
    Build a (n_neurons x n_freq) array freq_matrix.
    For each neuron and frequency:
      - If auROC > 90th percentile in both comparisons => store the positive auROC.
      - If auROC < 10th percentile in both comparisons => store that auROC (indicating inhibition).
      - Else => NaN.
    """
    n_freq = len(target_freqs)
    freq_matrix = np.full((n_neurons, n_freq), np.nan)

    # Track statistics for validation
    excitation_values = []
    inhibition_values = []

    for j, freq in enumerate(target_freqs):
        res = results_by_freq[freq]
        res_vs_hit = results_vs_hit_by_freq[freq]

        for i in range(n_neurons):
            if np.isnan(res['aurocA'][i]) or np.isnan(res_vs_hit['aurocA'][i]):
                continue

            # Excitation
            if (res['pvalA'][i] == 'above_95') and (res_vs_hit['pvalA'][i] == 'above_95'):
                freq_matrix[i, j] = res['aurocA'][i]
                excitation_values.append(res['aurocA'][i])
            # Inhibition
            elif (res['pvalA'][i] == 'below_5') and (res_vs_hit['pvalA'][i] == 'below_5'):
                freq_matrix[i, j] = res['aurocA'][i]
                inhibition_values.append(res['aurocA'][i])

    # Test and print validation results
    test_excitation_inhibition_values(excitation_values, inhibition_values)

    return freq_matrix


def test_excitation_inhibition_values(excitation_values, inhibition_values):
    """
    Test and print results to verify that excitation corresponds to auROC > 0.5 
    and inhibition corresponds to auROC < 0.5
    """
    print("\n" + "="*60)
    print("TESTING EXCITATION/INHIBITION auROC VALUES")
    print("="*60)
    
    if excitation_values:
        excitation_array = np.array(excitation_values)
        print(f"\nEXCITATION VALUES (n={len(excitation_values)}):")
        print(f"  Range: {excitation_array.min():.4f} to {excitation_array.max():.4f}")
        print(f"  Mean: {excitation_array.mean():.4f} ± {excitation_array.std():.4f}")
        
        # Test if all excitation values are > 0.5
        above_05 = np.sum(excitation_array > 0.5)
        below_05 = np.sum(excitation_array <= 0.5)
        print(f"  Values > 0.5: {above_05}/{len(excitation_values)} ({100*above_05/len(excitation_values):.1f}%)")
        print(f"  Values ≤ 0.5: {below_05}/{len(excitation_values)} ({100*below_05/len(excitation_values):.1f}%)")
        
        if below_05 > 0:
            print(f"  ⚠️  WARNING: {below_05} excitation values are ≤ 0.5!")
            problematic = excitation_array[excitation_array <= 0.5]
            print(f"     Problematic values: {problematic}")
        else:
            print(f"  ✅ All excitation values are > 0.5")
    else:
        print("\nNo excitation values found")
    
    if inhibition_values:
        inhibition_array = np.array(inhibition_values)
        print(f"\nINHIBITION VALUES (n={len(inhibition_values)}):")
        print(f"  Range: {inhibition_array.min():.4f} to {inhibition_array.max():.4f}")
        print(f"  Mean: {inhibition_array.mean():.4f} ± {inhibition_array.std():.4f}")
        
        # Test if all inhibition values are < 0.5
        below_05 = np.sum(inhibition_array < 0.5)
        above_05 = np.sum(inhibition_array >= 0.5)
        print(f"  Values < 0.5: {below_05}/{len(inhibition_values)} ({100*below_05/len(inhibition_values):.1f}%)")
        print(f"  Values ≥ 0.5: {above_05}/{len(inhibition_values)} ({100*above_05/len(inhibition_values):.1f}%)")
        
        if above_05 > 0:
            print(f"  ⚠️  WARNING: {above_05} inhibition values are ≥ 0.5!")
            problematic = inhibition_array[inhibition_array >= 0.5]
            print(f"     Problematic values: {problematic}")
        else:
            print(f"  ✅ All inhibition values are < 0.5")
    else:
        print("\nNo inhibition values found")
    
    print("\n" + "="*60)
    print("SUMMARY:")
    if excitation_values and inhibition_values:
        exc_correct = np.all(np.array(excitation_values) > 0.5)
        inh_correct = np.all(np.array(inhibition_values) < 0.5)
        
        if exc_correct and inh_correct:
            print("✅ PASSED: All excitation values > 0.5 and all inhibition values < 0.5")
        else:
            print("❌ FAILED: Some values don't match expected thresholds")
            if not exc_correct:
                print("   - Some excitation values ≤ 0.5")
            if not inh_correct:
                print("   - Some inhibition values ≥ 0.5")
    else:
        print("⚠️  Incomplete test: Missing excitation or inhibition values")
    print("="*60 + "\n")

def find_reward_significant_neurons(res_reward_vs_lightHit, res_lick_on_vs_lightHit, auROC_results_hit):
    """
    Find neurons that are significant for reward vs. lightHit.
    """
    reward_significant_increase = []
    reward_significant_decrease = []
    lick_on_significant_increase = []
    lick_on_significant_decrease = []

    # reward
    for i in range(len(auROC_results_hit['reward']['auroc'])):
        if auROC_results_hit['reward']['p_value'][i] == 'above_95' \
           and res_reward_vs_lightHit['classification'][i] == 'above_95':
            reward_significant_increase.append(i)
        elif auROC_results_hit['reward']['p_value'][i] == 'below_5' \
            and res_reward_vs_lightHit['classification'][i] == 'below_5':
            reward_significant_decrease.append(i)


    # lick on
    for i in range(len(auROC_results_hit['lick_onset']['auroc'])):
        if auROC_results_hit['lick_onset']['p_value'][i] == 'above_95' \
            and res_lick_on_vs_lightHit['classification'][i] == 'above_95':
            lick_on_significant_increase.append(i)
        elif auROC_results_hit['lick_onset']['p_value'][i] == 'below_5' \
            and res_lick_on_vs_lightHit['classification'][i] == 'below_5':
            lick_on_significant_decrease.append(i)

    # combine the reward and lick on significant neurons
    general_reward_significant_increase = set(reward_significant_increase).union(set(lick_on_significant_increase))
    general_reward_significant_decrease = set(reward_significant_decrease).union(set(lick_on_significant_decrease))
    general_reward_significant = general_reward_significant_increase.union(general_reward_significant_decrease)

    return general_reward_significant, general_reward_significant_increase, general_reward_significant_decrease


def find_light_significant_neurons(freq_matrix):
    """
    Find neurons that are significant for light responses, separated by increase/decrease.
    
    Parameters:
    -----------
    freq_matrix : np.ndarray
        Matrix of shape (n_neurons, n_frequencies) where:
        - Non-NaN values indicate significant responses
        - Values > 0.5 indicate increased responses (excitation)
        - Values < 0.5 indicate decreased responses (inhibition)
    
    Returns:
    --------
    light_significant_increase : list
        Indices of neurons showing significant increases to light
    light_significant_decrease : list  
        Indices of neurons showing significant decreases to light
    """
    n_neurons, n_freq = freq_matrix.shape
    
    light_significant_increase = []
    light_significant_decrease = []
    
    for i in range(n_neurons):
        # Get all non-NaN values for this neuron across frequencies
        valid_responses = freq_matrix[i, ~np.isnan(freq_matrix[i, :])]
        
        if len(valid_responses) > 0:
            # Check if any response shows increase (> 0.5)
            if np.any(valid_responses > 0.5):
                light_significant_increase.append(i)
            # Check if any response shows decrease (< 0.5)  
            elif np.any(valid_responses < 0.5):
                light_significant_decrease.append(i)
    
    return light_significant_increase, light_significant_decrease


def plot_neuron_figures(
        dff_run_list_sub, freq_indices, freq_matrix,
        target_freqs, pre_trial_length, rec_hz, save_loc
):
    """
    For each neuron, for each frequency with a non-NaN entry in freq_matrix:
      - Plot a heatmap of normalized dF/F across trials
      - Plot a mean time course with SEM
    Save each neuron's figure to <save_loc>/neuron_figures.
    """
    n_neurons = len(dff_run_list_sub)
    fig_dir = os.path.join(save_loc, 'neuron_figures')
    os.makedirs(fig_dir, exist_ok=True)

    for i in range(n_neurons):
        sig_freq_indices = np.where(~np.isnan(freq_matrix[i, :]))[0]
        if len(sig_freq_indices) == 0:
            continue

        fig = plt.figure(figsize=(4 * len(sig_freq_indices) + 2, 8))
        gs = gridspec.GridSpec(2, len(sig_freq_indices), height_ratios=[2, 1])

        for idx, col in enumerate(sig_freq_indices):
            freq = target_freqs[col]
            trial_idxs = freq_indices.get(freq, [])
            if len(trial_idxs) == 0:
                continue

            # Extract trials for the current neuron/freq
            try:
                selected_trials = [dff_run_list_sub[i][j] for j in trial_idxs]
                heatmap_data = pd.DataFrame(selected_trials)
            except Exception as e:
                print(f"Error stacking trials for neuron {i}, freq {freq}: {e}")
                continue

            # Heatmap
            ax_top = fig.add_subplot(gs[0, idx])
            n_timepoints = heatmap_data.shape[1]
            norm = TwoSlopeNorm(vcenter=0)
            im = ax_top.imshow(
                heatmap_data,
                aspect='auto',
                interpolation='nearest',
                extent=[0, n_timepoints / rec_hz, heatmap_data.shape[0], 0],
                cmap="coolwarm",
                norm=norm
            )
            auroc_val = freq_matrix[i, col]
            if auroc_val > 0.5:
                response_type = "excitation"
                display_val = auroc_val
            else:
                response_type = "inhibition"
                display_val = auroc_val

            add_axvline_opto_seconds(ax_top, pre_trial_length, rec_hz)
            ax_top.legend()
            ax_top.set_title(f'{freq} Hz\n{response_type}\nauROC={display_val:.2f}')
            ax_top.set_xlabel('Time (sec)')
            ax_top.set_ylabel('Trial')
            ax_top.set_xlim(0, 23)
            plt.colorbar(im, ax=ax_top, fraction=0.046, pad=0.04)

            # Mean time course
            mean_tc = np.nanmean(heatmap_data, axis=0)
            sem_tc = np.nanstd(heatmap_data, axis=0) / np.sqrt(heatmap_data.shape[0])
            ax_bottom = fig.add_subplot(gs[1, idx])
            time = np.arange(len(mean_tc))
            time_sec = time / rec_hz

            if freq_matrix[i, col] > 0.5:
                lbl = f'{freq} Hz (excitation, auROC={freq_matrix[i, col]:.2f})'
            else:
                lbl = f'{freq} Hz (inhibition, auROC={freq_matrix[i, col]:.2f})'

            ax_bottom.plot(time_sec, mean_tc, label=lbl)
            ax_bottom.fill_between(time_sec, mean_tc - sem_tc, mean_tc + sem_tc, alpha=0.3)
            ax_bottom.set_xlabel('Time (sec)')
            ax_bottom.set_ylabel('Normalized dF/F')
            ax_bottom.set_xlim(0, 23)
            add_axvline_opto_seconds(ax_bottom, pre_trial_length, rec_hz)

        plt.tight_layout()
        out_name = f'neuron_{i}_responses.png'
        fig.savefig(os.path.join(fig_dir, out_name))
        plt.close(fig)

    print(f"Saved figures for neurons with significant frequencies in {fig_dir}")

# Create a two_set_fractions function
def two_set_fractions(set1, set2, n_total):
    """
    Computes the fraction of neurons (percentage of n_total) unique to each set and shared between them.
    """
    if n_total == 0:
        return {"set1_only": 0.0, "set2_only": 0.0, "both": 0.0}

    set1_only_count = len(set1 - set2)
    set2_only_count = len(set2 - set1)
    both_count = len(set1 & set2)

    return {
        "set1_only": (set1_only_count / n_total) * 100.0,
        "set2_only": (set2_only_count / n_total) * 100.0,
        "both": (both_count / n_total) * 100.0
    }


def quantify_non15hz_responsive_to_15hz(freq_matrix, target_freqs):
    """
    Quantify how many cells responsive to frequencies other than 15 Hz are also responsive to 15 Hz.
    
    Parameters:
    -----------
    freq_matrix : np.ndarray
        Matrix of shape (n_neurons, n_frequencies) where non-NaN values indicate significant responses
    target_freqs : list
        List of target frequencies corresponding to columns in freq_matrix
    
    Returns:
    --------
    dict : Dictionary containing quantification results
        - 'non_15hz_responsive_cells': set of neuron indices responsive to non-15Hz frequencies
        - 'cells_15hz_responsive': set of neuron indices responsive to 15Hz
        - 'overlap_cells': set of neuron indices responsive to both non-15Hz and 15Hz
        - 'percentage_overlap': percentage of non-15Hz responsive cells that are also responsive to 15Hz
        - 'detailed_breakdown': breakdown by individual frequencies
    """
    n_neurons, n_freq = freq_matrix.shape
    
    # Find the index of 15 Hz frequency
    try:
        freq_15hz_idx = target_freqs.index(15)
    except ValueError:
        raise ValueError("15 Hz frequency not found in target_freqs")
    
    # Find cells responsive to 15 Hz
    cells_15hz_responsive = set()
    for i in range(n_neurons):
        if not np.isnan(freq_matrix[i, freq_15hz_idx]):
            cells_15hz_responsive.add(i)
    
    # Find cells responsive to any non-15Hz frequency
    non_15hz_responsive_cells = set()
    for i in range(n_neurons):
        for j in range(n_freq):
            if j != freq_15hz_idx and not np.isnan(freq_matrix[i, j]):
                non_15hz_responsive_cells.add(i)
                break  # Found at least one non-15Hz response for this neuron
    
    # Find overlap: cells responsive to both non-15Hz frequencies AND 15Hz
    overlap_cells = non_15hz_responsive_cells & cells_15hz_responsive
    
    # Calculate percentage
    if len(non_15hz_responsive_cells) > 0:
        percentage_overlap = (len(overlap_cells) / len(non_15hz_responsive_cells)) * 100.0
    else:
        percentage_overlap = 0.0
    
    # Detailed breakdown by individual frequencies
    detailed_breakdown = {}
    for j, freq in enumerate(target_freqs):
        if freq == 15:  # Skip 15 Hz for this analysis
            continue
            
        # Find cells responsive to this specific frequency
        cells_this_freq = set()
        for i in range(n_neurons):
            if not np.isnan(freq_matrix[i, j]):
                cells_this_freq.add(i)
        
        # Find overlap with 15 Hz
        overlap_this_freq = cells_this_freq & cells_15hz_responsive
        
        # Calculate percentage for this frequency
        if len(cells_this_freq) > 0:
            pct_this_freq = (len(overlap_this_freq) / len(cells_this_freq)) * 100.0
        else:
            pct_this_freq = 0.0
        
        detailed_breakdown[freq] = {
            'responsive_to_freq': len(cells_this_freq),
            'also_responsive_to_15hz': len(overlap_this_freq),
            'percentage': pct_this_freq
        }
    
    return {
        'non_15hz_responsive_cells': non_15hz_responsive_cells,
        'cells_15hz_responsive': cells_15hz_responsive,
        'overlap_cells': overlap_cells,
        'percentage_overlap': percentage_overlap,
        'detailed_breakdown': detailed_breakdown,
        'summary': {
            'total_non_15hz_responsive': len(non_15hz_responsive_cells),
            'total_15hz_responsive': len(cells_15hz_responsive),
            'total_overlap': len(overlap_cells),
            'percentage_non_15hz_also_15hz': percentage_overlap
        }
    }
###############################################################################
# Main routine
###############################################################################
if __name__ == '__main__':
    ##########################################################################
    # Settings
    ##########################################################################
    day = ['250227']
    mouse = 'OI208'
    ROOT_PATH = fr'Z:\2p_data\{mouse}\{day[0]}\ML\data'
    load_loc = fr'Z:\2p_data\{mouse}\{day[0]}\ML\data\processed_data\auROC_freq_psycho_new'
    save_loc = fr'Z:\2p_data\{mouse}\{day[0]}\ML\data\processed_data\auROC_freq_psycho_gaussian_sigmpoid'
    os.makedirs(save_loc, exist_ok=True)
    rec_hz = 15
    pre_trial_length = 4 * rec_hz
    n_priming_trials = 23   

    frequency_map = {6: 0, 7: 15, 8: 10, 9: 7.5, 10: 5, 11: 3, 12: 1, 13: 0.5}
    # target_freqs = [15]  # Only analyze 15 Hz frequency for light
    target_freqs = [0.5, 1, 3, 5, 7.5, 10, 15]

    ##########################################################################
    # Build freq_indices
    ##########################################################################
    conditions_played = np.load(join(ROOT_PATH, 'conditions_played.npy'))
    # remove priming trials
    conditions_played = conditions_played[n_priming_trials:]
    freq_indices = build_freq_indices(conditions_played, frequency_map, target_freqs)

    ##########################################################################
    # Load event data & compute freq_matrix
    ##########################################################################
    event_data_by_freq, results_by_freq, results_vs_hit_by_freq, \
        res_reward_vs_lightHit, res_lick_on_vs_lightHit, auROC_results_hit = load_event_data(load_loc, target_freqs)
    n_neurons = len(event_data_by_freq[15]['aurocA'])
    freq_matrix = build_freq_matrix(n_neurons, target_freqs, results_by_freq, results_vs_hit_by_freq)
        # 1) Filter to only neurons significant for >= 1 frequency
    mask = ~np.all(np.isnan(freq_matrix), axis=1)
    sig_light_indices = np.where(mask)[0]
    all_reward_significant, reward_significant_increase, reward_significant_decrease = find_reward_significant_neurons(res_reward_vs_lightHit, res_lick_on_vs_lightHit,auROC_results_hit)
    
    # Separate light significant neurons by direction
    light_significant_increase, light_significant_decrease = find_light_significant_neurons(freq_matrix)

    # Save freq_matrix to pickle
    df = pd.DataFrame(freq_matrix, columns=[str(freq) for freq in target_freqs])
    with open(os.path.join(save_loc, 'freq_matrix.pkl'), 'wb') as f:
        pickle.dump(df, f)

    ##########################################################################
    # Compute and save fraction data for plotting
    ##########################################################################
    # Convert to sets for fraction computation
    light_significant_increase_set = set(light_significant_increase)
    light_significant_decrease_set = set(light_significant_decrease)
    reward_significant_increase_set = set(reward_significant_increase)
    reward_significant_decrease_set = set(reward_significant_decrease)
    # load the indexes of the significant neurons for the gaussian and sigmoid fits
    gaussian_sigmoid_path = r'Z:\2p_data\{}\{}\ML\data\processed_data\auroc_neurometric_zscore_no_sub_v4'.format(
        mouse, day[0]
    )
    gaussian_neuron_indices = np.load(join(gaussian_sigmoid_path, 'gaussian_neuron_indices.npy'))
    sigmoid_neuron_indices = np.load(join(gaussian_sigmoid_path, 'sigmoid_neuron_indices.npy'))
    # create sets for the gaussian and sigmoid significant neurons
    gaussian_neuron_set = set(gaussian_neuron_indices)
    sigmoid_neuron_set = set(sigmoid_neuron_indices)
    # compute the overlap between the gaussian/sigmoid sets and the light/reward significant sets
    gaussian_light_increase_overlap = len(gaussian_neuron_set & light_significant_increase_set)
    gaussian_light_decrease_overlap = len(gaussian_neuron_set & light_significant_decrease_set)
    gaussian_reward_increase_overlap = len(gaussian_neuron_set & reward_significant_increase_set)
    gaussian_reward_decrease_overlap = len(gaussian_neuron_set & reward_significant_decrease_set)
    sigmoid_light_increase_overlap = len(sigmoid_neuron_set & light_significant_increase_set)
    sigmoid_light_decrease_overlap = len(sigmoid_neuron_set & light_significant_decrease_set)
    sigmoid_reward_increase_overlap = len(sigmoid_neuron_set & reward_significant_increase_set)
    sigmoid_reward_decrease_overlap = len(sigmoid_neuron_set & reward_significant_decrease_set)

    # --- Collapsed sets (direction-agnostic) ---
    all_light_set = light_significant_increase_set | light_significant_decrease_set
    all_reward_set = reward_significant_increase_set | reward_significant_decrease_set
    all_both_set = all_light_set & all_reward_set  # significant for both modalities (any direction)

    # Compute fractions for 2-set comparisons
    fractions_2set = {}

    # ---------------------------------------------------------------------
    # Original reward vs light comparisons
    # ---------------------------------------------------------------------
    fractions_2set["Hit(reward)_LightOn_All"] = two_set_fractions(
        all_reward_set, all_light_set, n_neurons
    )
    fractions_2set["Hit(reward)_LightOn_Increase"] = two_set_fractions(
        reward_significant_increase_set, light_significant_increase_set, n_neurons
    )
    fractions_2set["Hit(reward)_LightOn_Decrease"] = two_set_fractions(
        reward_significant_decrease_set, light_significant_decrease_set, n_neurons
    )

    # ---------------------------------------------------------------------
    # Gaussian comparisons
    # ---------------------------------------------------------------------
    fractions_2set["Gaussian_LightOn_All"] = two_set_fractions(
        gaussian_neuron_set, all_light_set, n_neurons
    )
    fractions_2set["Gaussian_LightOn_Increase"] = two_set_fractions(
        gaussian_neuron_set, light_significant_increase_set, n_neurons
    )
    fractions_2set["Gaussian_LightOn_Decrease"] = two_set_fractions(
        gaussian_neuron_set, light_significant_decrease_set, n_neurons
    )

    fractions_2set["Gaussian_Hit(reward)_All"] = two_set_fractions(
        gaussian_neuron_set, all_reward_set, n_neurons
    )
    fractions_2set["Gaussian_Hit(reward)_Increase"] = two_set_fractions(
        gaussian_neuron_set, reward_significant_increase_set, n_neurons
    )
    fractions_2set["Gaussian_Hit(reward)_Decrease"] = two_set_fractions(
        gaussian_neuron_set, reward_significant_decrease_set, n_neurons
    )

    fractions_2set["Gaussian_BothModalities_All"] = two_set_fractions(
        gaussian_neuron_set, all_both_set, n_neurons
    )

    # ---------------------------------------------------------------------
    # Sigmoid comparisons
    # ---------------------------------------------------------------------
    fractions_2set["Sigmoid_LightOn_All"] = two_set_fractions(
        sigmoid_neuron_set, all_light_set, n_neurons
    )
    fractions_2set["Sigmoid_LightOn_Increase"] = two_set_fractions(
        sigmoid_neuron_set, light_significant_increase_set, n_neurons
    )
    fractions_2set["Sigmoid_LightOn_Decrease"] = two_set_fractions(
        sigmoid_neuron_set, light_significant_decrease_set, n_neurons
    )

    fractions_2set["Sigmoid_Hit(reward)_All"] = two_set_fractions(
        sigmoid_neuron_set, all_reward_set, n_neurons
    )
    fractions_2set["Sigmoid_Hit(reward)_Increase"] = two_set_fractions(
        sigmoid_neuron_set, reward_significant_increase_set, n_neurons
    )
    fractions_2set["Sigmoid_Hit(reward)_Decrease"] = two_set_fractions(
        sigmoid_neuron_set, reward_significant_decrease_set, n_neurons
    )

    fractions_2set["Sigmoid_BothModalities_All"] = two_set_fractions(
        sigmoid_neuron_set, all_both_set, n_neurons
    )

    # Package fractions data
    fractions_dict = {"2set": fractions_2set}

    # Save fraction data
    day_str = day[0] if isinstance(day, list) else day
    frac_filename = f"comparison_fractions_{day_str}.pkl"
    with open(os.path.join(save_loc, frac_filename), 'wb') as f:
        pickle.dump(fractions_dict, f)

    print(f"Saved fraction data to {frac_filename}")
    ##########################################################################
    # Quantify non-15Hz responsive cells that are also responsive to 15Hz
    ##########################################################################
    freq_15hz_analysis = quantify_non15hz_responsive_to_15hz(freq_matrix, target_freqs)
    
    # Save the 15Hz analysis results
    with open(os.path.join(save_loc, f'non_15hz_to_15hz_analysis_{day_str}.pkl'), 'wb') as f:
        pickle.dump(freq_15hz_analysis, f)
    
    print(f"Saved 15Hz frequency analysis to non_15hz_to_15hz_analysis_{day_str}.pkl")
    
    # Print summary of significant neurons
    # Save summary and overlap information to a text file instead of printing
    summary_lines = []
    summary_lines.append("Significant neurons summary:\n")
    summary_lines.append(f"  Light increase: {len(light_significant_increase)} neurons\n")
    summary_lines.append(f"  Light decrease: {len(light_significant_decrease)} neurons\n")
    summary_lines.append(f"  Reward increase: {len(reward_significant_increase)} neurons\n")
    summary_lines.append(f"  Reward decrease: {len(reward_significant_decrease)} neurons\n")
    summary_lines.append(f"  Total neurons: {n_neurons}\n")
    
    # Overlap information
    reward_light_inc_overlap = len(reward_significant_increase_set & light_significant_increase_set)
    reward_light_dec_overlap = len(reward_significant_decrease_set & light_significant_decrease_set)
    summary_lines.append("\nOverlap summary:\n")
    summary_lines.append(f"  Reward increase ∩ Light increase: {reward_light_inc_overlap} neurons\n")
    summary_lines.append(f"  Reward decrease ∩ Light decrease: {reward_light_dec_overlap} neurons\n")
    
    # Add 15Hz frequency analysis results
    summary_lines.append("\n" + "="*60 + "\n")
    summary_lines.append("15 Hz FREQUENCY ANALYSIS\n")
    summary_lines.append("="*60 + "\n")
    summary_lines.append(f"Cells responsive to non-15Hz frequencies: {freq_15hz_analysis['summary']['total_non_15hz_responsive']} neurons\n")
    summary_lines.append(f"Cells responsive to 15Hz: {freq_15hz_analysis['summary']['total_15hz_responsive']} neurons\n")
    summary_lines.append(f"Cells responsive to both non-15Hz AND 15Hz: {freq_15hz_analysis['summary']['total_overlap']} neurons\n")
    summary_lines.append(f"Percentage of non-15Hz responsive cells that are also responsive to 15Hz: {freq_15hz_analysis['summary']['percentage_non_15hz_also_15hz']:.1f}%\n")
    
    summary_lines.append("\nDetailed breakdown by frequency:\n")
    for freq, data in freq_15hz_analysis['detailed_breakdown'].items():
        summary_lines.append(f"  {freq} Hz: {data['also_responsive_to_15hz']}/{data['responsive_to_freq']} ({data['percentage']:.1f}%) also responsive to 15Hz\n")

    # Write to file with UTF-8 encoding to handle Unicode characters
    summary_filename = os.path.join(save_loc, f"significant_neurons_summary_{day_str}.txt")
    with open(summary_filename, "w", encoding='utf-8') as f:
        f.writelines(summary_lines)

    # ##########################################################################
    # # Plot single-neuron figures
    # ##########################################################################
    # plot_neuron_figures(
    #     dff_run_list_sub, freq_indices, freq_matrix,
    #     target_freqs, pre_trial_length, rec_hz, data_loc
    # )

    #########################################################################
    # Generate summary heatmaps
    #########################################################################
    # generate_summary_heatmaps(freq_matrix, target_freqs, save_loc)
    #########################################################################
    # seperate the R² values for the neurons that had gassian fits and those that had sigmoid fits
    # significant_gaussian = np.load(join(ROOT_PATH,  r'processed_data\auroc_nurometric_substrated_r_shuffled\significant_neurons_gauss.npy'))
    # significant_sigmoid = np.load(join(ROOT_PATH,  r'processed_data\auroc_nurometric_substrated_r_shuffled\significant_neurons.npy'))



