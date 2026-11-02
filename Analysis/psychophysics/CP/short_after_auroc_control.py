"""
Choice probability (CP) significance against the CR-vs-CR control.

For one session:
  1. Load the CP (CR vs. FA auROC) distributions of the 5 CR-vs-CR control repetitions
     (auROC_freq_history_control_CR-CR_rep1..rep5) and compare them across repetitions.
  2. Define control thresholds from the percentiles of the control median CPs
     (excitatory: 99.5th / 99th / 97.5th / 95th; inhibitory: 0.5th / 1st / 2.5th / 5th).
  3. Compare the real CP of each neuron (auROC_freq_history) to these thresholds and save
     the counts, percentages and indices of the significant neurons (comparison_results.pkl),
     together with plots of representative neurons.
"""

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
import scipy.stats


matplotlib.use('Qt5Agg')
matplotlib.rcParams['pdf.fonttype'] = 42



# ---------------------------------------------------------------------------
# Data locations (set these before running)
# ---------------------------------------------------------------------------
DATA_ROOT = r'path\to\2p_data'  # folder with the imaging sessions (<mouse>/<session>/ML/...)

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
    ax.axvline(x=0, color='darkorange', linestyle='--', linewidth=1, label='Distractor light')
    ax.axvline(x=1, color='darkorange', linestyle='--', linewidth=1)
    ax.axvline(x=1, color='yellow', linestyle='--', linewidth=1,
               label='Opto stimulation')
    ax.axvline(x=5, color='yellow', linestyle='--', linewidth=1)
    ax.axvline(x=5 , color='blue', linestyle='--', linewidth=1, label='Odor')
    ax.axvline(x=7, color='blue', linestyle='--', linewidth=1)


def add_axvline_opto_frames(ax, pre_trial_length, rec_hz):
    """
    Draw vertical lines (in frames) on a plot, marking:
      - Distractor light (darkorange)
      - Opto stimulation (yellow)
      - Odor (blue)
    """
    ax.axvline(x=pre_trial_length, color='darkorange', linestyle='--', linewidth=1, label='Distractor light')
    ax.axvline(x=pre_trial_length + 9 * rec_hz, color='darkorange', linestyle='--', linewidth=1)
    ax.axvline(x=pre_trial_length + 1 * rec_hz, color='yellow', linestyle='--', linewidth=1, label='Opto stimulation')
    ax.axvline(x=pre_trial_length + 4 * rec_hz, color='yellow', linestyle='--', linewidth=1)
    ax.axvline(x=pre_trial_length + 5 * rec_hz, color='blue', linestyle='--', linewidth=1, label='Odor')
    ax.axvline(x=pre_trial_length + 8 * rec_hz, color='blue', linestyle='--', linewidth=1)


def build_freq_indices(conditions_played, frequency_map, target_freqs):
    """
    Build a dictionary mapping each target frequency to a list of trial indices.
    """
    freq_indices = {}
    for freq in target_freqs:
        freq_indices[freq] = []
    
    for i, condition in enumerate(conditions_played):
        if condition in frequency_map:
            freq = frequency_map[condition]
            if freq in target_freqs:
                freq_indices[freq].append(i)
    
    return freq_indices


def load_dff_run_list_sub(dff_file):
    """Load dF/F data from file."""
    with open(dff_file, 'rb') as f:
        return pickle.load(f)


def load_event_data(load_loc, target_freqs):
    """Load event data and build frequency indices."""
    conditions_played = np.load(join(load_loc, 'conditions_played.npy'))
    frequency_map = np.load(join(load_loc, 'frequency_map.npy'), allow_pickle=True).item()
    freq_indices = build_freq_indices(conditions_played, frequency_map, target_freqs)
    return freq_indices


def build_freq_matrix(n_neurons, target_freqs, results_by_freq, results_hit_by_freq):
    """
    Build a matrix where each row is a neuron and each column is a frequency.
    Values are auROC scores for neurons that are significant for that frequency.
    """
    freq_matrix = np.full((n_neurons, len(target_freqs)), np.nan)
    
    for freq_idx, freq in enumerate(target_freqs):
        if freq in results_by_freq:
            neuron_indices = results_by_freq[freq]
            auroc_values = results_hit_by_freq[freq]
            
            for neuron_idx, auroc_val in zip(neuron_indices, auroc_values):
                freq_matrix[neuron_idx, freq_idx] = auroc_val
    
    return freq_matrix


def plot_neuron_figures(
        dff_run_list_sub, freq_indices, freq_matrix,
        target_freqs, pre_trial_length, rec_hz, save_loc
):
    """
    For each neuron that is significant for at least one frequency:
      1) Plot the mean response (across trials) for each significant frequency.
      2) Add vertical lines marking the timing of events.
      3) Save the plot.
    """
    n_neurons = len(dff_run_list_sub)
    
    for i in range(n_neurons):
        # Check if neuron i is significant for any frequency
        sig_freq_indices = np.where(~np.isnan(freq_matrix[i, :]))[0]
        if len(sig_freq_indices) == 0:
            continue
        
        # Create subplot grid
        n_sig_freqs = len(sig_freq_indices)
        n_cols = min(3, n_sig_freqs)
        n_rows = (n_sig_freqs + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        if n_sig_freqs == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        
        for subplot_idx, freq_idx in enumerate(sig_freq_indices):
            freq = target_freqs[freq_idx]
            trial_idxs = freq_indices.get(freq, [])
            
            if len(trial_idxs) == 0:
                continue
            
            # Extract trials for the current neuron and frequency
            selected_trials = [dff_run_list_sub[i][j] for j in trial_idxs if j < len(dff_run_list_sub[i])]
            
            if len(selected_trials) == 0:
                continue
            
            # Compute mean response across trials
            trial_series = [pd.Series(trial.tolist()) for trial in selected_trials]
            df_trials = pd.DataFrame(trial_series)
            mean_resp = df_trials.mean(axis=0, skipna=True).values
            
            # Plot mean response
            time_axis = np.arange(len(mean_resp)) / rec_hz
            axes[subplot_idx].plot(time_axis, mean_resp, 'b-', linewidth=2)
            
            # Add vertical lines for events
            add_axvline_opto_seconds(axes[subplot_idx], pre_trial_length, rec_hz)
            
            axes[subplot_idx].set_title(f'Neuron {i}, {freq} Hz\nauROC = {freq_matrix[i, freq_idx]:.3f}')
            axes[subplot_idx].set_xlabel('Time (sec)')
            axes[subplot_idx].set_ylabel('dF/F')
            axes[subplot_idx].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for subplot_idx in range(n_sig_freqs, len(axes)):
            axes[subplot_idx].set_visible(False)
        
        plt.tight_layout()
        out_path = os.path.join(save_loc, f'neuron_{i}_responses.png')
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved neuron {i} plots to {out_path}")


def generate_summary_heatmaps(freq_matrix, target_freqs, save_loc):
    """
    Generate summary heatmaps showing auROC values for all neurons across frequencies.
    """
    n_neurons = freq_matrix.shape[0]
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    
    # Create mask for NaN values
    mask = np.isnan(freq_matrix)
    
    # Create heatmap
    sns.heatmap(freq_matrix, 
                xticklabels=target_freqs,
                yticklabels=False,
                cmap='coolwarm',
                center=0.5,
                mask=mask,
                cbar_kws={'label': 'auROC'})
    
    plt.title('auROC Values Across Frequencies')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Neuron')
    plt.tight_layout()
    
    out_path = os.path.join(save_loc, 'auroc_heatmap.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved auROC heatmap to {out_path}")
    
    # Create sorted heatmap
    plt.figure(figsize=(10, 8))
    
    # Sort neurons by their maximum auROC value
    max_aurocs = np.nanmax(freq_matrix, axis=1)
    sorted_indices = np.argsort(max_aurocs)[::-1]  # Sort in descending order
    sorted_matrix = freq_matrix[sorted_indices]
    
    # Create mask for NaN values
    mask = np.isnan(sorted_matrix)
    
    # Create heatmap
    sns.heatmap(sorted_matrix, 
                xticklabels=target_freqs,
                yticklabels=False,
                cmap='coolwarm',
                center=0.5,
                mask=mask,
                cbar_kws={'label': 'auROC'})
    
    plt.title('auROC Values Across Frequencies (Sorted by Max auROC)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Neuron (sorted)')
    plt.tight_layout()
    
    out_path = os.path.join(save_loc, 'auroc_heatmap_sorted.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved sorted auROC heatmap to {out_path}")


def plot_cr_vs_fa_all(
        dff_run_list_sub,
        pre_trial_length,
        rec_hz,
        load_loc
):
    # load trial indices
    conditions_played = np.load(join(load_loc, 'conditions_played.npy'))
    frequency_map = np.load(join(load_loc, 'frequency_map.npy'), allow_pickle=True).item()
    
    # Create CR and FA trial lists
    cr_trials = []
    fa_trials = []
    
    for i, condition in enumerate(conditions_played):
        if condition in frequency_map:
            freq = frequency_map[condition]
            # Add to appropriate list based on frequency
            if freq > 0:  # Assuming positive frequencies are CR trials
                cr_trials.append(i)
            else:  # Assuming negative or zero frequencies are FA trials
                fa_trials.append(i)
    
    def pad_list(lst):
        """Pad a list to the maximum length with NaN values."""
        max_len = max(len(item) for item in lst)
        padded = []
        for item in lst:
            if len(item) < max_len:
                padded_item = np.pad(item, (0, max_len - len(item)), mode='constant', constant_values=np.nan)
            else:
                padded_item = item
            padded.append(padded_item)
        return np.array(padded)
    
    # Extract CR and FA responses for each neuron
    n_neurons = len(dff_run_list_sub)
    cr_responses = []
    fa_responses = []
    
    for neuron_idx in range(n_neurons):
        # CR responses
        neuron_cr_trials = [dff_run_list_sub[neuron_idx][trial_idx] for trial_idx in cr_trials 
                           if trial_idx < len(dff_run_list_sub[neuron_idx])]
        if neuron_cr_trials:
            cr_responses.append(pad_list(neuron_cr_trials))
        
        # FA responses
        neuron_fa_trials = [dff_run_list_sub[neuron_idx][trial_idx] for trial_idx in fa_trials 
                           if trial_idx < len(dff_run_list_sub[neuron_idx])]
        if neuron_fa_trials:
            fa_responses.append(pad_list(neuron_fa_trials))
    
    # Plot mean responses
    if cr_responses:
        cr_mean = np.nanmean(cr_responses, axis=0)
        cr_sem = np.nanstd(cr_responses, axis=0) / np.sqrt(len(cr_responses))
        time_axis = np.arange(len(cr_mean)) / rec_hz
        
        plt.figure(figsize=(10, 6))
        plt.plot(time_axis, cr_mean, 'b-', linewidth=2, label='CR')
        plt.fill_between(time_axis, cr_mean - cr_sem, cr_mean + cr_sem, alpha=0.3, color='blue')
        
        add_axvline_opto_seconds(plt.gca(), pre_trial_length, rec_hz)
        
        plt.title('Mean CR Response Across All Neurons')
        plt.xlabel('Time (sec)')
        plt.ylabel('dF/F')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        out_path = os.path.join(load_loc, 'cr_response.png')
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved CR response plot to {out_path}")
    
    if fa_responses:
        fa_mean = np.nanmean(fa_responses, axis=0)
        fa_sem = np.nanstd(fa_responses, axis=0) / np.sqrt(len(fa_responses))
        time_axis = np.arange(len(fa_mean)) / rec_hz
        
        plt.figure(figsize=(10, 6))
        plt.plot(time_axis, fa_mean, 'r-', linewidth=2, label='FA')
        plt.fill_between(time_axis, fa_mean - fa_sem, fa_mean + fa_sem, alpha=0.3, color='red')
        
        add_axvline_opto_seconds(plt.gca(), pre_trial_length, rec_hz)
        
        plt.title('Mean FA Response Across All Neurons')
        plt.xlabel('Time (sec)')
        plt.ylabel('dF/F')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        out_path = os.path.join(load_loc, 'fa_response.png')
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved FA response plot to {out_path}")


def plot_mean_response_lines(
        dff_run_list_sub,  # list of neurons × trials (each is 1D array)
        freq_indices,  # dict: freq → list of trial-indices
        target_freqs,  # list of freqs to process
        pre_trial_length,  # in frames
        rec_hz,  # sampling rate
        load_loc  # folder where heatmaps + pkls live
):
    # load CR vs FA trial lists
    conditions_played = np.load(join(load_loc, 'conditions_played.npy'))
    frequency_map = np.load(join(load_loc, 'frequency_map.npy'), allow_pickle=True).item()
    
    # Create CR and FA trial lists
    cr_trials = []
    fa_trials = []
    
    for i, condition in enumerate(conditions_played):
        if condition in frequency_map:
            freq = frequency_map[condition]
            # Add to appropriate list based on frequency
            if freq > 0:  # Assuming positive frequencies are CR trials
                cr_trials.append(i)
            else:  # Assuming negative or zero frequencies are FA trials
                fa_trials.append(i)
    
    # Load choice probability data
    with open(os.path.join(load_loc, 'cp.pkl'), 'rb') as f:
        choice_prob = pickle.load(f)
    
    # Get significant neurons
    sig_neuron_indices = np.load(join(load_loc, 'significant_neurons_for_CP.pkl'))
    
    # Plot mean responses for significant neurons
    n_neurons = len(dff_run_list_sub)
    
    for i in sig_neuron_indices:
        if i >= n_neurons:
            continue
        
        # Extract CR and FA responses for this neuron
        neuron_cr_trials = [dff_run_list_sub[i][trial_idx] for trial_idx in cr_trials 
                           if trial_idx < len(dff_run_list_sub[i])]
        neuron_fa_trials = [dff_run_list_sub[i][trial_idx] for trial_idx in fa_trials 
                           if trial_idx < len(dff_run_list_sub[i])]
        
        if not neuron_cr_trials or not neuron_fa_trials:
            continue
        
        # Compute mean responses
        cr_mean = np.mean(neuron_cr_trials, axis=0)
        fa_mean = np.mean(neuron_fa_trials, axis=0)
        
        # Get choice probability for this neuron
        cp_val = np.nanmean(choice_prob['all_distributions'][i])
        
        # Plot
        time_axis = np.arange(len(cr_mean)) / rec_hz
        
        plt.figure(figsize=(10, 6))
        plt.plot(time_axis, cr_mean, 'b-', linewidth=2, label=f'CR (CP={cp_val:.3f})')
        plt.plot(time_axis, fa_mean, 'r-', linewidth=2, label='FA')
        
        add_axvline_opto_seconds(plt.gca(), pre_trial_length, rec_hz)
        
        plt.title(f'Neuron {i} Mean Responses')
        plt.xlabel('Time (sec)')
        plt.ylabel('dF/F')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        out_path = os.path.join(load_loc, f'neuron_{i}_cr_vs_fa.png')
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved neuron {i} CR vs FA plot to {out_path}")


def plot_mean_response_heatmaps(
        dff_run_list_sub, freq_indices, freq_matrix, target_freqs,
        pre_trial_length, rec_hz, save_loc
):
    """
    For each target_freq:
      - Find significant neurons (non-NaN in freq_matrix).
      - Sort them by auROC value (descending).
      - Compute & plot the mean response (across trials) as a heatmap (frames-based axes).
    """
    n_neurons = freq_matrix.shape[0]
    
    for freq in target_freqs:
        trial_inds = freq_indices.get(freq, [])
        if len(trial_inds) == 0:
            continue
        
        freq_col = target_freqs.index(freq)
        sig_neuron_indices = [
            i for i in range(n_neurons)
            if not np.isnan(freq_matrix[i, freq_col])
        ]
        if len(sig_neuron_indices) == 0:
            continue
        
        # Sort by descending auROC
        sig_neuron_indices = sorted(
            sig_neuron_indices,
            key=lambda i: freq_matrix[i, freq_col],
            reverse=True
        )
        
        mean_responses = []
        for i in sig_neuron_indices:
            trials = [dff_run_list_sub[i][j] for j in trial_inds if j < len(dff_run_list_sub[i])]
            if len(trials) == 0:
                continue
            trial_series = [pd.Series(trial.tolist()) for trial in trials]
            df_trials = pd.DataFrame(trial_series)
            mean_resp = df_trials.mean(axis=0, skipna=True).values
            # Clip to first 22 seconds if desired
            mean_responses.append(mean_resp[:22 * rec_hz])
        
        if len(mean_responses) == 0:
            continue
        
        df_mean = pd.DataFrame(mean_responses)
        vmin = df_mean.min().min() + 0.1
        vmax = df_mean.max().max() - 0.1
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        
        plt.figure(figsize=(10, 8))
        ax = sns.heatmap(df_mean, cmap='coolwarm', cbar=True, norm=norm)
        
        # X-axis ticks in frames; place them every 10 seconds => 10*rec_hz
        xtick_positions = np.arange(0, 22 * rec_hz, 10 * rec_hz)
        xtick_labels = [f"{pos / rec_hz:.1f}" for pos in xtick_positions]
        ax.set_xticks(xtick_positions)
        ax.set_xticklabels(xtick_labels)
        
        add_axvline_opto_frames(ax, pre_trial_length, rec_hz)
        ax.set_title(f"Mean Response Heatmap for {freq} Hz\n(Neurons = {df_mean.shape[0]})")
        ax.set_xlabel("Time (sec)")
        ax.set_xlim(0, 22 * rec_hz)
        ax.set_ylabel("Neuron (sorted by auROC)")
        
        plt.tight_layout()
        out_path = os.path.join(save_loc, f"mean_response_heatmap_{freq}Hz.png")
        plt.savefig(out_path)
        plt.close()
        print(f"Saved heatmap for {freq} Hz with {df_mean.shape[0]} neurons to {out_path}")


###############################################################################
# Updated function to handle choice probability and additional scatter (CP vs R²)
###############################################################################
def analyze_choice_probability_vs_r2(
        choice_prob, freq_matrix, target_freqs, save_loc, r_squared_values=None, sig_neuron=None, file_name=None
):
    """
    1. Restricts Analysis to neurons that are significant for at least one frequency.
    2. Plots a histogram of choice probability for "excitation" vs. "inhibition" groups.
    3. Generates three scatter plots:
       a) CP vs. frequency at which the neuron had its max (peak) auROC
       b) CP vs. frequency at which the neuron had its min (still-significant) auROC
       c) CP vs. R² (if r_squared_values is provided)
    """
    
    # We will look up each neuron's median CP (from choice_prob['all_distributions'][i])
    n_neurons = freq_matrix.shape[0]
    choice_medians = np.array([
        np.nanmean(choice_prob['all_distributions'][i]) for i in range(n_neurons)
    ])
    
    # 2) Plot CP distribution for excitatory vs. inhibitory neurons
    choice_prob_exc = np.full((n_neurons, 1), np.nan)
    choice_prob_inh = np.full((n_neurons, 1), np.nan)
    
    for i in range(n_neurons):
        median_val = choice_medians[i]
        if np.isnan(median_val):
            continue
        if median_val >= 0.5:
            choice_prob_exc[i] = median_val
        else:
            choice_prob_inh[i] = median_val
    
    bin_width = 0.01
    min_val = min(np.nanmin(choice_prob_exc), np.nanmin(choice_prob_inh))
    max_val = max(np.nanmax(choice_prob_exc), np.nanmax(choice_prob_inh))
    bins = np.arange(min_val, max_val + bin_width, bin_width)
    
    # Scatter: CP vs. R² (only if provided)
    if r_squared_values is not None:
        # Plot only for the same significant neurons
        r2_sig = r_squared_values[sig_neuron]
        cp_sig = choice_medians[sig_neuron]
        
        plt.figure()
        plt.scatter(cp_sig, r2_sig, alpha=0.6)
        # add identity line
        plt.plot([0, 1], [0, 1], color='black', linestyle='--')
        plt.xlabel('Choice Probability')
        plt.ylabel('r²')
        plt.title('Choice Probability vs. r² (Significant Neurons)')
        plt.tight_layout()
        out_scatter_r2 = os.path.join(save_loc, f'choice_prob_vs_r2_{file_name}.png')
        plt.savefig(out_scatter_r2)
        plt.show()
        print(f"Saved scatter of CP vs. r² to {out_scatter_r2}")


def choice_probability_hist(choice_prob,
                            save_loc,
                            type,
                            sig_neuron=None):
    """
    Histogram and rain-cloud helper, symmetrized CP′ in −1…1.
    """
    if sig_neuron is None:
        sig_neuron = []
    
    # Determine which distribution to use based on type
    if type == 'shuffled':
        distribution_key = 'all_shuffled_distributions'
    else:
        distribution_key = 'all_distributions'
    
    # --- compute median CP per neuron, then symmetrize ---
    n_neurons = len(choice_prob[distribution_key])
    medians = np.array([
        np.nanmean(choice_prob[distribution_key][i])
        for i in range(n_neurons)
    ])
    medians = medians[sig_neuron]  # keep only significant
    
    # Remove NaN values before processing
    valid_mask = ~np.isnan(medians)
    medians_clean = medians[valid_mask]
    
    cp_sym = 2 * (medians_clean - 0.5)  # CP′ range −1…1
    
    # split into excitation (CP′ ≥ 0) vs inhibition (CP′ < 0)
    cp_exc = cp_sym[cp_sym >= 0]
    cp_inh = cp_sym[cp_sym < 0]
    
    # overall stats
    overall_median = np.nanmedian(cp_sym)
    overall_mean = np.nanmean(cp_sym)
    
    # --- plot histogram ---
    plt.figure(figsize=(6, 4))
    bin_width = 0.02
    bins = np.arange(-1.0, 1.0 + bin_width, bin_width)
    
    # Plot excitation (red/salmon) only for values >= 0
    if len(cp_exc) > 0:
        plt.hist(cp_exc, bins=bins, facecolor='salmon',
                 edgecolor='black', alpha=0.8, label='Excitation')
    
    # Plot inhibition (blue/skyblue) only for values < 0
    if len(cp_inh) > 0:
        plt.hist(cp_inh, bins=bins, facecolor='skyblue',
                 edgecolor='black', alpha=0.8, label='Inhibition')
    
    plt.axvline(overall_median, color='black', linestyle='--', label='Median')
    plt.axvline(overall_mean, color='gray', linestyle='--', label='Mean')
    
    plt.title(f"Distribution of Symmetrized Choice Probability ({type})")
    plt.xlabel("CP′  (−1 … 1, 0 = chance)")
    plt.ylabel("Count")
    plt.xlim(-1, 1)
    plt.legend()
    plt.tight_layout()
    
    os.makedirs(save_loc, exist_ok=True)
    out_name = os.path.join(save_loc, f"choice_prob_distribution_sym_{type}.pdf")
    plt.savefig(out_name, dpi=300)
    plt.show()
    print(f"Saved histogram to {out_name}")
    np.save(os.path.join(save_loc, f'overall_median_CP_{type}.npy'), overall_median)
    np.save(os.path.join(save_loc, f'overall_mean_CP_{type}.npy'), overall_mean)
    
    # -----------------------------------------
    # D) Compute and Save Bin Percentages
    # -----------------------------------------
    # Use the cleaned medians for bin calculations
    sig_choice = medians_clean
    total_sig = len(sig_choice)
    
    # Define bins.
    # Inhibition bins (CP < 0.5)
    bin_inh_1 = np.sum((sig_choice >= 0.4) & (sig_choice < 0.5))  # "0.5-0.4"
    bin_inh_2 = np.sum((sig_choice >= 0.3) & (sig_choice < 0.4))  # "0.4-0.3"
    bin_inh_3 = np.sum((sig_choice >= 0.2) & (sig_choice < 0.3))  # "0.3-0.2"
    bin_inh_4 = np.sum(sig_choice < 0.2)  # "<0.2"
    # Excitation bins (CP >= 0.5)
    bin_exc_1 = np.sum((sig_choice >= 0.5) & (sig_choice < 0.6))  # "0.5-0.6"
    bin_exc_2 = np.sum((sig_choice >= 0.6) & (sig_choice < 0.7))  # "0.6-0.7"
    bin_exc_3 = np.sum((sig_choice >= 0.7) & (sig_choice < 0.8))  # "0.7-0.8"
    bin_exc_4 = np.sum(sig_choice >= 0.8)  # ">0.8"
    
    bins_dict = {
        "0.5-0.4": bin_inh_1,
        "0.5-0.6": bin_exc_1,
        "0.4-0.3": bin_inh_2,
        "0.6-0.7": bin_exc_2,
        "0.3-0.2": bin_inh_3,
        "0.7-0.8": bin_exc_3,
        "<0.2": bin_inh_4,
        ">0.8": bin_exc_4
    }
    
    percents = {k: (v / total_sig * 100) for k, v in bins_dict.items()}
    
    # Save percentages to CSV.
    bins_df = pd.DataFrame(list(percents.items()), columns=["bin", "percent"])
    out_csv = os.path.join(save_loc, f"choice_prob_bins_{type}.csv")
    bins_df.to_csv(out_csv, index=False)
    print(f"Saved choice probability bin percentages to {out_csv}")
    
    # (Optional) Return the computed percentages if needed.
    return percents


def analyze_choice_probability(choice_prob,
                               freq_matrix,
                               target_freqs,
                               save_loc,
                               sig_neuron=None):
    """
    1) Plots a histogram of median choice probability for "excitation" vs. "inhibition" groups.
    2) For each 'significant' neuron (specified by sig_neuron):
       - Determines the frequencies at which it is responsive
       - Extracts:
         a) Peak (max) auROC frequency -> CP
         b) Min auROC frequency       -> CP
         c) Number of responsive frequencies -> CP
       Then creates "raincloud" plots (box + half-violin + scatter) for each case.
    3) Performs Kruskal–Wallis tests to see if CP differs across the groups (frequencies or freq-counts).
       Adds significance stars in the plot title if p < 0.05, <0.01, etc.
    4) Uses a Studio Ghibli-inspired color palette, one color per group.
    5) Saves the generated figures to data_loc.
    """
    
    if sig_neuron is None:
        sig_neuron = []
    
    # -----------------------------------------
    # A) Helper function: convert p-value to stars
    # -----------------------------------------
    def p_value_to_stars(p):
        """Return a string of significance stars based on p-value."""
        if p < 0.0001:
            return "****"
        elif p < 0.001:
            return "***"
        elif p < 0.01:
            return "**"
        elif p < 0.05:
            return "*"
        else:
            return "ns"
    
    # -----------------------------------------
    # B) Studio Ghibli-like color palette
    #    (Feel free to adjust these hex codes.)
    # -----------------------------------------
    ghibli_colors = [
        "#A8C256",  # light/mossy green
        "#FAD9B1",  # pale peach
        "#F5E2CE",  # soft cream
        "#C3BCD2",  # light dusty lavender
        "#94B7BB",  # muted teal
        "#E6B19D",  # pastel terra-cotta
        "#BBD6C5",  # pale greenish mint
        "#E8C3A4",  # warm tan
        "#ADA396",  # soft gray-brown
        "#E8E2CE",  # near-white with a yellow tint
    ]
    
    # -----------------------------------------
    # C) Histogram of Exc vs. Inh median CP
    # -----------------------------------------
    n_neurons = freq_matrix.shape[0]
    
    # Compute each neuron's median CP
    choice_medians = np.array([
        np.nanmean(choice_prob['all_distributions'][i])
        for i in range(n_neurons)
    ])
    
    # Separate excitatory vs. inhibitory
    choice_prob_exc = []
    choice_prob_inh = []
    for median_val in choice_medians:
        if np.isnan(median_val):
            continue
        if median_val >= 0.5:
            choice_prob_exc.append(median_val)
        else:
            choice_prob_inh.append(median_val)
    
    # -----------------------------------------
    # D) Gather data for Peak freq, Min freq, # freq
    # -----------------------------------------
    max_freqs = []
    min_freqs = []
    cp_peak = []
    cp_min = []
    n_freqs_list = []
    cp_nfreq = []
    
    for i in sig_neuron:
        row = freq_matrix[i, :]
        valid_mask = ~np.isnan(row)
        if not np.any(valid_mask):
            continue
        
        freq_indices = np.where(valid_mask)[0]
        auroc_vals = row[valid_mask]
        these_freqs = [target_freqs[idx] for idx in freq_indices]
        
        # Peak
        argmax_ = np.argmax(auroc_vals)
        max_freqs.append(these_freqs[argmax_])
        cp_peak.append(choice_medians[i])
        
        # Min
        argmin_ = np.argmin(auroc_vals)
        min_freqs.append(these_freqs[argmin_])
        cp_min.append(choice_medians[i])
        
        # # of freq
        n_freqs_list.append(len(these_freqs))
        cp_nfreq.append(choice_medians[i])
    
    # -----------------------------------------
    # E) Raincloud: Peak freq vs. CP
    # -----------------------------------------
    freq2cp_peak = {}
    for f, val in zip(max_freqs, cp_peak):
        freq2cp_peak.setdefault(f, []).append(val)
    
    unique_peak_freqs = sorted(freq2cp_peak.keys())
    data_x_peak = [freq2cp_peak[f] for f in unique_peak_freqs]
    group_labels_peak = [str(f) for f in unique_peak_freqs]
    
    pval_peak = None
    if len(data_x_peak) > 1:
        kw_peak, p_peak_val = kruskal(*data_x_peak)
        pval_peak = p_peak_val
        print(f"[Peak freq] Kruskal–Wallis H={kw_peak:.3f}, p={p_peak_val:.3e}")
    else:
        print("Not enough distinct peak frequencies to run Kruskal–Wallis.")
    
    out_peak_file = os.path.join(save_loc, 'raincloud_peak_freq_vs_cp.png')
    raincloud_plot(data_groups=data_x_peak,
                   group_labels=group_labels_peak,
                   x_label='Choice Probability',
                   y_label='Peak Freq (Hz)',
                   title_prefix='Raincloud: CP vs. Peak Frequency',
                   out_filename=out_peak_file,
                   pval=pval_peak)
    
    # -----------------------------------------
    # G) Raincloud: Min freq vs. CP
    # -----------------------------------------
    freq2cp_min = {}
    for f, val in zip(min_freqs, cp_min):
        freq2cp_min.setdefault(f, []).append(val)
    
    unique_min_freqs = sorted(freq2cp_min.keys())
    data_x_min = [freq2cp_min[f] for f in unique_min_freqs]
    group_labels_min = [str(f) for f in unique_min_freqs]
    
    pval_min = None
    if len(data_x_min) > 1:
        kw_min, p_min_val = kruskal(*data_x_min)
        pval_min = p_min_val
        print(f"[Min freq] Kruskal–Wallis H={kw_min:.3f}, p={p_min_val:.3e}")
    else:
        print("Not enough distinct min frequencies to run Kruskal–Wallis.")
    
    out_min_file = os.path.join(save_loc, 'raincloud_min_freq_vs_cp.png')
    raincloud_plot(data_groups=data_x_min,
                   group_labels=group_labels_min,
                   x_label='Choice Probability',
                   y_label='Min Freq (Hz)',
                   title_prefix='Raincloud: CP vs. Min Frequency',
                   out_filename=out_min_file,
                   pval=pval_min)
    
    # -----------------------------------------
    # H) Raincloud: # freq responsive vs. CP
    # -----------------------------------------
    nfreq2cp = {}
    for nf, val in zip(n_freqs_list, cp_nfreq):
        nfreq2cp.setdefault(nf, []).append(val)
    
    unique_nfreqs = sorted(nfreq2cp.keys())
    data_x_nfreq = [nfreq2cp[nf] for nf in unique_nfreqs]
    group_labels_nfreq = [str(nf) for nf in unique_nfreqs]
    
    pval_nfreq = None
    if len(data_x_nfreq) > 1:
        kw_nfreq, p_nfreq_val = kruskal(*data_x_nfreq)
        pval_nfreq = p_nfreq_val
        print(f"[# freq] Kruskal–Wallis H={kw_nfreq:.3f}, p={p_nfreq_val:.3e}")
    else:
        print("Not enough distinct frequency counts to run Kruskal–Wallis.")
    
    out_nfreq_file = os.path.join(save_loc, 'raincloud_nfreq_vs_cp.png')
    raincloud_plot(data_groups=data_x_nfreq,
                   group_labels=group_labels_nfreq,
                   x_label='Choice Probability',
                   y_label='# Responsive Frequencies',
                   title_prefix='Raincloud: CP vs. # Responsive Frequencies',
                   out_filename=out_nfreq_file,
                   pval=pval_nfreq)


def raincloud_plot(data_groups,
                   group_labels,
                   x_label,
                   y_label,
                   title_prefix,
                   out_filename,
                   pval=None):
    """
    Create a raincloud plot (box + half-violin + scatter) for the given data groups.
    """
    # Helper function: convert p-value to stars
    def p_value_to_stars(p):
        if p < 0.0001:
            return "****"
        elif p < 0.001:
            return "***"
        elif p < 0.01:
            return "**"
        elif p < 0.05:
            return "*"
        else:
            return "ns"
    
    # Studio Ghibli-like color palette
    ghibli_colors = [
        "#A8C256",  # light/mossy green
        "#FAD9B1",  # pale peach
        "#F5E2CE",  # soft cream
        "#C3BCD2",  # light dusty lavender
        "#94B7BB",  # muted teal
        "#E6B19D",  # pastel terra-cotta
        "#BBD6C5",  # pale greenish mint
        "#E8C3A4",  # warm tan
        "#ADA396",  # soft gray-brown
        "#E8E2CE",  # near-white with a yellow tint
    ]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot each group
    for i, (data, label) in enumerate(zip(data_groups, group_labels)):
        color = ghibli_colors[i % len(ghibli_colors)]
        
        # Box plot
        bp = ax.boxplot(data, positions=[i], patch_artist=True, widths=0.6)
        bp['boxes'][0].set_facecolor(color)
        bp['boxes'][0].set_alpha(0.7)
        
        # Violin plot (half)
        vp = ax.violinplot(data, positions=[i], widths=0.8, showmeans=False)
        vp['bodies'][0].set_facecolor(color)
        vp['bodies'][0].set_alpha(0.3)
        
        # Scatter plot
        jitter = np.random.normal(0, 0.1, len(data))
        ax.scatter([i + jitter_val for jitter_val in jitter], data, 
                  color=color, alpha=0.6, s=20, edgecolors='black', linewidth=0.5)
    
    # Customize plot
    ax.set_xticks(range(len(group_labels)))
    ax.set_xticklabels(group_labels)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    
    # Add title with significance
    title = title_prefix
    if pval is not None:
        stars = p_value_to_stars(pval)
        title += f" (p={pval:.3e} {stars})"
    ax.set_title(title)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(out_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved raincloud plot to {out_filename}")


def analyze_choice_probability_norm(choice_prob,
                                    freq_matrix,
                                    target_freqs,
                                    save_loc,
                                    sig_neuron=None):
    """
    1) Plots a histogram of median choice probability for "excitation" vs. "inhibition" groups.
    2) For each 'significant' neuron (specified by sig_neuron):
       - Determines the frequencies at which it is responsive
       - Extracts:
         a) Peak (max) auROC frequency -> CP
         b) Min auROC frequency       -> CP
         c) Number of responsive frequencies -> CP
       Then creates "raincloud" plots (box + half-violin + scatter) for each case.
    3) Performs Kruskal–Wallis tests to see if CP differs across the groups (frequencies or freq-counts).
       Adds significance stars in the plot title if p < 0.05, <0.01, etc.
    4) Uses a Studio Ghibli-inspired color palette, one color per group.
    5) Saves the generated figures to data_loc.
    """
    
    if sig_neuron is None:
        sig_neuron = []
    
    # -----------------------------------------
    # A) Helper function: convert p-value to stars
    # -----------------------------------------
    def p_value_to_stars(p):
        """Return a string of significance stars based on p-value."""
        if p < 0.0001:
            return "****"
        elif p < 0.001:
            return "***"
        elif p < 0.01:
            return "**"
        elif p < 0.05:
            return "*"
        else:
            return "ns"
    
    # -----------------------------------------
    # B) Studio Ghibli-like color palette
    #    (Feel free to adjust these hex codes.)
    # -----------------------------------------
    ghibli_colors = [
        "#A8C256",  # light/mossy green
        "#FAD9B1",  # pale peach
        "#F5E2CE",  # soft cream
        "#C3BCD2",  # light dusty lavender
        "#94B7BB",  # muted teal
        "#E6B19D",  # pastel terra-cotta
        "#BBD6C5",  # pale greenish mint
        "#E8C3A4",  # warm tan
        "#ADA396",  # soft gray-brown
        "#E8E2CE",  # near-white with a yellow tint
    ]
    
    # -----------------------------------------
    # C) Histogram of Exc vs. Inh median CP
    # -----------------------------------------
    n_neurons = freq_matrix.shape[0]
    
    # Compute each neuron's median CP
    choice_medians = np.array([
        np.nanmean(choice_prob['all_distributions'][i])
        for i in range(n_neurons)
    ])
    
    # Separate excitatory vs. inhibitory
    choice_prob_exc = []
    choice_prob_inh = []
    for median_val in choice_medians:
        if np.isnan(median_val):
            continue
        if median_val >= 0.5:
            choice_prob_exc.append(median_val)
        else:
            choice_prob_inh.append(median_val)
    
    # -----------------------------------------
    # D) Gather data for Peak freq, Min freq, # freq
    # -----------------------------------------
    max_freqs = []
    min_freqs = []
    cp_peak = []
    cp_min = []
    n_freqs_list = []
    cp_nfreq = []
    
    for i in sig_neuron:
        row = freq_matrix[i, :]
        valid_mask = ~np.isnan(row)
        if not np.any(valid_mask):
            continue
        
        freq_indices = np.where(valid_mask)[0]
        auroc_vals = row[valid_mask]
        these_freqs = [target_freqs[idx] for idx in freq_indices]
        
        # Peak
        argmax_ = np.argmax(auroc_vals)
        max_freqs.append(these_freqs[argmax_])
        cp_peak.append(choice_medians[i])
        
        # Min
        argmin_ = np.argmin(auroc_vals)
        min_freqs.append(these_freqs[argmin_])
        cp_min.append(choice_medians[i])
        
        # # of freq
        n_freqs_list.append(len(these_freqs))
        cp_nfreq.append(choice_medians[i])
    
    # -----------------------------------------
    # E) Raincloud: Peak freq vs. CP
    # -----------------------------------------
    freq2cp_peak = {}
    for f, val in zip(max_freqs, cp_peak):
        freq2cp_peak.setdefault(f, []).append(val)
    
    unique_peak_freqs = sorted(freq2cp_peak.keys())
    data_x_peak = [freq2cp_peak[f] for f in unique_peak_freqs]
    group_labels_peak = [str(f) for f in unique_peak_freqs]
    
    pval_peak = None
    if len(data_x_peak) > 1:
        kw_peak, p_peak_val = kruskal(*data_x_peak)
        pval_peak = p_peak_val
        print(f"[Peak freq] Kruskal–Wallis H={kw_peak:.3f}, p={p_peak_val:.3e}")
    else:
        print("Not enough distinct peak frequencies to run Kruskal–Wallis.")
    
    out_peak_file = os.path.join(save_loc, 'raincloud_peak_freq_vs_cp.png')
    raincloud_plot(data_groups=data_x_peak,
                   group_labels=group_labels_peak,
                   x_label='Choice Probability',
                   y_label='Peak Freq (Hz)',
                   title_prefix='Raincloud: CP vs. Peak Frequency',
                   out_filename=out_peak_file,
                   pval=pval_peak)
    
    # -----------------------------------------
    # G) Raincloud: Min freq vs. CP
    # -----------------------------------------
    freq2cp_min = {}
    for f, val in zip(min_freqs, cp_min):
        freq2cp_min.setdefault(f, []).append(val)
    
    unique_min_freqs = sorted(freq2cp_min.keys())
    data_x_min = [freq2cp_min[f] for f in unique_min_freqs]
    group_labels_min = [str(f) for f in unique_min_freqs]
    
    pval_min = None
    if len(data_x_min) > 1:
        kw_min, p_min_val = kruskal(*data_x_min)
        pval_min = p_min_val
        print(f"[Min freq] Kruskal–Wallis H={kw_min:.3f}, p={p_min_val:.3e}")
    else:
        print("Not enough distinct min frequencies to run Kruskal–Wallis.")
    
    out_min_file = os.path.join(save_loc, 'raincloud_min_freq_vs_cp.png')
    raincloud_plot(data_groups=data_x_min,
                   group_labels=group_labels_min,
                   x_label='Choice Probability',
                   y_label='Min Freq (Hz)',
                   title_prefix='Raincloud: CP vs. Min Frequency',
                   out_filename=out_min_file,
                   pval=pval_min)
    
    # -----------------------------------------
    # H) Raincloud: # freq responsive vs. CP
    # -----------------------------------------
    nfreq2cp = {}
    for nf, val in zip(n_freqs_list, cp_nfreq):
        nfreq2cp.setdefault(nf, []).append(val)
    
    unique_nfreqs = sorted(nfreq2cp.keys())
    data_x_nfreq = [nfreq2cp[nf] for nf in unique_nfreqs]
    group_labels_nfreq = [str(nf) for nf in unique_nfreqs]
    
    pval_nfreq = None
    if len(data_x_nfreq) > 1:
        kw_nfreq, p_nfreq_val = kruskal(*data_x_nfreq)
        pval_nfreq = p_nfreq_val
        print(f"[# freq] Kruskal–Wallis H={kw_nfreq:.3f}, p={p_nfreq_val:.3e}")
    else:
        print("Not enough distinct freq-count groups to run Kruskal–Wallis.")
    
    out_nfreq_file = os.path.join(save_loc, 'raincloud_nfreq_vs_cp.png')
    raincloud_plot(data_groups=data_x_nfreq,
                   group_labels=group_labels_nfreq,
                   x_label='Choice Probability',
                   y_label='# Frequencies Responsive',
                   title_prefix='Raincloud: CP vs. # Freq Responsive',
                   out_filename=out_nfreq_file,
                   pval=pval_nfreq)


def compute_latency(mean_tc, baseline_frames, rec_hz,
                    window_size=5, threshold_factor=2.0):
    """
    Compute the latency of a response given an average time course.

    Parameters
    ----------
    mean_tc : 1D numpy array
        The average time course (mean dF/F over trials).
    baseline_frames : int
        Number of frames to consider as baseline (e.g., pre-trial period).
    rec_hz : float
        Imaging rate in Hz, for converting frames to seconds.
    window_size : int, optional
        Size of the centered rolling window (in frames) for smoothing.
    threshold_factor : float, optional
        Number of standard deviations above baseline mean used as threshold.

    Returns
    -------
    latency_sec : float
        The time (in seconds) at which the mean_tc first exceeds threshold
        in its rolling average. If no crossing is found, returns np.nan.
    latency_frame : int
        The same, but in units of frames (index).
    roll_means : np.ndarray
        The rolling average trace used for threshold detection (same length as mean_tc).
    """
    # Baseline statistics
    baseline_mean = np.mean(mean_tc[baseline_frames:baseline_frames + 1 * rec_hz])
    baseline_std = np.std(mean_tc[baseline_frames:baseline_frames + 1 * rec_hz])
    threshold = baseline_mean + threshold_factor * baseline_std
    threshold_inhibit = baseline_mean - threshold_factor * baseline_std

    # Compute rolling average (centered)
    # rolling average of the same length as mean_tc
    pad = window_size // 2
    padded_tc = np.pad(mean_tc, (pad, pad), mode='edge')
    kernel = np.ones(window_size) / window_size
    tmp_conv = np.convolve(padded_tc, kernel, mode='valid')
    # 'tmp_conv' is the rolling average for each frame in mean_tc
    roll_means = tmp_conv

    # Find the first time point (frame) after baseline where rolling mean > threshold
    latency_frame = None
    max_latency_frames = baseline_frames + (6 * rec_hz)  # Convert time limit to frames

    for frame_idx in range(len(roll_means)):
        # Only look after the Opto onset
        if frame_idx >= baseline_frames + 1 * rec_hz:
            if frame_idx > max_latency_frames:
                break  # Stop checking if we exceed the allowed latency range
            if roll_means[frame_idx] > threshold:
                latency_frame = frame_idx + (window_size // 2)
                break
            elif roll_means[frame_idx] < threshold_inhibit:
                latency_frame = frame_idx + (window_size // 2)
                break

    # Convert frame index to seconds
    if latency_frame is not None:
        latency_sec = latency_frame / rec_hz
    else:
        latency_sec = np.nan

    return latency_sec, latency_frame, roll_means


def compute_and_plot_latencies(
        dff_run_list_sub, freq_indices, freq_matrix,
        target_freqs, pre_trial_length, rec_hz,
        save_loc, window_size=5, threshold_factor=2.0
):
    """
    For each neuron, for each significant frequency:
      1) Compute the average response (mean_tc).
      2) Compute the latency for that mean_tc.
      3) Plot the mean_tc with a vertical line or marker at that latency.
      4) Save the plot.

    Returns a DataFrame with columns: ["neuron", "frequency", "latency_sec"]
    for all significant neuron–frequency pairs.
    """
    # Ensure directory exists for saving the single-neuron validation plots
    plot_dir = os.path.join(save_loc, "latency_validation_plots")
    os.makedirs(plot_dir, exist_ok=True)

    latencies = []
    baseline_frames = int(pre_trial_length)  # number of frames in pre-trial

    n_neurons = len(dff_run_list_sub)
    for i in range(n_neurons):
        sig_freq_indices = np.where(~np.isnan(freq_matrix[i, :]))[0]
        if len(sig_freq_indices) == 0:
            continue

        for col in sig_freq_indices:
            freq = target_freqs[col]
            trial_idxs = freq_indices.get(freq, [])
            if len(trial_idxs) == 0:
                continue

            # Gather trials for neuron i, freq col
            selected_trials = [dff_run_list_sub[i][j] for j in trial_idxs]
            if len(selected_trials) == 0:
                continue

            heatmap_data = pd.DataFrame(selected_trials)
            mean_tc = np.nanmean(heatmap_data.values, axis=0)

            # Compute latency
            lat_sec, lat_frame, roll_means = compute_latency(
                mean_tc,
                baseline_frames=baseline_frames,
                rec_hz=rec_hz,
                window_size=window_size,
                threshold_factor=threshold_factor
            )
            latencies.append([i, freq, lat_sec])

            # -----------------------------
            #  PLOT single-neuron validation
            # -----------------------------
            time = np.arange(len(mean_tc))
            time_sec = time / rec_hz

            plt.figure()
            # Plot the raw mean time course
            plt.plot(time_sec, mean_tc, label='Mean dF/F')
            # Plot the rolling average
            plt.plot(time_sec, roll_means, label='Rolling Avg', linestyle='--')

            # If we found a latency, add a marker/vertical line
            if not np.isnan(lat_sec):
                plt.axvline(lat_sec, color='red', linestyle=':', label=f'Latency = {lat_sec:.2f}s')

            # Add a vertical line at the stimulus onset (pre_trial_length in seconds)
            plt.axvline(pre_trial_length / rec_hz, color='gray', linestyle='-.', label='distractor light onset')
            plt.axvline(((pre_trial_length) / rec_hz) + 1, color='gray', linestyle='-.', label='Opto onset')

            plt.title(f'Neuron {i}, {freq} Hz\nLatency={lat_sec - 5:.2f} s')
            plt.xlabel("Time (s)")
            plt.ylabel("Mean dF/F")
            plt.legend()

            # Save this figure
            out_name = f"neuron_{i}_freq_{freq}_latency_validation.png"
            plt.savefig(os.path.join(plot_dir, out_name), dpi=300)
            plt.close()

    # Return DataFrame of latencies
    df_latencies = pd.DataFrame(latencies, columns=["neuron", "frequency", "latency_sec"])
    return df_latencies


def compute_and_plot_latencies_norm(
        dff_run_list_sub, freq_indices, freq_matrix,
        target_freqs, pre_trial_length, rec_hz,
        save_loc, window_size=5, threshold_factor=2.0, num_freqs=2
):
    """
    For each neuron that is responsive to at least 3 frequencies:
      1) For each significant frequency, compute the average response (mean_tc) and latency.
      2) Normalize the latencies by dividing each by the maximum latency for that neuron.
      3) Re-scale the time axis by the neuron's maximum latency and plot the mean_tc
         with a vertical line marking the normalized latency.
      4) Save the plot.

    Returns a DataFrame with columns: ["neuron", "frequency", "latency_sec"]
    where latency_sec is the normalized latency.
    """
    # Ensure directory exists for saving the single-neuron validation plots.
    plot_dir = os.path.join(save_loc, "latency_validation_plots")
    os.makedirs(plot_dir, exist_ok=True)

    latencies = []
    baseline_frames = int(pre_trial_length)  # number of frames in pre-trial
    n_neurons = len(dff_run_list_sub)

    for i in range(n_neurons):
        # Identify significant frequencies for neuron i.
        sig_freq_indices = np.where(~np.isnan(freq_matrix[i, :]))[0]
        if len(sig_freq_indices) < num_freqs:
            continue  # Only process neurons with >=3 responses.

        # Collect computed latency data for this neuron.
        neuron_data = []
        for col in sig_freq_indices:
            freq = target_freqs[col]
            trial_idxs = freq_indices.get(freq, [])
            if len(trial_idxs) == 0:
                continue

            # Gather trials for neuron i, frequency freq.
            selected_trials = [dff_run_list_sub[i][j] for j in trial_idxs]
            if len(selected_trials) == 0:
                continue

            heatmap_data = pd.DataFrame(selected_trials)
            mean_tc = np.nanmean(heatmap_data.values, axis=0)

            # Response latency (compute_latency)
            lat_sec, lat_frame, roll_means = compute_latency(
                mean_tc,
                baseline_frames=baseline_frames,
                rec_hz=rec_hz,
                window_size=window_size,
                threshold_factor=threshold_factor
            )
            neuron_data.append({
                "freq": freq,
                "lat_sec": lat_sec,
                "mean_tc": mean_tc,
                "roll_means": roll_means
            })

        # Skip if after filtering there are less than 3 frequency responses.
        if len(neuron_data) < num_freqs:
            continue

        # Determine the maximum valid (non-NaN) latency for this neuron.
        valid_latencies = [d["lat_sec"] for d in neuron_data if not np.isnan(d["lat_sec"])]
        if len(valid_latencies) == 0:
            continue  # no valid latency found
        max_latency = max(valid_latencies)
        # Prevent division by zero.
        if max_latency == 0:
            max_latency = 1

        # For each frequency response, compute the normalized latency and plot.
        for d in neuron_data:
            freq = d["freq"]
            raw_latency = d["lat_sec"]
            norm_latency = raw_latency / max_latency if not np.isnan(raw_latency) else np.nan
            latencies.append([i, freq, norm_latency])

            # Re-scale the time axis.
            time = np.arange(len(d["mean_tc"]))
            time_sec = time / rec_hz
            norm_time = time_sec / max_latency

            plt.figure()
            plt.plot(norm_time, d["mean_tc"], label='Mean dF/F')
            plt.plot(norm_time, d["roll_means"], label='Rolling Avg', linestyle='--')

            # Plot the normalized latency as a vertical line.
            if not np.isnan(raw_latency):
                plt.axvline(raw_latency / max_latency, color='red', linestyle=':',
                            label=f'Norm Latency = {norm_latency:.2f}')
                plt.text(raw_latency / max_latency, np.max(d["mean_tc"]), f'{norm_latency:.2f}',
                         color='red', fontsize=9)

            # Re-scale the stimulus onset markers.
            pre_trial_norm = (pre_trial_length / rec_hz) / max_latency
            plt.axvline(pre_trial_norm, color='gray', linestyle='-.', label='distractor light onset')
            plt.axvline(((pre_trial_length / rec_hz) + 1) / max_latency, color='gray', linestyle='-.',
                        label='Opto onset')

            plt.title(f'Neuron {i}, {freq} Hz\nRaw Latency = {raw_latency:.2f}s, Normalized = {norm_latency:.2f}')
            plt.xlabel("Normalized Time (max latency = 1)")
            plt.ylabel("Mean dF/F")
            plt.legend()

            out_name = f"neuron_{i}_freq_{freq}_latency_validation.png"
            plt.savefig(os.path.join(plot_dir, out_name), dpi=300)
            plt.close()

    # Return DataFrame of normalized latencies.
    df_latencies = pd.DataFrame(latencies, columns=["neuron", "frequency", "latency_sec"])
    return df_latencies


def plot_latency_vs_frequency(df_latencies, save_fig=False, out_path="latency_vs_freq.png"):
    """
    Creates a 'raincloud' plot (vertical manual IQR box + half-violin + jittered scatter)
    of (latency_sec - 5) vs. frequency, in shades of red.
    Darker red corresponds to higher frequency.

    This version manually draws the IQR box so that even if the quartile range is small,
    a visible box (or line with minimal thickness) is rendered.
    """

    # 1) Identify unique frequencies (sorted) and assign x positions.
    sorted_freqs = np.sort(df_latencies["frequency"].unique())
    positions = np.arange(1, len(sorted_freqs) + 1)

    # 2) Build list-of-lists: each element is (latency_sec - 5) for that frequency.
    data_x = [
        (df_latencies.loc[df_latencies["frequency"] == freq, "latency_sec"].values - 5)
        for freq in sorted_freqs
    ]

    # 3) Setup figure and axis.
    fig, ax = plt.subplots(figsize=(8, 4))

    # 4) Define colormap and normalization.
    cmap = plt.get_cmap("Reds")
    norm = plt.Normalize(vmin=sorted_freqs[0], vmax=sorted_freqs[-1])

    # 5) Manually compute and draw half-violin for each frequency group.
    for i, arr in enumerate(data_x):
        pos = positions[i]
        color = cmap(norm(sorted_freqs[i]))

        # Filter non-finite values.
        arr = arr[np.isfinite(arr)]
        if len(arr) < 2:
            continue

        # Compute kernel density estimate.
        kde = gaussian_kde(arr)
        y_min, y_max = np.min(arr), np.max(arr)
        y_vals = np.linspace(y_min, y_max, 200)
        densities = kde(y_vals)

        # Scale the density so that the half-violin width is a fraction of the total width.
        # Here, maximum width is 0.25.
        max_density = np.max(densities)
        scaled_densities = 0.25 * densities / max_density

        # Create polygon coordinates for the half-violin (right side of center).
        x_coords = pos + scaled_densities
        # Build a closed polygon: density curve then back to the center.
        poly_x = np.concatenate([x_coords, [pos], [pos]])
        poly_y = np.concatenate([y_vals, [y_vals[-1]], [y_vals[0]]])
        ax.fill(poly_x, poly_y, color=color, alpha=0.4, zorder=1)

    # 6) Manually compute and draw the box (IQR) for each frequency group.
    for i, arr in enumerate(data_x):
        pos = positions[i]
        color = cmap(norm(sorted_freqs[i]))
        arr = arr[np.isfinite(arr)]
        if len(arr) < 1:
            continue

        # Compute quartiles.
        q1, med, q3 = np.percentile(arr, [25, 50, 75])
        iqr = q3 - q1

        # If IQR is extremely small (or zero), force a minimal box height.
        if np.isclose(iqr, 0):
            data_range = np.max(arr) - np.min(arr)
            # Use 5% of the range or a fixed small value if range is 0.
            min_height = 0.05 * data_range if data_range > 0 else 0.1
            q1 = med - min_height
            q3 = med + min_height

        box_width = 0.3  # Total width of the box.
        # Draw the box as a rectangle.
        rect = patches.Rectangle(
            (pos - box_width / 2, q1),  # bottom left corner
            box_width,  # width
            q3 - q1,  # height
            facecolor=color,
            edgecolor='k',
            alpha=0.6,
            linewidth=1,
            zorder=2
        )
        ax.add_patch(rect)
        # Draw the median line.
        ax.plot([pos - box_width / 2, pos + box_width / 2], [med, med],
                color='k', linewidth=1.5, zorder=3)

    # 7) Draw jittered scatter points on top.
    for i, arr in enumerate(data_x):
        pos = positions[i]
        color = cmap(norm(sorted_freqs[i]))
        arr = arr[np.isfinite(arr)]
        x_jitter = pos + np.random.uniform(low=-0.05, high=0.05, size=len(arr))
        ax.scatter(x_jitter, arr, s=8, color=color, alpha=0.7, zorder=4)

    # 8) Cosmetic settings.
    ax.set_xticks(positions)
    ax.set_xticklabels([str(freq) for freq in sorted_freqs])
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Latency (s)")
    ax.set_title("")
    plt.tight_layout()

    if save_fig:
        plt.savefig(out_path, dpi=300)
    # plt.show()


def plot_latency_vs_frequency_norm(df_latencies, save_fig=False, out_path="latency_vs_freq.png"):
    """
    Creates a 'raincloud' plot (vertical manual IQR box + half-violin + jittered scatter)
    of the normalized latency vs. frequency. Darker red corresponds to higher frequency.
    """
    # 1) Identify unique frequencies (sorted) and assign x positions.
    sorted_freqs = np.sort(df_latencies["frequency"].unique())
    positions = np.arange(1, len(sorted_freqs) + 1)

    # 2) Build list-of-lists: each element is the normalized latency for that frequency.
    data_x = [
        df_latencies.loc[df_latencies["frequency"] == freq, "latency_sec"].values
        for freq in sorted_freqs
    ]

    # 3) Setup figure and axis.
    fig, ax = plt.subplots(figsize=(8, 4))

    # 4) Define colormap and normalization.
    cmap = plt.get_cmap("Reds")
    norm = plt.Normalize(vmin=sorted_freqs[0], vmax=sorted_freqs[-1])

    # 5) Manually compute and draw half-violin for each frequency group.
    for i, arr in enumerate(data_x):
        pos = positions[i]
        color = cmap(norm(sorted_freqs[i]))

        arr = arr[np.isfinite(arr)]
        if len(arr) < 2:
            continue

        kde = gaussian_kde(arr)
        y_min, y_max = np.min(arr), np.max(arr)
        y_vals = np.linspace(y_min, y_max, 200)
        densities = kde(y_vals)
        max_density = np.max(densities)
        scaled_densities = 0.25 * densities / max_density
        x_coords = pos + scaled_densities
        poly_x = np.concatenate([x_coords, [pos], [pos]])
        poly_y = np.concatenate([y_vals, [y_vals[-1]], [y_vals[0]]])
        ax.fill(poly_x, poly_y, color=color, alpha=0.5, zorder=1)

    # 6) Manually compute and draw the box (IQR) for each frequency group.
    for i, arr in enumerate(data_x):
        pos = positions[i]
        color = cmap(norm(sorted_freqs[i]))
        arr = arr[np.isfinite(arr)]
        if len(arr) < 1:
            continue

        q1, med, q3 = np.percentile(arr, [25, 50, 75])
        iqr = q3 - q1
        if np.isclose(iqr, 0):
            data_range = np.max(arr) - np.min(arr)
            min_height = 0.05 * data_range if data_range > 0 else 0.1
            q1 = med - min_height
            q3 = med + min_height

        box_width = 0.3
        rect = patches.Rectangle(
            (pos - box_width / 2, q1),
            box_width,
            q3 - q1,
            facecolor=color,
            edgecolor='k',
            alpha=0.6,
            linewidth=1,
            zorder=2
        )
        ax.add_patch(rect)
        ax.plot([pos - box_width / 2, pos + box_width / 2], [med, med],
                color='k', linewidth=1.5, zorder=3)

    # 7) Draw jittered scatter points on top.
    for i, arr in enumerate(data_x):
        pos = positions[i]
        color = cmap(norm(sorted_freqs[i]))
        arr = arr[np.isfinite(arr)]
        x_jitter = pos + np.random.uniform(low=-0.05, high=0.05, size=len(arr))
        ax.scatter(x_jitter, arr, s=8, color=color, alpha=0.8, zorder=4)

    ax.set_xticks(positions)
    ax.set_xticklabels([str(freq) for freq in sorted_freqs])
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Normalized Latency")
    ax.set_title("Latency vs Frequency (Normalized per Neuron)")
    plt.tight_layout()

    if save_fig:
        plt.savefig(out_path, dpi=300)
    # plt.show()


def plot_fano_factor_vs_frequency(df_fano, save_fig=False, out_path="fano_factor_vs_freq.png"):
    """
    Creates a 'raincloud' plot (vertical manual IQR box + half-violin + jittered scatter)
    of the Fano factor vs. frequency. Darker red corresponds to higher frequency.
    """

    # 1) Identify unique frequencies (sorted) and assign x positions.
    sorted_freqs = np.sort(df_fano["frequency"].unique())
    positions = np.arange(1, len(sorted_freqs) + 1)

    # 2) Build list-of-lists: each element is the Fano factor for that frequency.
    data_y = [
        df_fano.loc[df_fano["frequency"] == freq, "fano_factor"].values
        for freq in sorted_freqs
    ]

    # 3) Setup figure and axis.
    fig, ax = plt.subplots(figsize=(8, 4))

    # 4) Define colormap and normalization.
    cmap = plt.get_cmap("Reds")
    norm = plt.Normalize(vmin=sorted_freqs[0], vmax=sorted_freqs[-1])

    # 5) Manually compute and draw half-violin for each frequency group.
    for i, arr in enumerate(data_y):
        pos = positions[i]
        color = cmap(norm(sorted_freqs[i]))

        arr = arr[np.isfinite(arr)]
        if len(arr) < 2:
            continue

        kde = gaussian_kde(arr)
        y_min, y_max = np.min(arr), np.max(arr)
        y_vals = np.linspace(y_min, y_max, 200)
        densities = kde(y_vals)
        max_density = np.max(densities)
        scaled_densities = 0.25 * densities / max_density
        x_coords = pos + scaled_densities
        poly_x = np.concatenate([x_coords, [pos], [pos]])
        poly_y = np.concatenate([y_vals, [y_vals[-1]], [y_vals[0]]])
        ax.fill(poly_x, poly_y, color=color, alpha=0.5, zorder=1)

    # 6) Manually compute and draw the box (IQR) for each frequency group.
    for i, arr in enumerate(data_y):
        pos = positions[i]
        color = cmap(norm(sorted_freqs[i]))
        arr = arr[np.isfinite(arr)]
        if len(arr) < 1:
            continue

        q1, med, q3 = np.percentile(arr, [25, 50, 75])
        iqr = q3 - q1
        if np.isclose(iqr, 0):
            data_range = np.max(arr) - np.min(arr)
            min_height = 0.05 * data_range if data_range > 0 else 0.1
            q1 = med - min_height
            q3 = med + min_height

        box_width = 0.3
        rect = patches.Rectangle(
            (pos - box_width / 2, q1),
            box_width,
            q3 - q1,
            facecolor=color,
            edgecolor='k',
            alpha=0.6,
            linewidth=1,
            zorder=2
        )
        ax.add_patch(rect)
        ax.plot([pos - box_width / 2, pos + box_width / 2], [med, med],
                color='k', linewidth=1.5, zorder=3)

    # 7) Draw jittered scatter points on top.
    for i, arr in enumerate(data_y):
        pos = positions[i]
        color = cmap(norm(sorted_freqs[i]))
        arr = arr[np.isfinite(arr)]
        x_jitter = pos + np.random.uniform(low=-0.05, high=0.05, size=len(arr))
        ax.scatter(x_jitter, arr, s=8, color=color, alpha=0.8, zorder=4)

    ax.set_xticks(positions)
    ax.set_xticklabels([str(freq) for freq in sorted_freqs])
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Fano Factor")
    ax.set_title("Fano Factor vs Frequency")
    plt.tight_layout()

    if save_fig:
        plt.savefig(out_path, dpi=300)
    # plt.show()


def plot_fano_factor_vs_frequency_norm(df_fano, save_fig=False, out_path="fano_factor_vs_freq.png"):
    """
    Creates a 'raincloud' plot (vertical manual IQR box + half-violin + jittered scatter)
    of the Fano factor vs. frequency. Darker red corresponds to higher frequency.
    """

    # 1) Identify unique frequencies (sorted) and assign x positions.
    sorted_freqs = np.sort(df_fano["frequency"].unique())
    positions = np.arange(1, len(sorted_freqs) + 1)

    # 2) Build list-of-lists: each element is the Fano factor for that frequency.
    data_y = [
        df_fano.loc[df_fano["frequency"] == freq, "norm_fano_factor"].values
        for freq in sorted_freqs
    ]

    # 3) Setup figure and axis.
    fig, ax = plt.subplots(figsize=(8, 4))

    # 4) Define colormap and normalization.
    cmap = plt.get_cmap("Reds")
    norm = plt.Normalize(vmin=sorted_freqs[0], vmax=sorted_freqs[-1])

    # 5) Manually compute and draw half-violin for each frequency group.
    for i, arr in enumerate(data_y):
        pos = positions[i]
        color = cmap(norm(sorted_freqs[i]))

        arr = arr[np.isfinite(arr)]
        if len(arr) < 2:
            continue

        kde = gaussian_kde(arr)
        y_min, y_max = np.min(arr), np.max(arr)
        y_vals = np.linspace(y_min, y_max, 200)
        densities = kde(y_vals)
        max_density = np.max(densities)
        scaled_densities = 0.25 * densities / max_density
        x_coords = pos + scaled_densities
        poly_x = np.concatenate([x_coords, [pos], [pos]])
        poly_y = np.concatenate([y_vals, [y_vals[-1]], [y_vals[0]]])
        ax.fill(poly_x, poly_y, color=color, alpha=0.5, zorder=1)

    # 6) Manually compute and draw the box (IQR) for each frequency group.
    for i, arr in enumerate(data_y):
        pos = positions[i]
        color = cmap(norm(sorted_freqs[i]))
        arr = arr[np.isfinite(arr)]
        if len(arr) < 1:
            continue

        q1, med, q3 = np.percentile(arr, [25, 50, 75])
        iqr = q3 - q1
        if np.isclose(iqr, 0):
            data_range = np.max(arr) - np.min(arr)
            min_height = 0.05 * data_range if data_range > 0 else 0.1
            q1 = med - min_height
            q3 = med + min_height

        box_width = 0.3
        rect = patches.Rectangle(
            (pos - box_width / 2, q1),
            box_width,
            q3 - q1,
            facecolor=color,
            edgecolor='k',
            alpha=0.6,
            linewidth=1,
            zorder=2
        )
        ax.add_patch(rect)
        ax.plot([pos - box_width / 2, pos + box_width / 2], [med, med],
                color='k', linewidth=1.5, zorder=3)

    # 7) Draw jittered scatter points on top.
    for i, arr in enumerate(data_y):
        pos = positions[i]
        color = cmap(norm(sorted_freqs[i]))
        arr = arr[np.isfinite(arr)]
        x_jitter = pos + np.random.uniform(low=-0.05, high=0.05, size=len(arr))
        ax.scatter(x_jitter, arr, s=8, color=color, alpha=0.8, zorder=4)

    ax.set_xticks(positions)
    ax.set_xticklabels([str(freq) for freq in sorted_freqs])
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Fano Factor (norm)")
    ax.set_title("Fano Factor vs Frequency")
    plt.tight_layout()

    if save_fig:
        plt.savefig(out_path, dpi=300)
    # plt.show()


def compute_and_plot_fano_factor_normalized(
        dff_run_list_sub, freq_indices, freq_matrix,
        target_freqs, pre_trial_length, rec_hz,
        save_loc, num_freqs=2
):
    """
    For each neuron that is responsive to at least `num_freqs` frequencies:
      1) For each significant frequency, extract the average dF/F value from each trial
         in the stimulation window (from pre_trial_length + 2 sec to pre_trial_length + 5 sec).
      2) Compute the Fano factor as variance/mean of these trial-averaged responses.
      3) Normalize the Fano factors for that neuron by dividing each by the maximum Fano factor.
      4) Plot the individual trial averages and mark the normalized Fano factor.
      5) Save the plot.

    Returns a DataFrame with columns: ["neuron", "frequency", "norm_fano_factor"]
    where norm_fano_factor is the Fano factor normalized by the maximum Fano factor for that neuron.
    """

    # Directory for saving plots.
    plot_dir = os.path.join(save_loc, "fano_factor_validation_plots")
    os.makedirs(plot_dir, exist_ok=True)

    results = []
    n_neurons = len(dff_run_list_sub)

    # Define stimulation window indices.
    # Note: This assumes pre_trial_length is in seconds.
    stim_start = int(pre_trial_length + (2 * rec_hz))
    stim_end = int(pre_trial_length + (5 * rec_hz))

    for i in range(n_neurons):
        # Identify significant frequencies for neuron i.
        sig_freq_indices = np.where(~np.isnan(freq_matrix[i, :]))[0]
        if len(sig_freq_indices) < num_freqs:
            continue  # Skip neurons with fewer than the required number of responses.

        # For each frequency, compute the Fano factor.
        neuron_results = []
        for col in sig_freq_indices:
            freq = target_freqs[col]
            trial_idxs = freq_indices.get(freq, [])
            if len(trial_idxs) == 0:
                continue

            # Extract trials for the current neuron and frequency.
            selected_trials = [dff_run_list_sub[i][j] for j in trial_idxs]
            if len(selected_trials) == 0:
                continue

            # For each trial, average the dF/F values in the stimulation window.
            trial_averages = []
            for trial in selected_trials:
                # Ensure the trial is long enough.
                if len(trial) < stim_end:
                    continue
                avg_val = np.nanmean(trial[stim_start:stim_end])
                trial_averages.append(avg_val)

            # Only compute Fano factor if we have trial data.
            if len(trial_averages) == 0:
                continue

            trial_averages = np.array(trial_averages)
            mean_val = abs(np.nanmean(trial_averages))
            var_val = np.nanvar(trial_averages)
            # Avoid division by zero.
            fano_factor = var_val / mean_val if mean_val != 0 else np.nan

            neuron_results.append({
                "frequency": freq,
                "trial_averages": trial_averages,
                "mean_val": mean_val,
                "fano_factor": fano_factor
            })

        if len(neuron_results) == 0:
            continue

        # Find maximum valid Fano factor for normalization.
        valid_fanos = [item["fano_factor"] for item in neuron_results if not np.isnan(item["fano_factor"])]
        if len(valid_fanos) == 0:
            continue
        max_fano = max(valid_fanos)
        # Avoid division by zero.
        if max_fano == 0:
            max_fano = 1

        # Now, for each frequency response, normalize Fano factor and plot.
        for item in neuron_results:
            freq = item["frequency"]
            raw_fano = item["fano_factor"]
            norm_fano = raw_fano / max_fano if not np.isnan(raw_fano) else np.nan

            results.append([i, freq, norm_fano])

            # Plot the individual trial averages and mark the normalized Fano factor.
            plt.figure()
            plt.plot(item["trial_averages"], 'o', label='Trial Averages')
            plt.axhline(item["mean_val"], color='red', linestyle='--', label=f'Mean = {item["mean_val"]:.2f}')
            plt.title(f'Neuron {i}, {freq} Hz\nNormalized Fano Factor = {norm_fano:.2f}')
            plt.xlabel("Trial")
            plt.ylabel("Average dF/F in Stim Window")
            plt.legend()

            out_name = f"neuron_{i}_freq_{freq}_norm_fano_factor.png"
            plt.savefig(os.path.join(plot_dir, out_name), dpi=300)
            plt.close()

    # Return a DataFrame of the computed normalized Fano factors.
    df_fano_norm = pd.DataFrame(results, columns=["neuron", "frequency", "norm_fano_factor"])
    return df_fano_norm


def raincloud_plot_two_groups_horizontal_overlay(cp_gaussian, cp_sigmoid, cp_others,
                                                 save_fig=False,
                                                 out_path="CP_raincloud_horizontal_overlay_tight.png"):
    """
    Creates an overlaid horizontal raincloud plot for two groups (Gaussian and Sigmoid)
    with overlapping density clouds and summary statistics (boxplots and scatter points)
    arranged at different vertical positions, with a tighter layout.

    The x-axis represents Choice Probability.
      - Both groups' density clouds are overlaid at the same vertical level.
      - The boxplots and individual data points are drawn separately:
          * Gaussian summary at one y position.
          * Sigmoid summary at a lower y position.

    Parameters:
      cp_gaussian : array-like
          Choice probability values for the Gaussian group.
      cp_sigmoid : array-like
          Choice probability values for the Sigmoid group.
      save_fig : bool, optional
          If True, saves the figure to out_path.
      out_path : str, optional
          Path for saving the figure.
    """
    # Convert inputs to numpy arrays and filter out non-finite values.
    cp_gaussian = np.array(cp_gaussian)
    cp_sigmoid = np.array(cp_sigmoid)
    cp_others = np.array(cp_others)
    cp_gaussian = cp_gaussian[np.isfinite(cp_gaussian)]
    cp_sigmoid = cp_sigmoid[np.isfinite(cp_sigmoid)]
    cp_others = cp_others[np.isfinite(cp_others)]

    # Set up the figure.
    fig, ax = plt.subplots(figsize=(8, 6))

    # --- Density (raincloud) parameters ---
    # Adjusted parameters for a tighter layout.
    density_y_baseline = 0.5  # Lower the baseline for the density cloud.
    density_height = 0.2  # Maximum vertical extent of the density cloud.
    # Define a common x-range (Choice Probability) for the KDEs.
    x_min = min(np.min(cp_gaussian), np.min(cp_sigmoid))
    x_max = max(np.max(cp_gaussian), np.max(cp_sigmoid))
    x_range = np.linspace(x_min, x_max, 200)

    def plot_density_cloud(data, color):
        """Compute a KDE for the data and plot a density cloud above a common baseline."""
        if len(data) < 2:
            return
        kde = gaussian_kde(data)
        densities = kde(x_range)
        # Scale densities so that the maximum vertical extent equals density_height.
        scaled_dens = densities / np.max(densities) * density_height
        # Build a polygon that starts at the baseline, follows the top of the density,
        # then returns along the baseline.
        poly_x = np.concatenate([x_range, x_range[::-1]])
        poly_y = np.concatenate([np.full_like(x_range, density_y_baseline),
                                 (density_y_baseline + scaled_dens)[::-1]])
        ax.fill(poly_x, poly_y, color=color, alpha=0.4, zorder=1)

    # Plot both densities overlaid.
    plot_density_cloud(cp_gaussian, color='tan')
    plot_density_cloud(cp_sigmoid, color='lightsteelblue')
    plot_density_cloud(cp_others, color='lightgray')

    # --- Summary Statistics (Boxplots and Scatter) ---
    # Adjusted positions for a tighter layout.
    summary_y_gaussian = 0.45  # y-coordinate for Gaussian summary stats.
    summary_y_others = 0.4  # y-coordinate for other summary stats.
    summary_y_sigmoid = 0.35  # y-coordinate for Sigmoid summary stats.

    box_height = 0.03  # Vertical thickness of the box.

    def plot_boxplot(data, y_center, color):
        """Plot a horizontal boxplot for data along the x-axis at a given y_center."""
        if len(data) < 2:
            return
        q1, med, q3 = np.percentile(data, [25, 50, 75])
        # Draw a rectangle spanning from q1 to q3.
        rect = plt.Rectangle((q1, y_center - box_height / 2),
                             q3 - q1, box_height,
                             color=color, alpha=0.6, zorder=2)
        ax.add_patch(rect)
        # Draw a vertical line at the median.
        ax.plot([med, med], [y_center - box_height / 2, y_center + box_height / 2],
                color='k', zorder=3)

    def plot_scatter(data, y_center, color):
        """Plot jittered scatter points for data along the x-axis at a given y_center."""
        jitter = np.random.uniform(-0.005, 0.005, size=len(data))
        ax.scatter(data, np.full_like(data, y_center) + jitter,
                   color=color, alpha=0.7, s=10, zorder=4)

    # Plot boxplots and scatter for each group at their respective vertical positions.
    plot_boxplot(cp_gaussian, summary_y_gaussian, color='tan')
    plot_boxplot(cp_others, summary_y_others, color='lightgray')
    plot_boxplot(cp_sigmoid, summary_y_sigmoid, color='lightsteelblue')
    plot_scatter(cp_gaussian, summary_y_gaussian, color='tan')
    plot_scatter(cp_others, summary_y_others, color='lightgray')
    plot_scatter(cp_sigmoid, summary_y_sigmoid, color='lightsteelblue')

    # --- Aesthetics ---
    ax.set_xlabel("Choice Probability")
    ax.set_ylabel("")  # Vertical space is used only for layout.
    ax.set_yticks([])  # Hide y ticks.
    ax.set_title("Gaussian vs. Sigmoid Choice Probability")
    # Legend for the two groups.
    ax.legend(["Gaussian", "Sigmoid", "Other"], loc='upper right')
    plt.tight_layout()
    if save_fig:
        plt.savefig(out_path, dpi=300)
    plt.show()


def compute_opto_windows_mean_activity(
        all_CR,  # list of shape (#neurons,) each -> (#trials, 2)
        all_FA,  # list of shape (#neurons,) each -> (#trials,)
        n_fa,  # how many trials to subsample each time
        n_boot=1000,  # how many subsampling iterations
        random_state=42
):
    if random_state is not None:
        np.random.seed(random_state)

    n_neurons = len(all_CR)
    # distribution for each neuron
    all_distributions = [None] * n_neurons

    mean_means = np.full(n_neurons, np.nan)
    sem_means = np.full(n_neurons, np.nan)
    FA_means = np.full(n_neurons, np.nan)

    for i in range(n_neurons):
        arr_2d = all_CR[i]  # shape (#trials, 2) => col0 = event, col1 = something else?
        if i >= len(all_FA):
            continue

        FA_vals = all_FA[i]
        windowA = arr_2d[:, 0]

        wA_vals = windowA
        FA_data = FA_vals[:, 0]
        FA_means[i] = np.mean(FA_data)
        n_avail = len(wA_vals)
        if n_avail < n_fa:
            mean_means[i] = np.mean(wA_vals)

        else:
            # Build the distribution of auROCs
            dist_means = np.zeros(n_boot, dtype=float)

            for b in range(n_boot):
                # 1) random subset of size n_fa
                # pick indices from 0..n_avail-1
                sub_idxs = random.sample(range(n_avail), k=n_fa)
                wA_sub = wA_vals[sub_idxs]
                this_mean = np.mean(wA_sub)
                dist_means[b] = this_mean

            all_distributions[i] = dist_means
            mean_means[i] = np.mean(dist_means)
            sem_means[i] = np.std(dist_means) / np.sqrt(n_boot)

    results = {
        'all_distributions': all_distributions,
        'mean': mean_means,
        'sem': sem_means,
        'FA_means': FA_means
    }
    return results


def compute_and_plot_fano_factor(
        dff_run_list_sub, freq_indices, freq_matrix,
        target_freqs, pre_trial_length, rec_hz,
        save_loc, num_freqs=2
):
    """
    For each neuron that is responsive to at least `num_freqs` frequencies:
      1) For each significant frequency, extract the average dF/F value from each trial
         in the stimulation window (from pre_trial_length + 2 sec to pre_trial_length + 5 sec).
      2) Compute the Fano factor as variance/mean of these trial-averaged responses.
      3) Plot the mean trace (averaged over trials in the stimulation window) along with a
         marker displaying the computed Fano factor.
      4) Save the plot.

    Returns a DataFrame with columns: ["neuron", "frequency", "fano_factor"]
    where fano_factor is the computed reliability measure.
    """
    # Directory for saving plots.
    plot_dir = os.path.join(save_loc, "fano_factor_validation_plots")
    os.makedirs(plot_dir, exist_ok=True)

    results = []
    n_neurons = len(dff_run_list_sub)

    # Define stimulation window indices.
    stim_start = int(pre_trial_length + (2 * rec_hz))
    stim_end = int(pre_trial_length + (5 * rec_hz))

    for i in range(n_neurons):
        # Identify significant frequencies for neuron i.
        sig_freq_indices = np.where(~np.isnan(freq_matrix[i, :]))[0]
        if len(sig_freq_indices) < num_freqs:
            continue  # Skip neurons with fewer than the required number of responses.

        for col in sig_freq_indices:
            freq = target_freqs[col]
            trial_idxs = freq_indices.get(freq, [])
            if len(trial_idxs) == 0:
                continue

            # Extract trials for the current neuron and frequency.
            selected_trials = [dff_run_list_sub[i][j] for j in trial_idxs]
            if len(selected_trials) == 0:
                continue

            # For each trial, average the dF/F values in the stimulation window.
            trial_averages = []
            for trial in selected_trials:
                # Ensure the trial is long enough.
                if len(trial) < stim_end:
                    continue
                avg_val = np.nanmean(trial[stim_start:stim_end])
                trial_averages.append(avg_val)

            # Only compute Fano factor if we have trial data.
            if len(trial_averages) == 0:
                continue

            trial_averages = np.array(trial_averages)
            mean_val = abs(np.nanmean(trial_averages))
            var_val = np.nanvar(trial_averages)
            # Avoid division by zero.
            fano_factor = var_val / mean_val if mean_val != 0 else np.nan

            results.append([i, freq, fano_factor])

            # Plot the individual trial averages and mark the mean and computed Fano factor.
            plt.figure()
            # Plot each trial's average as a dot.
            plt.plot(trial_averages, 'o', label='Trial Averages')
            # Plot the overall mean as a horizontal line.
            plt.axhline(mean_val, color='red', linestyle='--', label=f'Mean = {mean_val:.2f}')
            plt.title(f'Neuron {i}, {freq} Hz\nFano Factor = {fano_factor:.2f}')
            plt.xlabel("Trial")
            plt.ylabel("Average dF/F in Stim Window")
            plt.legend()

            out_name = f"neuron_{i}_freq_{freq}_fano_factor.png"
            plt.savefig(os.path.join(plot_dir, out_name), dpi=300)
            plt.close()

    # Return a DataFrame of the computed Fano factors.
    df_fano = pd.DataFrame(results, columns=["neuron", "frequency", "fano_factor"])
    return df_fano


###############################################################################
# Updated function to handle choice probability and additional scatter (CP vs R²)
###############################################################################
def calculate_cp_percentile_thresholds_from_averaged(avg_cp, save_loc=None, data_type='averaged_control'):
    """
    Calculate CP percentile thresholds directly from averaged CP values.
    
    Parameters:
    -----------
    avg_cp : array
        Array of averaged CP values for each neuron
    save_loc : str, optional
        Directory to save results
    data_type : str, optional
        Type of data for labeling
        
    Returns:
    --------
    dict : Dictionary containing threshold values and statistics
    """
    # Separate excitatory (CP >= 0.5) and inhibitory (CP < 0.5) neurons
    exc_mask = avg_cp >= 0.5
    inh_mask = avg_cp < 0.5
    
    cp_exc = avg_cp[exc_mask]
    cp_inh = avg_cp[inh_mask]
    
    # Calculate percentiles for excitatory neurons (upper tail)
    # For excitatory neurons, we want the top 0.5%, 1%, 2.5%, and 5%, so 99.5th, 99th, 97.5th, and 95th percentiles
    exc_99_5th = np.percentile(cp_exc, 99.5) if len(cp_exc) > 0 else np.nan
    exc_99th = np.percentile(cp_exc, 99) if len(cp_exc) > 0 else np.nan
    exc_97_5th = np.percentile(cp_exc, 97.5) if len(cp_exc) > 0 else np.nan
    exc_95th = np.percentile(cp_exc, 95) if len(cp_exc) > 0 else np.nan
    
    # Calculate percentiles for inhibitory neurons (lower tail)
    # For inhibitory neurons, we want the bottom 0.5%, 1%, 2.5%, and 5%, so 0.5th, 1st, 2.5th, and 5th percentiles
    inh_0_5th = np.percentile(cp_inh, 0.5) if len(cp_inh) > 0 else np.nan
    inh_1st = np.percentile(cp_inh, 1) if len(cp_inh) > 0 else np.nan
    inh_2_5th = np.percentile(cp_inh, 2.5) if len(cp_inh) > 0 else np.nan
    inh_5th = np.percentile(cp_inh, 5) if len(cp_inh) > 0 else np.nan
    
    # Calculate overall statistics
    overall_median = np.median(avg_cp)
    overall_mean = np.mean(avg_cp)
    overall_std = np.std(avg_cp)
    
    # Count neurons above/below thresholds
    n_exc_99_5th = np.sum(cp_exc >= exc_99_5th) if not np.isnan(exc_99_5th) else 0
    n_exc_99th = np.sum(cp_exc >= exc_99th) if not np.isnan(exc_99th) else 0
    n_exc_97_5th = np.sum(cp_exc >= exc_97_5th) if not np.isnan(exc_97_5th) else 0
    n_exc_95th = np.sum(cp_exc >= exc_95th) if not np.isnan(exc_95th) else 0
    n_inh_0_5th = np.sum(cp_inh <= inh_0_5th) if not np.isnan(inh_0_5th) else 0
    n_inh_1st = np.sum(cp_inh <= inh_1st) if not np.isnan(inh_1st) else 0
    n_inh_2_5th = np.sum(cp_inh <= inh_2_5th) if not np.isnan(inh_2_5th) else 0
    n_inh_5th = np.sum(cp_inh <= inh_5th) if not np.isnan(inh_5th) else 0
    
    # Compile results
    results = {
        'excitatory_99_5th_percentile': exc_99_5th,
        'excitatory_99th_percentile': exc_99th,
        'excitatory_97_5th_percentile': exc_97_5th,
        'excitatory_95th_percentile': exc_95th,
        'inhibitory_0_5th_percentile': inh_0_5th,
        'inhibitory_1st_percentile': inh_1st,
        'inhibitory_2_5th_percentile': inh_2_5th,
        'inhibitory_5th_percentile': inh_5th,
        'n_excitatory_neurons': len(cp_exc),
        'n_inhibitory_neurons': len(cp_inh),
        'n_total_neurons': len(avg_cp),
        'n_above_99_5th': n_exc_99_5th,
        'n_above_99th': n_exc_99th,
        'n_above_97_5th': n_exc_97_5th,
        'n_above_95th': n_exc_95th,
        'n_below_0_5th': n_inh_0_5th,
        'n_below_1st': n_inh_1st,
        'n_below_2_5th': n_inh_2_5th,
        'n_below_5th': n_inh_5th,
        'overall_median': overall_median,
        'overall_mean': overall_mean,
        'overall_std': overall_std,
        'all_cp_values': avg_cp,
        'excitatory_cp_values': cp_exc,
        'inhibitory_cp_values': cp_inh
    }
    
    # Print results with clear explanation
    print(f"\n=== CP Percentile Thresholds Analysis ({data_type}) ===")
    print(f"Total neurons analyzed: {len(avg_cp)}")
    print(f"Excitatory neurons (CP >= 0.5): {len(cp_exc)}")
    print(f"Inhibitory neurons (CP < 0.5): {len(cp_inh)}")
    print(f"\nExcitatory thresholds (upper tail - top 0.5%, 1%, 2.5%, and 5% of excitatory neurons):")
    print(f"  99.5th percentile (top 0.5%): {exc_99_5th:.4f} ({n_exc_99_5th} neurons)")
    print(f"  99th percentile (top 1%): {exc_99th:.4f} ({n_exc_99th} neurons)")
    print(f"  97.5th percentile (top 2.5%): {exc_97_5th:.4f} ({n_exc_97_5th} neurons)")
    print(f"  95th percentile (top 5%): {exc_95th:.4f} ({n_exc_95th} neurons)")
    print(f"\nInhibitory thresholds (lower tail - bottom 0.5%, 1%, 2.5%, and 5% of inhibitory neurons):")
    print(f"  0.5th percentile (bottom 0.5%): {inh_0_5th:.4f} ({n_inh_0_5th} neurons)")
    print(f"  1st percentile (bottom 1%): {inh_1st:.4f} ({n_inh_1st} neurons)")
    print(f"  2.5th percentile (bottom 2.5%): {inh_2_5th:.4f} ({n_inh_2_5th} neurons)")
    print(f"  5th percentile (bottom 5%): {inh_5th:.4f} ({n_inh_5th} neurons)")
    print(f"\nOverall statistics:")
    print(f"  Median: {overall_median:.4f}")
    print(f"  Mean: {overall_mean:.4f}")
    print(f"  Std: {overall_std:.4f}")
    
    # Save results if save_loc provided
    if save_loc is not None:
        os.makedirs(save_loc, exist_ok=True)
        
        # Save as pickle
        with open(os.path.join(save_loc, f'cp_percentile_thresholds_{data_type}.pkl'), 'wb') as f:
            pickle.dump(results, f)
        
        # Save as CSV for easy viewing
        summary_df = pd.DataFrame({
            'Metric': [
                'Excitatory 99.5th percentile (top 0.5%)',
                'Excitatory 99th percentile (top 1%)',
                'Excitatory 97.5th percentile (top 2.5%)',
                'Excitatory 95th percentile (top 5%)', 
                'Inhibitory 0.5th percentile (bottom 0.5%)',
                'Inhibitory 1st percentile (bottom 1%)',
                'Inhibitory 2.5th percentile (bottom 2.5%)',
                'Inhibitory 5th percentile (bottom 5%)',
                'Overall median',
                'Overall mean',
                'Overall std',
                'N excitatory neurons',
                'N inhibitory neurons',
                'N total neurons'
            ],
            'Value': [
                exc_99_5th, exc_99th, exc_97_5th, exc_95th, 
                inh_0_5th, inh_1st, inh_2_5th, inh_5th,
                overall_median, overall_mean, overall_std,
                len(cp_exc), len(cp_inh), len(avg_cp)
            ]
        })
        summary_df.to_csv(os.path.join(save_loc, f'cp_percentile_thresholds_{data_type}.csv'), index=False)
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # Main histogram
        plt.subplot(2, 2, 1)
        plt.hist(cp_exc, bins=30, alpha=0.7, color='red', label='Excitatory', density=True)
        plt.hist(cp_inh, bins=30, alpha=0.7, color='blue', label='Inhibitory', density=True)
        plt.axvline(0.5, color='black', linestyle='--', alpha=0.5, label='Chance (0.5)')
        
        # Add threshold lines
        if not np.isnan(exc_99_5th):
            plt.axvline(exc_99_5th, color='red', linestyle='-', linewidth=2, label=f'99.5th percentile ({exc_99_5th:.3f})')
        if not np.isnan(exc_99th):
            plt.axvline(exc_99th, color='red', linestyle=':', linewidth=2, label=f'99th percentile ({exc_99th:.3f})')
        if not np.isnan(exc_97_5th):
            plt.axvline(exc_97_5th, color='red', linestyle='-.', linewidth=2, label=f'97.5th percentile ({exc_97_5th:.3f})')
        if not np.isnan(exc_95th):
            plt.axvline(exc_95th, color='red', linestyle='--', linewidth=2, label=f'95th percentile ({exc_95th:.3f})')
        if not np.isnan(inh_0_5th):
            plt.axvline(inh_0_5th, color='blue', linestyle='-', linewidth=2, label=f'0.5th percentile ({inh_0_5th:.3f})')
        if not np.isnan(inh_1st):
            plt.axvline(inh_1st, color='blue', linestyle=':', linewidth=2, label=f'1st percentile ({inh_1st:.3f})')
        if not np.isnan(inh_2_5th):
            plt.axvline(inh_2_5th, color='blue', linestyle='-.', linewidth=2, label=f'2.5th percentile ({inh_2_5th:.3f})')
        if not np.isnan(inh_5th):
            plt.axvline(inh_5th, color='blue', linestyle='--', linewidth=2, label=f'5th percentile ({inh_5th:.3f})')
        
        plt.xlabel('Choice Probability')
        plt.ylabel('Density')
        plt.title(f'CP Distribution with Percentile Thresholds ({data_type})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Cumulative distribution
        plt.subplot(2, 2, 2)
        sorted_cp = np.sort(avg_cp)
        cumulative = np.arange(1, len(sorted_cp) + 1) / len(sorted_cp)
        plt.plot(sorted_cp, cumulative, 'k-', linewidth=2, label='All neurons')
        
        # Add threshold markers
        if not np.isnan(exc_99_5th):
            plt.axhline(0.995, color='red', linestyle='-', alpha=0.7)
            plt.axvline(exc_99_5th, color='red', linestyle='-', alpha=0.7)
        if not np.isnan(exc_99th):
            plt.axhline(0.99, color='red', linestyle=':', alpha=0.7)
            plt.axvline(exc_99th, color='red', linestyle=':', alpha=0.7)
        if not np.isnan(exc_97_5th):
            plt.axhline(0.975, color='red', linestyle='-.', alpha=0.7)
            plt.axvline(exc_97_5th, color='red', linestyle='-.', alpha=0.7)
        if not np.isnan(exc_95th):
            plt.axhline(0.95, color='red', linestyle='--', alpha=0.7)
            plt.axvline(exc_95th, color='red', linestyle='--', alpha=0.7)
        if not np.isnan(inh_0_5th):
            plt.axhline(0.005, color='blue', linestyle='-', alpha=0.7)
            plt.axvline(inh_0_5th, color='blue', linestyle='-', alpha=0.7)
        if not np.isnan(inh_1st):
            plt.axhline(0.01, color='blue', linestyle=':', alpha=0.7)
            plt.axvline(inh_1st, color='blue', linestyle=':', alpha=0.7)
        if not np.isnan(inh_2_5th):
            plt.axhline(0.025, color='blue', linestyle='-.', alpha=0.7)
            plt.axvline(inh_2_5th, color='blue', linestyle='-.', alpha=0.7)
        if not np.isnan(inh_5th):
            plt.axhline(0.05, color='blue', linestyle='--', alpha=0.7)
            plt.axvline(inh_5th, color='blue', linestyle='--', alpha=0.7)
        
        plt.xlabel('Choice Probability')
        plt.ylabel('Cumulative Probability')
        plt.title(f'Cumulative CP Distribution ({data_type})')
        plt.grid(True, alpha=0.3)
        
        # Box plot
        plt.subplot(2, 2, 3)
        data_to_plot = [cp_inh, cp_exc] if len(cp_inh) > 0 and len(cp_exc) > 0 else [avg_cp]
        labels = ['Inhibitory', 'Excitatory'] if len(cp_inh) > 0 and len(cp_exc) > 0 else ['All neurons']
        plt.boxplot(data_to_plot, labels=labels)
        plt.ylabel('Choice Probability')
        plt.title(f'CP Distribution by Type ({data_type})')
        plt.grid(True, alpha=0.3)
        
        # Summary statistics
        plt.subplot(2, 2, 4)
        plt.axis('off')
        summary_text = f"""
Summary Statistics ({data_type}):
-------------------------------
Total neurons: {len(avg_cp)}
Excitatory neurons: {len(cp_exc)}
Inhibitory neurons: {len(cp_inh)}

Excitatory thresholds:
  99.5th percentile: {exc_99_5th:.4f}
  99th percentile: {exc_99th:.4f}
  97.5th percentile: {exc_97_5th:.4f}
  95th percentile: {exc_95th:.4f}

Inhibitory thresholds:
  0.5th percentile: {inh_0_5th:.4f}
  1st percentile: {inh_1st:.4f}
  2.5th percentile: {inh_2_5th:.4f}
  5th percentile: {inh_5th:.4f}

Overall statistics:
  Median: {overall_median:.4f}
  Mean: {overall_mean:.4f}
  Std: {overall_std:.4f}
"""
        plt.text(0.1, 0.9, summary_text, transform=plt.gca().transAxes, 
               fontsize=10, verticalalignment='top',
               bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_loc, f'cp_percentile_thresholds_{data_type}.pdf'))
        plt.close()
    
    return results

def plot_neurons_by_index(
        dff_run_list_sub,
        cp_med,          # dict with 'all_distributions'
        load_loc,
        save_loc,
        rec_hz,
        pre_trial_length,
        neuron_indices,       # iterable of neuron indices to plot
        tag='selected'        # used in folder and filename
):
    """
    Plot CR vs FA mean ± SEM for specific neurons given by indices.
    Files will include the neuron index in the name.
    """
    os.makedirs(os.path.join(save_loc, "rep_neurons_cp", tag), exist_ok=True)
    out_dir = os.path.join(save_loc, "rep_neurons_cp", tag)

    # median CP per neuron for titles
    n_neurons = len(choice_prob['all_distributions'])

    # load CR and FA trial indices
    with open(os.path.join(load_loc, 'correct_reject.pkl'), 'rb') as f:
        correct_reject_trials = pickle.load(f)
    with open(os.path.join(load_loc, 'false_alarm.pkl'), 'rb') as f:
        fa_trials = pickle.load(f)
    with open(os.path.join(load_loc, 'hit.pkl'), 'rb') as f:
        hit_trials = pickle.load(f)

    hit_set = set(hit_trials)

    # For FA and CR, create "after hit" and "not after hit" lists
    fa_after_hit_idx = []
    fa_not_after_hit_idx = []
    cr_after_hit_idx = []
    cr_not_after_hit_idx = []

    for idx in fa_trials:
        if idx - 1 in hit_set:
            fa_after_hit_idx.append(idx)
        else:
            fa_not_after_hit_idx.append(idx)

    for idx in correct_reject_trials:
        if idx - 1 in hit_set:
            cr_after_hit_idx.append(idx)
        else:
            cr_not_after_hit_idx.append(idx)


    # helper
    def mean_sem_over_trials(neuron_idx, trial_inds):
        trs = [dff_run_list_sub[neuron_idx][j] for j in trial_inds if j < len(dff_run_list_sub[neuron_idx])]
        if len(trs) == 0:
            return None, None, None
        df = pd.DataFrame([pd.Series(np.asarray(t).astype(float)) for t in trs])
        m = df.mean(axis=0, skipna=True).values
        s = df.sem(axis=0, skipna=True).values
        T = len(m)
        t_sec = (np.arange(T) - pre_trial_length) / rec_hz
        return t_sec, m, s

    neuron_indices = list(neuron_indices)  # ensure iterable
    for i in neuron_indices:
        # build stats
        t_cr, m_cr, s_cr = mean_sem_over_trials(i, cr_not_after_hit_idx)
        t_fa, m_fa, s_fa = mean_sem_over_trials(i, fa_not_after_hit_idx)

        if t_cr is None and t_fa is None:
            print(f"[{tag}] neuron {i}: no CR or FA trials, skipping")
            continue

        # align time bases if both exist
        if t_cr is None:
            t = t_fa
        elif t_fa is None:
            t = t_cr
        else:
            T = min(len(t_cr), len(t_fa))
            t = t_cr[:T]
            m_cr, s_cr = m_cr[:T], s_cr[:T]
            m_fa, s_fa = m_fa[:T], s_fa[:T]

        plt.figure(figsize=(6, 3.5))
        if t_cr is not None:
            plt.plot(t, m_cr[:len(t)], label='CR', linewidth=2)
            plt.fill_between(t, (m_cr - s_cr)[:len(t)], (m_cr + s_cr)[:len(t)], alpha=0.25)
        if t_fa is not None:
            plt.plot(t, m_fa[:len(t)], linestyle='--', label='FA', linewidth=2)
            plt.fill_between(t, (m_fa - s_fa)[:len(t)], (m_fa + s_fa)[:len(t)], alpha=0.25)

        plt.axvline(0, linestyle='--', linewidth=1)
        add_axvline_opto_seconds(plt.gca(), pre_trial_length, rec_hz)

        plt.xlim(t[0], min(t[-1], 20))
        plt.xlabel("Time (s)")
        plt.ylabel("dF/F")
        plt.title(f"{tag} | neuron {i} | CP={cp_med[i]:.2f}")
        plt.legend(loc='best')
        plt.tight_layout()

        fname = os.path.join(out_dir, f"{tag}_neuron_{i}_CP_{cp_med[i]:.2f}.pdf")
        plt.savefig(fname, dpi=300)
        plt.close()
        print("Saved:", fname)

def compare_real_data_to_thresholds(real_data_path, control_thresholds, save_loc=None):
    """
    Load real data and compare it against the control thresholds.
    """
    # Load real data
    print(f"\nLoading real data from: {real_data_path}")

    # Load the choice probability data for real data
    with open(os.path.join(real_data_path, data_name), 'rb') as f:
        real_choice_prob = pickle.load(f)

    # Get median CP values for real data
    n_neurons_real = len(real_choice_prob['all_distributions'])
    real_choice_medians = np.array([
        np.nanmean(real_choice_prob['all_distributions'][i]) for i in range(n_neurons_real)
    ])

    # Remove NaN values
    valid_mask_real = ~np.isnan(real_choice_medians)
    valid_cp_real = real_choice_medians[valid_mask_real]

    # Map from "valid_cp_real" positions back to original neuron indices
    idx_valid = np.nonzero(valid_mask_real)[0]

    # Helper to safely get indices given a comparator and a control value
    def _indices(comp, control_value):
        if np.isnan(control_value):
            return np.array([], dtype=int)
        mask = comp(valid_cp_real, control_value)
        return idx_valid[mask]

    # Threshold values from the control
    exc_99_5th_control = control_thresholds['excitatory_99_5th_percentile']
    exc_99th_control   = control_thresholds['excitatory_99th_percentile']
    exc_97_5th_control = control_thresholds['excitatory_97_5th_percentile']
    exc_95th_control   = control_thresholds['excitatory_95th_percentile']
    inh_0_5th_control  = control_thresholds['inhibitory_0_5th_percentile']
    inh_1st_control    = control_thresholds['inhibitory_1st_percentile']
    inh_2_5th_control  = control_thresholds['inhibitory_2_5th_percentile']
    inh_5th_control    = control_thresholds['inhibitory_5th_percentile']

    # indices of neurons passing each threshold (in the original 0..n_neurons_real-1 space)
    exc_above_99_5th_idx = _indices(np.greater_equal, exc_99_5th_control)
    exc_above_99th_idx   = _indices(np.greater_equal, exc_99th_control)
    exc_above_97_5th_idx = _indices(np.greater_equal, exc_97_5th_control)
    exc_above_95th_idx   = _indices(np.greater_equal, exc_95th_control)

    inh_below_0_5th_idx  = _indices(np.less_equal,    inh_0_5th_control)
    inh_below_1st_idx    = _indices(np.less_equal,    inh_1st_control)
    inh_below_2_5th_idx  = _indices(np.less_equal,    inh_2_5th_control)
    inh_below_5th_idx    = _indices(np.less_equal,    inh_5th_control)

    # counts = number of neurons passing each threshold
    n_exc_above_99_5th = len(exc_above_99_5th_idx)
    n_exc_above_99th   = len(exc_above_99th_idx)
    n_exc_above_97_5th = len(exc_above_97_5th_idx)
    n_exc_above_95th   = len(exc_above_95th_idx)
    n_inh_below_0_5th  = len(inh_below_0_5th_idx)
    n_inh_below_1st    = len(inh_below_1st_idx)
    n_inh_below_2_5th  = len(inh_below_2_5th_idx)
    n_inh_below_5th    = len(inh_below_5th_idx)

    # Calculate percentages
    pct_exc_above_99_5th = (n_exc_above_99_5th / len(valid_cp_real) * 100) if len(valid_cp_real) > 0 else 0
    pct_exc_above_99th = (n_exc_above_99th / len(valid_cp_real) * 100) if len(valid_cp_real) > 0 else 0
    pct_exc_above_97_5th = (n_exc_above_97_5th / len(valid_cp_real) * 100) if len(valid_cp_real) > 0 else 0
    pct_exc_above_95th = (n_exc_above_95th / len(valid_cp_real) * 100) if len(valid_cp_real) > 0 else 0
    pct_inh_below_0_5th = (n_inh_below_0_5th / len(valid_cp_real) * 100) if len(valid_cp_real) > 0 else 0
    pct_inh_below_1st = (n_inh_below_1st / len(valid_cp_real) * 100) if len(valid_cp_real) > 0 else 0
    pct_inh_below_2_5th = (n_inh_below_2_5th / len(valid_cp_real) * 100) if len(valid_cp_real) > 0 else 0
    pct_inh_below_5th = (n_inh_below_5th / len(valid_cp_real) * 100) if len(valid_cp_real) > 0 else 0
    
    # Compile results
    comparison_results = {
        'neurons_passing_thresholds': {
            'n_exc_above_99_5th': n_exc_above_99_5th,
            'n_exc_above_99th': n_exc_above_99th,
            'n_exc_above_97_5th': n_exc_above_97_5th,
            'n_exc_above_95th': n_exc_above_95th,
            'n_inh_below_0_5th': n_inh_below_0_5th,
            'n_inh_below_1st': n_inh_below_1st,
            'n_inh_below_2_5th': n_inh_below_2_5th,
            'n_inh_below_5th': n_inh_below_5th
        },
        'percentages_passing_thresholds': {
            'pct_exc_above_99_5th': pct_exc_above_99_5th,
            'pct_exc_above_99th': pct_exc_above_99th,
            'pct_exc_above_97_5th': pct_exc_above_97_5th,
            'pct_exc_above_95th': pct_exc_above_95th,
            'pct_inh_below_0_5th': pct_inh_below_0_5th,
            'pct_inh_below_1st': pct_inh_below_1st,
            'pct_inh_below_2_5th': pct_inh_below_2_5th,
            'pct_inh_below_5th': pct_inh_below_5th
        },
        'indices_passing_thresholds': {
            'exc_above_99_5th_idx': exc_above_99_5th_idx.tolist(),
            'exc_above_99th_idx': exc_above_99th_idx.tolist(),
            'exc_above_97_5th_idx': exc_above_97_5th_idx.tolist(),
            'exc_above_95th_idx': exc_above_95th_idx.tolist(),
            'inh_below_0_5th_idx': inh_below_0_5th_idx.tolist(),
            'inh_below_1st_idx': inh_below_1st_idx.tolist(),
            'inh_below_2_5th_idx': inh_below_2_5th_idx.tolist(),
            'inh_below_5th_idx': inh_below_5th_idx.tolist()
        }
    }
    if save_loc is not None:
        os.makedirs(save_loc, exist_ok=True)
    # Print results
    print(f"\n=== Real Data vs Control Thresholds Comparison ===")
    print(f"Real data neurons passing control thresholds:")
    print(f"  Excitatory above 99.5th: {n_exc_above_99_5th} ({pct_exc_above_99_5th:.2f}%)")
    print(f"  Excitatory above 99th: {n_exc_above_99th} ({pct_exc_above_99th:.2f}%)")
    print(f"  Excitatory above 97.5th: {n_exc_above_97_5th} ({pct_exc_above_97_5th:.2f}%)")
    print(f"  Excitatory above 95th: {n_exc_above_95th} ({pct_exc_above_95th:.2f}%)")
    print(f"  Inhibitory below 0.5th: {n_inh_below_0_5th} ({pct_inh_below_0_5th:.2f}%)")
    print(f"  Inhibitory below 1st: {n_inh_below_1st} ({pct_inh_below_1st:.2f}%)")
    print(f"  Inhibitory below 2.5th: {n_inh_below_2_5th} ({pct_inh_below_2_5th:.2f}%)")
    print(f"  Inhibitory below 5th: {n_inh_below_5th} ({pct_inh_below_5th:.2f}%)")

    # add a plot to show how many cells pass the thresholds
    plt.figure(figsize=(12, 6))
    
    # Create bar plot with all thresholds
    thresholds = ['Excitatory 99.5th', 'Excitatory 99th', 'Excitatory 97.5th', 'Excitatory 95th', 
                  'Inhibitory 0.5th', 'Inhibitory 1st', 'Inhibitory 2.5th', 'Inhibitory 5th']
    percentages = [pct_exc_above_99_5th, pct_exc_above_99th, pct_exc_above_97_5th, pct_exc_above_95th,
                   pct_inh_below_0_5th, pct_inh_below_1st, pct_inh_below_2_5th, pct_inh_below_5th]
    
    colors = ['red', 'red', 'red', 'red', 'blue', 'blue', 'blue', 'blue']
    
    bars = plt.bar(thresholds, percentages, color=colors, alpha=0.7)
    plt.ylabel('Percentage of neurons')
    plt.title('Neurons passing thresholds')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(save_loc, 'neurons_passing_thresholds.png'))
    plt.show()

    # Save the full results, including indices
    with open(os.path.join(save_loc, 'comparison_results.pkl'), 'wb') as f:
        pickle.dump(comparison_results, f)

    # Optional: also write a human-readable CSV with one row per threshold
    try:
        import pandas as pd
        rows = []
        for name, idxs in comparison_results['indices_passing_thresholds'].items():
            rows.append({
                'threshold': name,
                'count': len(idxs),
                'indices': ';'.join(map(str, idxs))
            })
        pd.DataFrame(rows).to_csv(os.path.join(save_loc, 'indices_passing_thresholds.csv'), index=False)
    except Exception as e:
        print(f'Note: could not write CSV of indices: {e}')

    return comparison_results, real_choice_medians

def plot_representative_neurons_by_cp(
        dff_run_list_sub,
        choice_prob,          # dict that has 'all_distributions'
        load_loc,             # folder with correct_reject.pkl and false_alarm.pkl
        save_loc,             # where to save figures
        rec_hz,
        pre_trial_length,     # in frames
        cp_bins=None,         # list of (low, high) tuples in CP space
        n_per_bin=3,          # how many neurons to plot from each bin
        seed=0
):
    """
    For each CP bin select up to n_per_bin representative neurons, then plot
    CR vs FA mean ± SEM time courses for each neuron.
    """
    np.random.seed(seed)
    os.makedirs(os.path.join(save_loc, "rep_neurons_cp"), exist_ok=True)
    out_dir = os.path.join(save_loc, "rep_neurons_cp")

    # 1) median CP per neuron
    n_neurons = len(choice_prob['all_distributions'])
    cp_med = np.array([np.nanmedian(choice_prob['all_distributions'][i]) for i in range(n_neurons)])

    # default bins if not provided
    if cp_bins is None:
        cp_bins = [(0.0, 0.4), (0.45, 0.55), (0.6, 1.0)]  # low, around chance, high

    # 2) load CR and FA trial indices
    with open(os.path.join(load_loc, 'correct_reject.pkl'), 'rb') as f:
        correct_reject_trials = pickle.load(f)
    with open(os.path.join(load_loc, 'false_alarm.pkl'), 'rb') as f:
        fa_trials = pickle.load(f)

    # helper to build mean and sem across trials for a single neuron
    def mean_sem_over_trials(neuron_idx, trial_inds):
        trs = [dff_run_list_sub[neuron_idx][j] for j in trial_inds if j < len(dff_run_list_sub[neuron_idx])]
        if len(trs) == 0:
            return None, None, None
        # allow unequal lengths safely
        df = pd.DataFrame([pd.Series(np.asarray(t).astype(float)) for t in trs])
        m = df.mean(axis=0, skipna=True).values
        s = df.sem(axis=0, skipna=True).values
        T = len(m)
        t_sec = (np.arange(T) - pre_trial_length) / rec_hz
        return t_sec, m, s

    # 3) iterate bins and pick neurons
    for low, high in cp_bins:
        in_bin = np.where((cp_med >= low) & (cp_med <= high))[0]
        if len(in_bin) == 0:
            print(f"[rep CP] no neurons in bin {low:.2f}-{high:.2f}")
            continue

        # prefer neurons closest to bin center, then sample if needed
        center = 0.5 * (low + high)
        order = np.argsort(np.abs(cp_med[in_bin] - center))
        chosen = in_bin[order][:min(n_per_bin, len(in_bin))]

        for i in chosen:
            # build CR stats
            t_cr, m_cr, s_cr = mean_sem_over_trials(i, correct_reject_trials)
            # build FA stats
            t_fa, m_fa, s_fa = mean_sem_over_trials(i, fa_trials)

            if t_cr is None and t_fa is None:
                continue

            # align time bases if needed
            if t_cr is None:
                t = t_fa
            elif t_fa is None:
                t = t_cr
            else:
                # make same length for plotting convenience
                T = min(len(t_cr), len(t_fa))
                t = t_cr[:T]
                m_cr, s_cr = m_cr[:T], s_cr[:T]
                m_fa, s_fa = m_fa[:T], s_fa[:T]

            plt.figure(figsize=(6, 3.5))
            # CR
            if t_cr is not None:
                plt.plot(t, m_cr[:len(t)], label='CR', linewidth=2)
                plt.fill_between(t, (m_cr - s_cr)[:len(t)], (m_cr + s_cr)[:len(t)], alpha=0.25)
            # FA
            if t_fa is not None:
                plt.plot(t, m_fa[:len(t)], linestyle='--', label='FA', linewidth=2)
                plt.fill_between(t, (m_fa - s_fa)[:len(t)], (m_fa + s_fa)[:len(t)], alpha=0.25)

            # reference lines and event markers
            plt.axvline(0, linestyle='--', linewidth=1)
            add_axvline_opto_seconds(plt.gca(), pre_trial_length, rec_hz)

            plt.xlim(t[0], min(t[-1], 20))
            plt.xlabel("Time (s)")
            plt.ylabel("dF/F")
            plt.title(f"Neuron {i}  CP={cp_med[i]:.2f}  bin {low:.2f}-{high:.2f}")
            plt.legend(loc='best')
            plt.tight_layout()

            fname = os.path.join(out_dir, f"neuron_{i}_CP_{cp_med[i]:.2f}_bin_{low:.2f}-{high:.2f}.png")
            plt.savefig(fname, dpi=300)
            plt.close()
            print("Saved representative CP plot:", fname)

def baseline_subtract_neurons(norm_data, pre_trial_length, baseline_time):
    import copy
    concat_data = copy.deepcopy(norm_data)
    j = 0
    for trial in concat_data:
        i = 0
        for neuron in trial:
            # Calculate baseline activity
            neuron = np.asarray(neuron)
            baseline = np.mean(neuron[pre_trial_length - baseline_time:pre_trial_length])

            # Subtract baseline from all data points
            sub_neuron = neuron - baseline
            concat_data[j][i] = sub_neuron
            i += 1
        j += 1

    return concat_data
###############################################################################
# Main routine
###############################################################################
if __name__ == '__main__':
    ##########################################################################
    # Settings
    ##########################################################################

    day = ['sess06']
    mouse = 'mouse09'
    base_load_loc = DATA_ROOT + r'\{}\{}\ML\data\processed_data\auROC_freq_history_control_CR-CR'.format(mouse, day[0])
    base_load_loc2 = DATA_ROOT + r'\{}\{}\ML\data\processed_data\auROC_freq_history_control_CR-CR'.format(mouse, day[0])
    gaussian_sigmoid_path = DATA_ROOT + r'\{}\{}\ML\data\processed_data\auroc_neurometric_zscore_no_sub_v4'.format(mouse, day[0])
    save_loc = DATA_ROOT + r'\{}\{}\ML\data\processed_data\auROC_freq_history_control_CR-CR_rep1\test_all_reps_v4'.format(mouse, day[0])
    # base_load_loc = DATA_ROOT + r'\{}\{}\ML\data\processed_data\auROC_z_no_sub_history_CR_control'.format(mouse, day[0])
    # base_load_loc2 = DATA_ROOT + r'\{}\{}\ML\data\processed_data\auROC_z_no_sub_hist ory_CR_control'.format(mouse, day[0])
    # save_loc = DATA_ROOT + r'\{}\{}\ML\data\processed_data\auROC_z_no_sub_history_CR_control_rep1\test_all_reps_v2'.format(mouse, day[0])
    # Path to real data 
    real_data_path = DATA_ROOT + r'\{}\{}\ML\data\processed_data\auROC_freq_history'.format(mouse, day[0])
    data_name = 'res_cr_subsample_global.pkl'#'res_cr_subsample_global.pkl'#'res_cr_subsample_v2.pkl'
    os.makedirs(save_loc, exist_ok=True)
    rec_hz = 15
    pre_trial_length = 4 * rec_hz

    # Define repetitions to analyze
    repetitions = ['rep1', 'rep2', 'rep3', 'rep4', 'rep5'] #, 'rep2', 'rep3', 'rep4', 'rep5'
    with open(join(real_data_path, 'dff_run_list.pkl'), 'rb') as f:
        dff_run_list = pickle.load(f)
    # baseline subtract dff for plotting
    dff_run_list_sub = baseline_subtract_neurons(dff_run_list, pre_trial_length, baseline_time=4 * rec_hz)
    # Load and analyze all repetitions
    ##########################################################################
    
    class NumpyCoreUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            # Patch any module path that starts with 'numpy._core'
            if module.startswith("numpy._core"):
                module = module.replace("numpy._core", "numpy.core")
            return super().find_class(module, name)

    # Dictionary to store results from all repetitions
    all_repetition_results = {}
    
    for rep in repetitions:
        print(f"\n{'='*60}")
        print(f"ANALYZING REPETITION: {rep}")
        print(f"{'='*60}")
        
        # Construct paths for this repetition
        load_loc = f"{base_load_loc}_{rep}"
        load_loc2 = f"{base_load_loc2}_{rep}"
        
        # Check if this repetition exists
        if not os.path.exists(load_loc):
            print(f"Warning: {load_loc} does not exist. Skipping {rep}.")
            continue
            
        # Load choice probability data for this repetition
        try:
            with open(os.path.join(load_loc, data_name), 'rb') as f:
                choice_prob = NumpyCoreUnpickler(f).load()
        except FileNotFoundError:
            print(f"Warning: {data_name} not found in {load_loc}. Skipping {rep}.")
            continue

        # Calculate CP values for this repetition
        sig_neuron_indices = np.arange(len(choice_prob['all_distributions']))
        CP_all = []
        for i in sig_neuron_indices:
            CP_all.append(np.nanmean(choice_prob['all_distributions'][i]))
        
        # Store results for this repetition
        all_repetition_results[rep] = {
            'CP_all': CP_all,
            'n_neurons': len(sig_neuron_indices)
        }
        
        print(f"Repetition {rep}: {len(sig_neuron_indices)} neurons")

    ##########################################################################
    # K-S test and averaging across repetitions
    ##########################################################################
    
    if len(all_repetition_results) > 1:
        print(f"\n{'='*60}")
        print("CP DISTRIBUTION ANALYSIS AND AVERAGING")
        print(f"{'='*60}")
        
        # Get all repetitions with CP data
        reps_with_cp_data = [rep for rep, results in all_repetition_results.items() 
                            if 'CP_all' in results]
        
        if len(reps_with_cp_data) > 1:
            # 1. Test for differences in CP distributions (Kolmogorov-Smirnov test)
            print("\n1. Testing CP distribution differences (Kolmogorov-Smirnov test):")
            cp_distributions = [all_repetition_results[rep]['CP_all'] for rep in reps_with_cp_data]
            
            ks_results = []
            for i in range(len(cp_distributions)):
                for j in range(i+1, len(cp_distributions)):
                    stat, p_val = scipy.stats.ks_2samp(cp_distributions[i], cp_distributions[j])
                    ks_results.append((reps_with_cp_data[i], reps_with_cp_data[j], stat, p_val))
                    print(f"   {reps_with_cp_data[i]} vs {reps_with_cp_data[j]}: D={stat:.3f}, p={p_val:.3f}")
            
            # Check if any comparisons are significant
            alpha = 0.05
            significant_differences = [p < alpha for _, _, _, p in ks_results]
            any_significant = any(significant_differences)
            
            print(f"\nSignificant differences found: {any_significant}")
            if any_significant:
                print("   Some CP distributions are significantly different, but proceeding with averaging anyway.")
            
            ##########################################################################
            # Average CP distributions across repetitions
            ##########################################################################
            
            print(f"\n2. Averaging CP distributions across {len(reps_with_cp_data)} repetitions:")
            
            # Get the number of neurons (should be consistent across reps)
            n_neurons = all_repetition_results[reps_with_cp_data[0]]['n_neurons']
            
            # Collect CP data from each repetition
            cp_data = []
            for rep in reps_with_cp_data:
                cp_data.append(all_repetition_results[rep]['CP_all'])
            
            # Convert to numpy array for easier manipulation
            cp_data = np.array(cp_data)  # shape: (n_reps, n_neurons)
            
            # Debug: Check the data structure
            print(f"   Data shape: {cp_data.shape}")
            print(f"   First few CP values from each repetition:")
            for i, rep in enumerate(reps_with_cp_data):
                print(f"     {rep}: {cp_data[i, :5]}")  # First 5 neurons
            
            # Calculate mean CP across repetitions
            avg_cp = np.mean(cp_data, axis=0)
            se_cp = np.std(cp_data, axis=0) / np.sqrt(len(reps_with_cp_data))
            
            print(f"   Averaged {n_neurons} neurons across {len(reps_with_cp_data)} repetitions")
            print(f"   Mean CP: {np.mean(avg_cp):.3f} ± {np.std(avg_cp):.3f}")
            print(f"   First few averaged CP values: {avg_cp[:5]}")
            
            # Debug: Compare individual repetition statistics with averaged
            print(f"\n   Individual repetition statistics:")
            for rep in reps_with_cp_data:
                rep_cp = all_repetition_results[rep]['CP_all']
                print(f"     {rep}: mean={np.mean(rep_cp):.3f}, std={np.std(rep_cp):.3f}, min={np.min(rep_cp):.3f}, max={np.max(rep_cp):.3f}")
            
            print(f"   Averaged: mean={np.mean(avg_cp):.3f}, std={np.std(avg_cp):.3f}, min={np.min(avg_cp):.3f}, max={np.max(avg_cp):.3f}")
            
            # Debug: Check if neurons are in the same order across repetitions
            print(f"\n   Checking neuron order consistency:")
            for i in range(min(5, n_neurons)):  # Check first 5 neurons
                neuron_values = cp_data[:, i]
                print(f"     Neuron {i}: {neuron_values}")
            
            # Create a debug plot to see the distributions
            plt.figure(figsize=(15, 10))
            
            # Plot 1: Individual repetition distributions
            plt.subplot(2, 3, 1)
            for i, rep in enumerate(reps_with_cp_data):
                plt.hist(all_repetition_results[rep]['CP_all'], bins=20, alpha=0.3, 
                        label=f'{rep}', density=True)
            plt.xlabel('Choice Probability')
            plt.ylabel('Density')
            plt.title('Individual Repetition Distributions')
            plt.legend()
            plt.axvline(0.5, color='black', linestyle='--', alpha=0.5)
            
            # Plot 2: Averaged distribution
            plt.subplot(2, 3, 2)
            plt.hist(avg_cp, bins=20, alpha=0.7, color='red', density=True, label='Averaged')
            plt.xlabel('Choice Probability')
            plt.ylabel('Density')
            plt.title('Averaged Distribution')
            plt.legend()
            plt.axvline(0.5, color='black', linestyle='--', alpha=0.5)
            
            # Plot 3: Overlay comparison
            plt.subplot(2, 3, 3)
            for i, rep in enumerate(reps_with_cp_data):
                plt.hist(all_repetition_results[rep]['CP_all'], bins=20, alpha=0.2, 
                        density=True)
            plt.hist(avg_cp, bins=20, alpha=0.8, color='red', linewidth=2, 
                    histtype='step', label='Averaged', density=True)
            plt.xlabel('Choice Probability')
            plt.ylabel('Density')
            plt.title('Overlay Comparison')
            plt.legend()
            plt.axvline(0.5, color='black', linestyle='--', alpha=0.5)
            
            # Plot 4: Scatter plot of individual vs averaged
            plt.subplot(2, 3, 4)
            for i, rep in enumerate(reps_with_cp_data):
                plt.scatter(all_repetition_results[rep]['CP_all'], avg_cp, alpha=0.3, s=10)
            plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
            plt.xlabel('Individual Repetition CP')
            plt.ylabel('Averaged CP')
            plt.title('Individual vs Averaged CP')
            plt.grid(True, alpha=0.3)
            
            # Plot 5: Distribution of differences
            plt.subplot(2, 3, 5)
            differences = []
            for rep in reps_with_cp_data:
                diff = np.array(all_repetition_results[rep]['CP_all']) - avg_cp
                differences.extend(diff)
            plt.hist(differences, bins=30, alpha=0.7, color='green')
            plt.xlabel('Difference (Individual - Averaged)')
            plt.ylabel('Count')
            plt.title('Distribution of Differences')
            plt.axvline(0, color='black', linestyle='--', alpha=0.5)
            plt.grid(True, alpha=0.3)
            
            # Plot 6: Box plot of CP values by repetition
            plt.subplot(2, 3, 6)
            cp_by_rep = [all_repetition_results[rep]['CP_all'] for rep in reps_with_cp_data]
            plt.boxplot(cp_by_rep, labels=reps_with_cp_data)
            plt.ylabel('Choice Probability')
            plt.title('CP Distribution by Repetition')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(join(save_loc, 'averaging_debug.pdf'), dpi=300)
            plt.savefig(join(save_loc, 'averaging_debug.png'), dpi=300)
            plt.show()
            
            # Check if there's a systematic bias
            print(f"\n   Systematic bias Analysis:")
            for rep in reps_with_cp_data:
                rep_cp = np.array(all_repetition_results[rep]['CP_all'])
                bias = np.mean(rep_cp - avg_cp)
                print(f"     {rep} bias: {bias:.4f}")
            
            # Check if the issue is with neuron ordering
            print(f"\n   Checking if neurons are properly aligned:")
            # Calculate correlation between repetitions
            for i in range(len(reps_with_cp_data)):
                for j in range(i+1, len(reps_with_cp_data)):
                    rep1_cp = np.array(all_repetition_results[reps_with_cp_data[i]]['CP_all'])
                    rep2_cp = np.array(all_repetition_results[reps_with_cp_data[j]]['CP_all'])
                    corr = np.corrcoef(rep1_cp, rep2_cp)[0, 1]
                    print(f"     Correlation {reps_with_cp_data[i]} vs {reps_with_cp_data[j]}: {corr:.3f}")
            
            ##########################################################################
            # Create thresholds from averaged CP distribution
            ##########################################################################
            
            print(f"\n3. Creating thresholds from averaged CP distribution:")
            
            # Debug: Compare individual repetition thresholds vs averaged thresholds
            print("\nDEBUG: Comparing individual vs averaged thresholds:")
            
            # Calculate thresholds for each individual repetition
            individual_thresholds = {}
            for rep in reps_with_cp_data:
                rep_cp = all_repetition_results[rep]['CP_all']
                
                # Separate excitatory and inhibitory
                
                # Calculate percentiles for excitatory neurons (upper tail)
                # For excitatory neurons, we want the top 0.5%, 1%, 2.5%, and 5%, so 99.5th, 99th, 97.5th, and 95th percentiles
                exc_99_5th = np.percentile(rep_cp, 99.5) if len(rep_cp) > 0 else np.nan
                exc_99th = np.percentile(rep_cp, 99) if len(rep_cp) > 0 else np.nan
                exc_97_5th = np.percentile(rep_cp, 97.5) if len(rep_cp) > 0 else np.nan
                exc_95th = np.percentile(rep_cp, 95) if len(rep_cp) > 0 else np.nan
                
                # Calculate percentiles for inhibitory neurons (lower tail)
                # For inhibitory neurons, we want the bottom 0.5%, 1%, 2.5%, and 5%, so 0.5th, 1st, 2.5th, and 5th percentiles
                inh_0_5th = np.percentile(rep_cp, 0.5) if len(rep_cp) > 0 else np.nan
                inh_1st = np.percentile(rep_cp, 1) if len(rep_cp) > 0 else np.nan
                inh_2_5th = np.percentile(rep_cp, 2.5) if len(rep_cp) > 0 else np.nan
                inh_5th = np.percentile(rep_cp, 5) if len(rep_cp) > 0 else np.nan
                
                individual_thresholds[rep] = {
                    'excitatory_99_5th': exc_99_5th,
                    'excitatory_99th': exc_99th,
                    'excitatory_97_5th': exc_97_5th,
                    'excitatory_95th': exc_95th,
                    'inhibitory_0_5th': inh_0_5th,
                    'inhibitory_1st': inh_1st,
                    'inhibitory_2_5th': inh_2_5th,
                    'inhibitory_5th': inh_5th,
                    'n_excitatory': len(rep_cp),
                    'n_inhibitory': len(rep_cp)
                }
                
                print(f"  {rep}: 99.5th={exc_99_5th:.3f}, 99th={exc_99th:.3f}, 97.5th={exc_97_5th:.3f}, 95th={exc_95th:.3f}, 0.5th={inh_0_5th:.3f}, 1st={inh_1st:.3f}, 2.5th={inh_2_5th:.3f}, 5th={inh_5th:.3f}")
            
            # Calculate averaged thresholds
            # cp_thresholds = calculate_cp_percentile_thresholds_from_averaged(
            #     avg_cp,
            #     save_loc=save_loc,
            #     data_type='averaged_control'
            # )
            
            # print(f"\nAveraged thresholds:")
            # print(f"  99th={cp_thresholds['excitatory_99th_percentile']:.3f}, 95th={cp_thresholds['excitatory_95th_percentile']:.3f}")
            # print(f"  1st={cp_thresholds['inhibitory_1st_percentile']:.3f}, 5th={cp_thresholds['inhibitory_5th_percentile']:.3f}")
                        # Average the thresholds (not the CP values)
            avg_exc_99_5th = np.mean([individual_thresholds[rep]['excitatory_99_5th'] for rep in reps_with_cp_data])
            avg_exc_99th = np.mean([individual_thresholds[rep]['excitatory_99th'] for rep in reps_with_cp_data])
            avg_exc_97_5th = np.mean([individual_thresholds[rep]['excitatory_97_5th'] for rep in reps_with_cp_data])
            avg_exc_95th = np.mean([individual_thresholds[rep]['excitatory_95th'] for rep in reps_with_cp_data])
            avg_inh_0_5th = np.mean([individual_thresholds[rep]['inhibitory_0_5th'] for rep in reps_with_cp_data])
            avg_inh_1st = np.mean([individual_thresholds[rep]['inhibitory_1st'] for rep in reps_with_cp_data])
            avg_inh_2_5th = np.mean([individual_thresholds[rep]['inhibitory_2_5th'] for rep in reps_with_cp_data])
            avg_inh_5th = np.mean([individual_thresholds[rep]['inhibitory_5th'] for rep in reps_with_cp_data])
            
            # Calculate standard errors of the thresholds
            se_exc_99_5th = np.std([individual_thresholds[rep]['excitatory_99_5th'] for rep in reps_with_cp_data]) / np.sqrt(len(reps_with_cp_data))
            se_exc_99th = np.std([individual_thresholds[rep]['excitatory_99th'] for rep in reps_with_cp_data]) / np.sqrt(len(reps_with_cp_data))
            se_exc_97_5th = np.std([individual_thresholds[rep]['excitatory_97_5th'] for rep in reps_with_cp_data]) / np.sqrt(len(reps_with_cp_data))
            se_exc_95th = np.std([individual_thresholds[rep]['excitatory_95th'] for rep in reps_with_cp_data]) / np.sqrt(len(reps_with_cp_data))
            se_inh_0_5th = np.std([individual_thresholds[rep]['inhibitory_0_5th'] for rep in reps_with_cp_data]) / np.sqrt(len(reps_with_cp_data))
            se_inh_1st = np.std([individual_thresholds[rep]['inhibitory_1st'] for rep in reps_with_cp_data]) / np.sqrt(len(reps_with_cp_data))
            se_inh_2_5th = np.std([individual_thresholds[rep]['inhibitory_2_5th'] for rep in reps_with_cp_data]) / np.sqrt(len(reps_with_cp_data))
            se_inh_5th = np.std([individual_thresholds[rep]['inhibitory_5th'] for rep in reps_with_cp_data]) / np.sqrt(len(reps_with_cp_data))
            
            # Create averaged thresholds dictionary
            cp_thresholds = {
                'excitatory_99_5th_percentile': avg_exc_99_5th,
                'excitatory_99th_percentile': avg_exc_99th,
                'excitatory_97_5th_percentile': avg_exc_97_5th,
                'excitatory_95th_percentile': avg_exc_95th,
                'inhibitory_0_5th_percentile': avg_inh_0_5th,
                'inhibitory_1st_percentile': avg_inh_1st,
                'inhibitory_2_5th_percentile': avg_inh_2_5th,
                'inhibitory_5th_percentile': avg_inh_5th,
                'excitatory_99_5th_se': se_exc_99_5th,
                'excitatory_99th_se': se_exc_99th,
                'excitatory_97_5th_se': se_exc_97_5th,
                'excitatory_95th_se': se_exc_95th,
                'inhibitory_0_5th_se': se_inh_0_5th,
                'inhibitory_1st_se': se_inh_1st,
                'inhibitory_2_5th_se': se_inh_2_5th,
                'inhibitory_5th_se': se_inh_5th,
                'individual_thresholds': individual_thresholds,
                'n_repetitions': len(reps_with_cp_data)
            }
            
            print(f"\nAveraged thresholds (from individual repetition thresholds):")
            print(f"  99.5th={avg_exc_99_5th:.3f} ± {se_exc_99_5th:.3f}")
            print(f"  99th={avg_exc_99th:.3f} ± {se_exc_99th:.3f}")
            print(f"  97.5th={avg_exc_97_5th:.3f} ± {se_exc_97_5th:.3f}")
            print(f"  95th={avg_exc_95th:.3f} ± {se_exc_95th:.3f}")
            print(f"  0.5th={avg_inh_0_5th:.3f} ± {se_inh_0_5th:.3f}")
            print(f"  1st={avg_inh_1st:.3f} ± {se_inh_1st:.3f}")
            print(f"  2.5th={avg_inh_2_5th:.3f} ± {se_inh_2_5th:.3f}")
            print(f"  5th={avg_inh_5th:.3f} ± {se_inh_5th:.3f}")
            
            # Compare with mean of individual thresholds
            mean_99_5th = np.mean([individual_thresholds[rep]['excitatory_99_5th'] for rep in reps_with_cp_data])
            mean_99th = np.mean([individual_thresholds[rep]['excitatory_99th'] for rep in reps_with_cp_data])
            mean_97_5th = np.mean([individual_thresholds[rep]['excitatory_97_5th'] for rep in reps_with_cp_data])
            mean_95th = np.mean([individual_thresholds[rep]['excitatory_95th'] for rep in reps_with_cp_data])
            mean_0_5th = np.mean([individual_thresholds[rep]['inhibitory_0_5th'] for rep in reps_with_cp_data])
            mean_1st = np.mean([individual_thresholds[rep]['inhibitory_1st'] for rep in reps_with_cp_data])
            mean_2_5th = np.mean([individual_thresholds[rep]['inhibitory_2_5th'] for rep in reps_with_cp_data])
            mean_5th = np.mean([individual_thresholds[rep]['inhibitory_5th'] for rep in reps_with_cp_data])
            
            print(f"\nMean of individual thresholds:")
            print(f"  99.5th={mean_99_5th:.3f}, 99th={mean_99th:.3f}, 97.5th={mean_97_5th:.3f}, 95th={mean_95th:.3f}")
            print(f"  0.5th={mean_0_5th:.3f}, 1st={mean_1st:.3f}, 2.5th={mean_2_5th:.3f}, 5th={mean_5th:.3f}")
            
            print(f"\nDifference (averaged - mean_individual):")
            print(f"  99.5th: {cp_thresholds['excitatory_99_5th_percentile'] - mean_99_5th:.3f}")
            print(f"  99th: {cp_thresholds['excitatory_99th_percentile'] - mean_99th:.3f}")
            print(f"  97.5th: {cp_thresholds['excitatory_97_5th_percentile'] - mean_97_5th:.3f}")
            print(f"  95th: {cp_thresholds['excitatory_95th_percentile'] - mean_95th:.3f}")
            print(f"  0.5th: {cp_thresholds['inhibitory_0_5th_percentile'] - mean_0_5th:.3f}")
            print(f"  1st: {cp_thresholds['inhibitory_1st_percentile'] - mean_1st:.3f}")
            print(f"  2.5th: {cp_thresholds['inhibitory_2_5th_percentile'] - mean_2_5th:.3f}")
            print(f"  5th: {cp_thresholds['inhibitory_5th_percentile'] - mean_5th:.3f}")
            
            # Create comparison plot
            plt.figure(figsize=(15, 10))
            
            # Plot 1: Individual repetition thresholds
            plt.subplot(2, 3, 1)
            reps_list = list(individual_thresholds.keys())
            thresholds_99th = [individual_thresholds[rep]['excitatory_99th'] for rep in reps_list]
            thresholds_95th = [individual_thresholds[rep]['excitatory_95th'] for rep in reps_list]
            thresholds_1st = [individual_thresholds[rep]['inhibitory_1st'] for rep in reps_list]
            thresholds_5th = [individual_thresholds[rep]['inhibitory_5th'] for rep in reps_list]
            
            x_pos = np.arange(len(reps_list))
            width = 0.2
            
            plt.bar(x_pos - width*1.5, thresholds_99th, width, label='99th percentile', alpha=0.7)
            plt.bar(x_pos - width*0.5, thresholds_95th, width, label='95th percentile', alpha=0.7)
            plt.bar(x_pos + width*0.5, thresholds_1st, width, label='1st percentile', alpha=0.7)
            plt.bar(x_pos + width*1.5, thresholds_5th, width, label='5th percentile', alpha=0.7)
            
            # Add averaged thresholds as horizontal lines
            plt.axhline(cp_thresholds['excitatory_99th_percentile'], color='red', linestyle='--', alpha=0.7, label='Averaged 99th')
            plt.axhline(cp_thresholds['excitatory_95th_percentile'], color='orange', linestyle='--', alpha=0.7, label='Averaged 95th')
            plt.axhline(cp_thresholds['inhibitory_1st_percentile'], color='blue', linestyle='--', alpha=0.7, label='Averaged 1st')
            plt.axhline(cp_thresholds['inhibitory_5th_percentile'], color='green', linestyle='--', alpha=0.7, label='Averaged 5th')
            
            plt.xlabel('Repetition')
            plt.ylabel('Threshold Value')
            plt.title('Individual vs Averaged Thresholds')
            plt.xticks(x_pos, reps_list)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Plot 2: CP distributions comparison
            plt.subplot(2, 3, 2)
            for i, rep in enumerate(reps_with_cp_data):
                plt.hist(all_repetition_results[rep]['CP_all'], bins=20, alpha=0.3, 
                        label=f'{rep}', density=True)
            plt.hist(avg_cp, bins=20, alpha=0.7, color='black', linewidth=2, 
                    histtype='step', label='Averaged', density=True)
            plt.xlabel('Choice Probability')
            plt.ylabel('Density')
            plt.title('CP Distributions')
            plt.legend()
            plt.axvline(0.5, color='black', linestyle='--', alpha=0.5)
            
            # Plot 3: Distribution of differences
            plt.subplot(2, 3, 3)
            # Calculate how many neurons would pass thresholds in each repetition
            n_passing_99th = []
            n_passing_95th = []
            n_passing_1st = []
            n_passing_5th = []
            
            for rep in reps_with_cp_data:
                rep_cp = np.array(all_repetition_results[rep]['CP_all'])

                n_passing_99th.append(np.sum(rep_cp >= cp_thresholds['excitatory_99th_percentile']))
                n_passing_95th.append(np.sum(rep_cp >= cp_thresholds['excitatory_95th_percentile']))
                n_passing_1st.append(np.sum(rep_cp <= cp_thresholds['inhibitory_1st_percentile']))
                n_passing_5th.append(np.sum(rep_cp <= cp_thresholds['inhibitory_5th_percentile']))
            
            plt.bar(['99th', '95th', '1st', '5th'], 
                   [np.mean(n_passing_99th), np.mean(n_passing_95th), 
                    np.mean(n_passing_1st), np.mean(n_passing_5th)],
                   yerr=[np.std(n_passing_99th), np.std(n_passing_95th),
                        np.std(n_passing_1st), np.std(n_passing_5th)],
                   capsize=5, alpha=0.7)
            plt.ylabel('Mean number of neurons passing threshold')
            plt.title('Neurons passing averaged thresholds\n(across individual repetitions)')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(join(save_loc, 'threshold_comparison_debug.pdf'), dpi=300)
            plt.savefig(join(save_loc, 'threshold_comparison_debug.png'), dpi=300)
            plt.show()
            
            print(f"\nNeurons passing averaged thresholds in each repetition:")
            for i, rep in enumerate(reps_with_cp_data):
                print(f"  {rep}: 99th={n_passing_99th[i]}, 95th={n_passing_95th[i]}, 1st={n_passing_1st[i]}, 5th={n_passing_5th[i]}")
            
            print(f"\nMean across repetitions: 99th={np.mean(n_passing_99th):.1f}, 95th={np.mean(n_passing_95th):.1f}, 1st={np.mean(n_passing_1st):.1f}, 5th={np.mean(n_passing_5th):.1f}")
            
            ##########################################################################
            # Test real data against averaged thresholds
            ##########################################################################
            
            print(f"\n4. Testing real data against averaged thresholds:")
            
            # Check if real data path exists
            if os.path.exists(real_data_path):
                print(f"   Real data path found: {real_data_path}")
                
                # Compare real data to averaged control thresholds
                comparison_results, cp_values_real_data = compare_real_data_to_thresholds(
                    real_data_path=real_data_path,
                    control_thresholds=cp_thresholds,
                    save_loc=save_loc
                )

                # load and take the cp values for sigmodial neurons and gaussian neurons
                gaussian_neuron_indices = np.load(join(gaussian_sigmoid_path, 'gaussian_neuron_indices.npy'))
                sigmoid_neuron_indices = np.load(join(gaussian_sigmoid_path, 'sigmoid_neuron_indices.npy'))
                cp_gaussian_neurons = cp_values_real_data[gaussian_neuron_indices]
                cp_sigmoid_neurons = cp_values_real_data[sigmoid_neuron_indices]
                # plot hist of cp values for gaussian and sigmoid neurons
                plt.figure(figsize=(10, 5))
                plt.hist(cp_gaussian_neurons, bins=20, alpha=0.7, label='Gaussian Neurons', color='blue', density=True)
                plt.hist(cp_sigmoid_neurons, bins=20, alpha=0.7, label='Sigmoid Neurons', color='orange', density=True)
                plt.xlabel('Choice Probability')
                plt.ylabel('Density')
                plt.title('CP Distribution for Gaussian vs Sigmoid Neurons')
                plt.legend()
                plt.axvline(0.5, color='black', linestyle='--', alpha=0.5)
                plt.savefig(join(save_loc, 'cp_gaussian_vs_sigmoid.pdf'), dpi=300)
                plt.savefig(join(save_loc, 'cp_gaussian_vs_sigmoid.png'), dpi=300)
                plt.show()


                # Print summary
                print(f"\n{'='*60}")
                print("SUMMARY OF REAL DATA COMPARISON")
                print(f"{'='*60}")
                print(f"Real data neurons passing averaged control thresholds:")
                print(f"  Excitatory above 99th percentile: {comparison_results['neurons_passing_thresholds']['n_exc_above_99_5th']} "
                      f"({comparison_results['percentages_passing_thresholds']['pct_exc_above_99_5th']:.2f}%)")
                print(f"  Excitatory above 99th: {comparison_results['neurons_passing_thresholds']['n_exc_above_99th']} "
                      f"({comparison_results['percentages_passing_thresholds']['pct_exc_above_99th']:.2f}%)")
                print(f"  Excitatory above 97.5th: {comparison_results['neurons_passing_thresholds']['n_exc_above_97_5th']} "
                      f"({comparison_results['percentages_passing_thresholds']['pct_exc_above_97_5th']:.2f}%)")
                print(f"  Excitatory above 95th: {comparison_results['neurons_passing_thresholds']['n_exc_above_95th']} "
                      f"({comparison_results['percentages_passing_thresholds']['pct_exc_above_95th']:.2f}%)")
                print(f"  Inhibitory below 0.5th: {comparison_results['neurons_passing_thresholds']['n_inh_below_0_5th']} "
                      f"({comparison_results['percentages_passing_thresholds']['pct_inh_below_0_5th']:.2f}%)")
                print(f"  Inhibitory below 1st: {comparison_results['neurons_passing_thresholds']['n_inh_below_1st']} "
                      f"({comparison_results['percentages_passing_thresholds']['pct_inh_below_1st']:.2f}%)")
                print(f"  Inhibitory below 2.5th: {comparison_results['neurons_passing_thresholds']['n_inh_below_2_5th']} "
                      f"({comparison_results['percentages_passing_thresholds']['pct_inh_below_2_5th']:.2f}%)")
                print(f"  Inhibitory below 5th: {comparison_results['neurons_passing_thresholds']['n_inh_below_5th']} "
                      f"({comparison_results['percentages_passing_thresholds']['pct_inh_below_5th']:.2f}%)")

                exc_idx = comparison_results['indices_passing_thresholds']['exc_above_97_5th_idx']
                inh_idx = comparison_results['indices_passing_thresholds']['inh_below_2_5th_idx']
                not_idx = [6, 8, 10, 79]

                # plot_neurons_by_index(
                #     dff_run_list_sub, cp_values_real_data, real_data_path, save_loc, rec_hz, pre_trial_length,
                #     neuron_indices=exc_idx, tag='exc_97_5_pass'
                # )
                #
                # plot_neurons_by_index(
                #     dff_run_list_sub, cp_values_real_data, real_data_path, save_loc, rec_hz, pre_trial_length,
                #     neuron_indices=inh_idx, tag='inh_2_5_pass'
                # )

                plot_neurons_by_index(
                    dff_run_list_sub, cp_values_real_data, real_data_path, save_loc, rec_hz, pre_trial_length,
                    neuron_indices=not_idx, tag='not_pass')
                ##########################################################################
                # Create comparison plots
                ##########################################################################

                # Plot 1: CP distributions comparison
                plt.figure(figsize=(15, 5))

                # Subplot 1: Individual repetition distributions
                plt.subplot(1, 3, 1)
                colors = plt.cm.Set1(np.linspace(0, 1, len(reps_with_cp_data)))

                for i, rep in enumerate(reps_with_cp_data):
                    plt.hist(all_repetition_results[rep]['CP_all'], bins=20, alpha=0.6,
                            label=f'{rep} (n={len(all_repetition_results[rep]["CP_all"])})',
                            color=colors[i], density=True)

                plt.xlabel('Choice Probability')
                plt.ylabel('Density')
                plt.title('CP Distributions by Repetition')
                plt.legend()
                plt.axvline(0.5, color='black', linestyle='--', alpha=0.5)

                # Subplot 2: Averaged distribution
                plt.subplot(1, 3, 2)
                plt.hist(avg_cp, bins=20, alpha=0.7, color='blue', density=True,
                        label=f'Averaged (n={len(avg_cp)})')
                plt.xlabel('Choice Probability')
                plt.ylabel('Density')
                plt.title('Averaged CP Distribution')
                plt.legend()
                plt.axvline(0.5, color='black', linestyle='--', alpha=0.5)

                # Add threshold lines
                plt.axvline(cp_thresholds['excitatory_99th_percentile'], color='red', linestyle=':', alpha=0.7, label='99th percentile')
                plt.axvline(cp_thresholds['excitatory_95th_percentile'], color='orange', linestyle=':', alpha=0.7, label='95th percentile')
                plt.axvline(cp_thresholds['inhibitory_5th_percentile'], color='green', linestyle=':', alpha=0.7, label='5th percentile')
                plt.axvline(cp_thresholds['inhibitory_1st_percentile'], color='purple', linestyle=':', alpha=0.7, label='1st percentile')
                plt.legend()

                # Subplot 3: Real data vs thresholds
                plt.subplot(1, 3, 3)

                # Load and plot real data
                try:
                    with open(os.path.join(real_data_path, data_name), 'rb') as f:
                        real_choice_prob = NumpyCoreUnpickler(f).load()

                    real_cp = []
                    for i in range(len(real_choice_prob['all_distributions'])):
                        real_cp.append(np.nanmean(real_choice_prob['all_distributions'][i]))


                    real_cp = np.array(real_cp)
                    cp_sym = 2 * (real_cp - 0.5)  # CP′ range −1…1

                    # split into excitation (CP′ ≥ 0) vs inhibition (CP′ < 0)
                    cp_exc = cp_sym[cp_sym >= 0]
                    cp_inh = cp_sym[cp_sym < 0]
                    bin_width = 0.02
                    bins = np.arange(-1.0, 1.0 + bin_width, bin_width)

                    # add the shuffled data
                    shuffled_cp = []
                    for i in range(len(real_choice_prob['all_shuffled_distributions'])):
                        shuffled_cp.append(np.nanmean(real_choice_prob['all_shuffled_distributions'][i]))
                    # convert to CP′ scale
                    shuffled_cp = np.array(shuffled_cp)
                    shuffled_cp = 2 * (shuffled_cp - 0.5)

                    # Calculate histograms first to get max values for normalization
                    hist_exc, bin_edges = np.histogram(cp_exc, bins=bins, density=False)
                    hist_inh, _ = np.histogram(cp_inh, bins=bins, density=False)
                    hist_shuffled, _ = np.histogram(shuffled_cp, bins=bins, density=False)

                    # Normalize each histogram to have max height of 1
                    hist_exc_norm = hist_exc / np.max(hist_exc) if np.max(hist_exc) > 0 else hist_exc
                    hist_inh_norm = hist_inh / np.max(hist_inh) if np.max(hist_inh) > 0 else hist_inh
                    hist_shuffled_norm = hist_shuffled / np.max(hist_shuffled) if np.max(hist_shuffled) > 0 else hist_shuffled

                    # Plot normalized histograms using bar plots for better control
                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                    bin_width = bin_edges[1] - bin_edges[0]

                    plt.bar(bin_centers, hist_exc_norm, width=bin_width*0.8, alpha=0.7, color='salmon',
                           edgecolor='black', label=f'Real data ≥ 0.5 (n={len(cp_exc)})')
                    plt.bar(bin_centers, hist_inh_norm, width=bin_width*0.8, alpha=0.7, color='lightblue',
                           edgecolor='black', label=f'Real data < 0.5 (n={len(cp_inh)})')
                    plt.bar(bin_centers, hist_shuffled_norm, width=bin_width*0.8, alpha=0.7, color='gray',
                           edgecolor='black', label=f'Shuffled data (n={len(shuffled_cp)})')

                    # overall stats
                    overall_median = np.nanmedian(cp_sym)
                    overall_mean = np.nanmean(cp_sym)
                    plt.axvline(overall_median, color='black', linestyle='--', label='Median', linewidth=2)
                    plt.axvline(overall_mean, color='gray', linestyle='--', label='Mean', linewidth=2)

                    # convert the thresholds to CP′ scale
                    cp_thresholds_cp_sym = {
                        'excitatory_99_5th_percentile': 2 * (cp_thresholds['excitatory_99_5th_percentile'] - 0.5),
                        'excitatory_99th_percentile': 2 * (cp_thresholds['excitatory_99th_percentile'] - 0.5),
                        'excitatory_97_5th_percentile': 2 * (cp_thresholds['excitatory_97_5th_percentile'] - 0.5),
                        'excitatory_95th_percentile': 2 * (cp_thresholds['excitatory_95th_percentile'] - 0.5),
                        'inhibitory_0_5th_percentile': 2 * (cp_thresholds['inhibitory_0_5th_percentile'] - 0.5),
                        'inhibitory_1st_percentile': 2 * (cp_thresholds['inhibitory_1st_percentile'] - 0.5),
                        'inhibitory_2_5th_percentile': 2 * (cp_thresholds['inhibitory_2_5th_percentile'] - 0.5),
                        'inhibitory_5th_percentile': 2 * (cp_thresholds['inhibitory_5th_percentile'] - 0.5)
                    }
                    # Add threshold lines
                    plt.axvline(cp_thresholds_cp_sym['excitatory_99_5th_percentile'], color='green', linestyle=':', alpha=0.7, label='99.5th percentile', linewidth=2)
                    plt.axvline(cp_thresholds_cp_sym['excitatory_99th_percentile'], color='purple', linestyle=':', alpha=0.7, label='99th percentile', linewidth=2)
                    plt.axvline(cp_thresholds_cp_sym['excitatory_97_5th_percentile'], color='red', linestyle=':', alpha=0.7, label='97.5th percentile', linewidth=2)
                    plt.axvline(cp_thresholds_cp_sym['excitatory_95th_percentile'], color='blue', linestyle=':', alpha=0.7, label='95th percentile', linewidth=2)
                    plt.axvline(cp_thresholds_cp_sym['inhibitory_0_5th_percentile'], color='green', linestyle=':', alpha=0.7, label='0.5th percentile', linewidth=2)
                    plt.axvline(cp_thresholds_cp_sym['inhibitory_1st_percentile'], color='purple', linestyle=':', alpha=0.7, label='1st percentile', linewidth=2)
                    plt.axvline(cp_thresholds_cp_sym['inhibitory_2_5th_percentile'], color='red', linestyle=':', alpha=0.7, label='2.5th percentile', linewidth=2)
                    plt.axvline(cp_thresholds_cp_sym['inhibitory_5th_percentile'], color='blue', linestyle=':', alpha=0.7, label='5th percentile', linewidth=2)

                    plt.xlabel('Choice Probability')
                    plt.ylabel('Normalized Density')
                    plt.title('Real Data vs Averaged Thresholds')
                    plt.legend()
                    plt.axvline(0.5, color='black', linestyle='--', alpha=0.5, linewidth=2)

                except FileNotFoundError:
                    plt.text(0.5, 0.5, 'Real data not found', ha='center', va='center', transform=plt.gca().transAxes)
                    plt.title('Real Data vs Averaged Thresholds')

                plt.tight_layout()
                plt.savefig(join(save_loc, 'CP_distribution_comparison_with_thresholds.pdf'), dpi=300)
                plt.savefig(join(save_loc, 'CP_distribution_comparison_with_thresholds.png'), dpi=300)
                plt.show()

                # Save averaged results and thresholds
                averaged_results = {
                    'avg_CP': avg_cp,
                    'se_CP': se_cp,
                    'n_neurons': n_neurons,
                    'n_repetitions': len(reps_with_cp_data),
                    'repetitions_used': reps_with_cp_data,
                    'cp_thresholds': cp_thresholds,
                    'ks_test_results': ks_results,
                    'any_significant_differences': any_significant,
                    'individual_rep_data': cp_data
                }

                with open(join(save_loc, 'averaged_cp_results.pkl'), 'wb') as f:
                    pickle.dump(averaged_results, f)

            else:
                print(f"\nReal data path not found: {real_data_path}")
                print("Skipping real data comparison. Please update the real_data_path variable.")

                # Still save the averaged results
                averaged_results = {
                    'avg_CP': avg_cp,
                    'se_CP': se_cp,
                    'n_neurons': n_neurons,
                    'n_repetitions': len(reps_with_cp_data),
                    'repetitions_used': reps_with_cp_data,
                    'cp_thresholds': cp_thresholds,
                    'ks_test_results': ks_results,
                    'any_significant_differences': any_significant,
                    'individual_rep_data': cp_data
                }

                with open(join(save_loc, 'averaged_cp_results.pkl'), 'wb') as f:
                    pickle.dump(averaged_results, f)

        else:
            print("Not enough repetitions with CP data for Analysis.")

    else:
        print("Only one repetition found. Cannot perform CP distribution Analysis.")


    ##########################################################################
    # Single-repetition analysis (rep1)
    ##########################################################################

    print(f"\n{'='*60}")
    print("RUNNING ORIGINAL ANALYSIS FOR REP1 (BACKUP)")
    print(f"{'='*60}")

    # Use rep1
    load_loc = f"{base_load_loc}_rep1"
    load_loc2 = f"{base_load_loc2}_rep1"

    with open(os.path.join(load_loc, data_name), 'rb') as f:
        choice_prob = NumpyCoreUnpickler(f).load()

    # 1) Filter to only neurons significant for >= 1 frequency
    sig_neuron_indices = np.arange(len(choice_prob['all_distributions']))

    bin_percents = choice_probability_hist(choice_prob, save_loc, 'control_CR-CR',
                                           sig_neuron=sig_neuron_indices)


    # compare the CP to the mean response
    with open(join(load_loc2, 'cr_light_averages.pkl'), 'rb') as f:
        cr_event_averages = pickle.load(f)
    with open(join(load_loc2, 'fa_light_averages.pkl'), 'rb') as f:
        fa_event_averages = pickle.load(f)

    N_FA = len(fa_event_averages[0])
    all_results = compute_opto_windows_mean_activity(cr_event_averages, fa_event_averages, N_FA, 1000)
    FA_average = np.asarray(all_results['FA_means'][sig_neuron_indices])
    CR_average = np.asarray(all_results['mean'][sig_neuron_indices])
    CR_sem = np.asarray(all_results['sem'][sig_neuron_indices])

    # choice probability of significant neurons
    CP_all = []
    for i in sig_neuron_indices:
        CP_all.append(np.nanmean(choice_prob['all_distributions'][i]))

    r, p = pearsonr(CP_all, CR_average)
    # save r and p values
    with open(join(save_loc, 'CP_vs_CR_r.pkl'), 'wb') as f:
        pickle.dump(r, f)
    # Make a scatter plot of the mean response vs the CP with SEM error bars
    plt.figure()
    plt.errorbar(CP_all, CR_average, yerr=CR_sem, fmt='o', color='gray', alpha=0.5, ecolor='black', capsize=3)
    plt.text(0.1, max(CR_average) * 0.9, f'r = {r:.2f}, p = {p:.3f}', fontsize=12,
             bbox=dict(facecolor='white', alpha=0.5))
    plt.xlabel('Choice Probability')
    plt.ylabel('Mean response')
    plt.title('Correct Rejection')
    # Add dashed reference lines
    plt.axhline(0, color='black', linestyle='--')
    plt.axvline(0.5, color='black', linestyle='--')
    plt.tight_layout()
    plt.savefig(join(save_loc, 'CP_vs_CR_response.pdf'))
    plt.close()

    r_FA, p_FA = pearsonr(CP_all, FA_average)
    # save r and p values
    with open(join(save_loc, 'CP_vs_FA_r.pkl'), 'wb') as f:
        pickle.dump(r_FA, f)
    plt.figure()
    plt.scatter(CP_all, np.asarray(FA_average), color='gray', alpha=0.5)
    plt.xlabel('Choice Probability')
    plt.ylabel('Mean response')
    plt.title('False Alarm')
    plt.axhline(0, color='black', linestyle='--')
    plt.axvline(0.5, color='black', linestyle='--')
    plt.tight_layout()
    plt.savefig(join(save_loc, 'CP_vs_FA_response.pdf'))












