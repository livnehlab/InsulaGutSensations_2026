"""
Detection day: across-mice average of the running classifier (SVM_classifier_only15Hz.py /
NB_classifier_only15Hz.py). Loads [mean, SEM] of one comparison from each session, and saves
all_accuracy_results_<groups>.pkl (per-mouse curves) and combined_accuracy_results_<groups>.pkl
(mean, SEM across mice; SEM = std (ddof=0) / sqrt(n mice)) with a plot, in save_loc.

Run once per comparison, with groups and the per-session file set accordingly:
  CR_FA               <- classifier_CR_FA_not_after_combined.pkl
  FA_HIT              <- classifier_FA_HIT_combined.pkl
  CR_HIT_not_balanced <- classifier_CR_HIT_combined_not_balanced.pkl
"""
import os
import math
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Qt5Agg')
matplotlib.rcParams['pdf.fonttype'] = 42

# ---------------------------------------------------------------------------
# Data locations (set these before running)
# ---------------------------------------------------------------------------
DATA_ROOT    = r'path\to\2p_data'  # folder with the imaging sessions (<mouse>/<session>/ML/...)
RESULTS_ROOT = r'path\to\results'  # folder where the across-mice results are saved

# ----- User-defined parameters -----
rec_hz           = 15       # Recording frequency (frames per second)
pre_trial_length = 4 * rec_hz
save_loc         = RESULTS_ROOT + r'\SVM_classifier_all_new'
os.makedirs(save_loc, exist_ok=True)

def plot_combined_accuracy(accuracy, sem, file_name):
    """
    Plots a single 'combined' accuracy trace with SEM shading.
    """
    # xlim is how far in frames to plot
    xlim = 271
    TICKS_LABELS = np.arange(0, xlim, step=rec_hz * 2) / rec_hz

    plt.figure(figsize=(20, 10))

    # Plot the mean accuracy
    plt.plot(accuracy, color='black', label='Combined Accuracy')

    # Plot the SEM shading
    lower = accuracy - sem
    upper = accuracy + sem
    plt.fill_between(range(len(accuracy)), lower, upper, alpha=0.2, color='gray')

    # ----- Add vertical lines (event markers) -----
    # Distractor light
    plt.axvline(x=pre_trial_length,                color='darkorange', linestyle='--', linewidth=1, label='Distractor on')
    plt.axvline(x=pre_trial_length + 9 * rec_hz,   color='darkorange', linestyle='--', linewidth=1)

    # Opto
    plt.axvline(x=pre_trial_length + 1 * rec_hz,               color='yellow', linestyle='--', linewidth=1, label='Opto on')
    plt.axvline(x=pre_trial_length + 1 * rec_hz + 4 * rec_hz,   color='yellow', linestyle='--', linewidth=1)

    # Odor
    plt.axvline(x=pre_trial_length + 5 * rec_hz,               color='blue', linestyle='--', linewidth=1, label='Odor on')
    plt.axvline(x=pre_trial_length + 5 * rec_hz + 2 * rec_hz,   color='blue', linestyle='--', linewidth=1)

    # Training window shading
    plt.axvspan(pre_trial_length + 2 * rec_hz,
                pre_trial_length + 5 * rec_hz,
                color='gray', alpha=0.2)

    # ----- Aesthetics -----
    plt.xticks(np.arange(0, xlim, step=rec_hz * 2), TICKS_LABELS)
    plt.yticks(np.arange(0, 1.1, step=0.1), np.arange(0, 110, step=10))
    plt.xlim([0, xlim])
    plt.ylim([0, 1])  # Accuracy from 0 to 1
    plt.xlabel('Time (sec)')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    # ----- Save the figure -----
    plt.savefig(os.path.join(save_loc, file_name))
    plt.savefig(os.path.join(save_loc, file_name.replace('.png', '.pdf')))
    plt.close()


if __name__ == "__main__":

    # Detection sessions (mouse, day)
    # mice    = ['mouse01']
    # days    = ['sess17']
    mice    = ['mouse01', 'mouse05','mouse09', 'mouse14', 'mouse10']
    days    = ['sess01','sess03','sess05', 'sess11', 'sess07']
    all_acc = []
    groups = 'CR_FA'

    for mouse, day in zip(mice, days):
        load_loc = DATA_ROOT + fr'\{mouse}\{day}\ML\data\classifier_leading_z_no_sub_history_v2'
        # [mean, SEM] of the combined accuracy of this session; keep the mean
        with open(os.path.join(load_loc, 'classifier_CR_FA_not_after_combined.pkl'), 'rb') as f:
            combined_data = pickle.load(f)  # [mean, SEM], each of shape [time_points]
            all_acc.append(combined_data[0])

    # Convert to array: shape -> [num_mice x time_points]
    all_acc = np.array(all_acc)
    # save the all accuracy results
    all_results_path = os.path.join(save_loc, f'all_accuracy_results_{groups}.pkl')
    with open(all_results_path, 'wb') as f:
        pickle.dump(all_acc, f)

    # Compute mean & SEM across mice
    mean_acc = np.mean(all_acc, axis=0)
    sem_acc  = np.std(all_acc, axis=0) / math.sqrt(len(all_acc))

    # save the combined accuracy results
    combined_results_path = os.path.join(save_loc, f'combined_accuracy_results_{groups}.pkl')
    with open(combined_results_path, 'wb') as f:
        pickle.dump((mean_acc, sem_acc), f)


    # Plot the results
    plot_combined_accuracy(mean_acc, sem_acc, f'combined_accuracy_{groups}.pdf')

    print("Done. The combined accuracy plot has been saved.")
