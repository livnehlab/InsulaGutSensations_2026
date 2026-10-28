"""
Fractions of light-, reward- and both-responsive neurons in the detection vs. reversal
sessions (cells registered across days with CellReg): mean across mice (bars) and
individual mice (lines). Inputs: <mouse>/cellreg_detection_reversal/figures_without_both/fractions.pkl.
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
matplotlib.rcParams['pdf.fonttype'] = 42


# ---------------------------------------------------------------------------
# Data locations (set these before running)
# ---------------------------------------------------------------------------
DATA_ROOT    = r'path\to\2p_data'  # folder with the imaging sessions (<mouse>/<session>/ML/...)
RESULTS_ROOT = r'path\to\results'  # folder where the across-mice results are saved

# 1) configure
data_root = DATA_ROOT
save_dir = RESULTS_ROOT + r'\cellreg_detection_reversal\figures_without_both'
os.makedirs(save_dir, exist_ok=True)
mouse_ids = ['mouse09', 'mouse10', 'mouse16', 'mouse14']
orig_labels = ['Detection1', 'Reversal']
plot_labels = ['Detection1', 'Reversal']
idx_map     = [orig_labels.index(l) for l in plot_labels]

cats  = ['light','reward','both','none']
modes = ['all_cells'] #'all_cells',

# 2) load each mouse's fractions.pkl
fracs_by_mouse = {}
for m in mouse_ids:
    frac_file = os.path.join(
        data_root, m,
        'cellreg_detection_reversal',
        'figures_without_both', 'fractions.pkl'
    )
    with open(frac_file, 'rb') as f:
        fracs_by_mouse[m] = pickle.load(f)

# 3) compute across-mouse means (in plot_labels order)
mean_fracs = {mode: {} for mode in modes}
for mode in modes:
    for c in cats:
        # collect matrix of shape (n_mice, n_days_original)
        mat = np.vstack([fracs_by_mouse[m][mode][c] for m in mouse_ids])
        # reorder columns to plot_labels
        mat = mat[:, idx_map]
        mean_fracs[mode][c] = mat.mean(axis=0)

# 4) plot
x = np.arange(len(plot_labels))

for mode in modes:
    for c in cats:
        fig, ax = plt.subplots()

        # first draw the mean bars
        ax.bar(x, mean_fracs[mode][c],
               width=0.6,
               alpha=0.4,
               edgecolor='k')

        # then overlay each mouse's line+marker
        for m in mouse_ids:
            y_old = fracs_by_mouse[m][mode][c]
            y     = [y_old[i] for i in idx_map]
            ax.plot(x, y, '--o', label=m)

        ax.set_xticks(x)
        ax.set_xticklabels(plot_labels)
        ax.set_ylim(0, 1)
        ax.set_ylabel('Fraction of cells')
        ax.set_title(f'{c.capitalize()} - {mode.replace("_", " ")}')
        ax.legend(loc='best', fontsize='small')
        plt.tight_layout()

        # save and show
        plt.savefig(os.path.join(save_dir, f'{mode}_{c}_group.png'))
        plt.show()
        plt.close(fig)
