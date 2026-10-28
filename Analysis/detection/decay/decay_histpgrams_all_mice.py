"""
Across-mice summary of the opto-response decay.

Loads the per-neuron decay summary tables saved for each mouse (fluorescence and
deconvolved signals, see decay_opto.py) and plots:
  - per-mouse averages of the median fast decay time constant (tau_fast) and of the
    median 2/3-drop time,
  - pooled-across-mice histograms of the same measures,
  - the across-mice odor summary (% of the peak left at odor start and % decrease from the peak).
"""

import os
import pickle
from os.path import join, exists
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# ---------------------------------------------------------------------------
# Data locations (set these before running)
# ---------------------------------------------------------------------------
DATA_ROOT    = r'path\to\2p_data'  # folder with the imaging sessions (<mouse>/<session>/ML/...)
RESULTS_ROOT = r'path\to\results'  # folder where the across-mice results are saved

def load_table(path):
    """
    Load a saved per-neuron summary table from .csv or .pkl
    """
    if path.endswith('.csv'):
        return pd.read_csv(path)
    elif path.endswith('.pkl'):
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        if isinstance(obj, pd.DataFrame):
            return obj
        raise ValueError(f'PKL at {path} does not contain a DataFrame.')
    else:
        raise ValueError(f'Unsupported file type: {path}')


def load_mouse_tables(mouse_dirs,
                      fluor_filename,
                      deconv_filename):
    """
    mouse_dirs: dict
        mouse_name -> folder containing the saved summary files

    Returns:
        fluor_mouse_dfs, deconv_mouse_dfs
    """
    fluor_mouse_dfs = {}
    deconv_mouse_dfs = {}

    for mouse_name, mouse_dir in mouse_dirs.items():
        fluor_path = join(mouse_dir, fluor_filename)
        deconv_path = join(mouse_dir, deconv_filename)

        if not exists(fluor_path):
            raise FileNotFoundError(f'Fluorescence file not found for {mouse_name}: {fluor_path}')
        if not exists(deconv_path):
            raise FileNotFoundError(f'Deconvolved file not found for {mouse_name}: {deconv_path}')

        fluor_df = load_table(fluor_path)
        deconv_df = load_table(deconv_path)

        fluor_df = fluor_df.copy()
        deconv_df = deconv_df.copy()

        fluor_df['mouse'] = mouse_name
        deconv_df['mouse'] = mouse_name

        fluor_mouse_dfs[mouse_name] = fluor_df
        deconv_mouse_dfs[mouse_name] = deconv_df

    return fluor_mouse_dfs, deconv_mouse_dfs


def pool_across_mice(mouse_dfs):
    """Concatenates the per-mouse tables (empty tables are skipped)."""
    dfs = [df for df in mouse_dfs.values() if df is not None and len(df) > 0]
    if len(dfs) == 0:
        return pd.DataFrame()
    return pd.concat(dfs, axis=0, ignore_index=True)


def plot_pooled_histograms_from_saved(
        fluor_all,
        deconv_all,
        save_dir,
        prefix='pooled_across_mice_summary',
        tau_col='median_tau_fast_s',
        drop_col='median_t_2third_drop_s',
        fluor_label='Fluorescence',
        deconv_label='Deconvolved',
        bins=20
):
    """
    Histograms of the per-neuron median tau_fast and median 2/3-drop time, pooled across
    mice (the fluorescence histograms are commented out; only deconvolved is plotted).
    The pooled tables are also saved as .csv and .pkl.
    """
    os.makedirs(save_dir, exist_ok=True)

    if tau_col not in fluor_all.columns:
        raise ValueError(f'Fluorescence table missing tau column: {tau_col}')
    if tau_col not in deconv_all.columns:
        raise ValueError(f'Deconvolved table missing tau column: {tau_col}')
    if drop_col not in fluor_all.columns:
        raise ValueError(f'Fluorescence table missing drop column: {drop_col}')
    if drop_col not in deconv_all.columns:
        raise ValueError(f'Deconvolved table missing drop column: {drop_col}')

    fluor_tau = pd.to_numeric(fluor_all[tau_col], errors='coerce').to_numpy()
    deconv_tau = pd.to_numeric(deconv_all[tau_col], errors='coerce').to_numpy()

    fluor_drop = pd.to_numeric(fluor_all[drop_col], errors='coerce').to_numpy()
    deconv_drop = pd.to_numeric(deconv_all[drop_col], errors='coerce').to_numpy()

    fluor_tau = fluor_tau[np.isfinite(fluor_tau)]
    deconv_tau = deconv_tau[np.isfinite(deconv_tau)]
    fluor_drop = fluor_drop[np.isfinite(fluor_drop)]
    deconv_drop = deconv_drop[np.isfinite(deconv_drop)]

    all_tau = np.concatenate([fluor_tau, deconv_tau]) if (len(fluor_tau) + len(deconv_tau)) > 0 else np.array([])
    all_drop = np.concatenate([fluor_drop, deconv_drop]) if (len(fluor_drop) + len(deconv_drop)) > 0 else np.array([])

    tau_bins = np.histogram_bin_edges(all_tau, bins=bins) if len(all_tau) > 0 else bins
    drop_bins = np.histogram_bin_edges(all_drop, bins=bins) if len(all_drop) > 0 else bins

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # axes[0].hist(
    #     fluor_tau, bins=tau_bins, alpha=0.5, edgecolor='black',
    #     label=f'{fluor_label} (n={len(fluor_tau)})'
    # )
    axes[0].hist(
        deconv_tau, bins=tau_bins, alpha=0.5, edgecolor='black',
        label=f'{deconv_label} (n={len(deconv_tau)})'
    )
    axes[0].set_title('Pooled across mice: median tau_fast per neuron')
    axes[0].set_xlabel('Median tau_fast (s)')
    axes[0].set_ylabel('Neuron count')
    axes[0].legend()

    # axes[1].hist(
    #     fluor_drop, bins=drop_bins, alpha=0.5, edgecolor='black',
    #     label=f'{fluor_label} (n={len(fluor_drop)})'
    # )
    axes[1].hist(
        deconv_drop, bins=drop_bins, alpha=0.5, edgecolor='black',
        label=f'{deconv_label} (n={len(deconv_drop)})'
    )
    axes[1].set_title('Pooled across mice: median 2/3-drop time per neuron')
    axes[1].set_xlabel('Median 2/3-drop time (s)')
    axes[1].set_ylabel('Neuron count')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(join(save_dir, f'{prefix}.png'))
    plt.savefig(join(save_dir, f'{prefix}.pdf'))
    plt.close(fig)

    fluor_all.to_csv(join(save_dir, f'{prefix}_fluor_pooled.csv'), index=False)
    deconv_all.to_csv(join(save_dir, f'{prefix}_deconv_pooled.csv'), index=False)

    with open(join(save_dir, f'{prefix}_fluor_pooled.pkl'), 'wb') as f:
        pickle.dump(fluor_all, f)

    with open(join(save_dir, f'{prefix}_deconv_pooled.pkl'), 'wb') as f:
        pickle.dump(deconv_all, f)

def plot_average_per_mouse(
        fluor_mouse_dfs,
        deconv_mouse_dfs,
        save_dir,
        prefix='average_per_mouse',
        tau_col='median_tau_fast_s',
        drop_col='median_t_2third_drop_s',
        fluor_label='Fluorescence',
        deconv_label='Deconvolved',
        agg='mean',
        error_type='sem',
        jitter=0.04,
        connect_pairs=True
):
    """
    For each mouse:
        first reduce neurons -> one value per mouse (mean or median, see `agg`)
    Then plot:
        - one bar for fluorescence
        - one bar for deconvolved
        - error bars across mice (SEM or SD, see `error_type`)
        - individual mice as dots
        - optional paired line connecting each mouse across modalities
    """
    import os
    import pickle
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from os.path import join

    os.makedirs(save_dir, exist_ok=True)

    if agg not in ['mean', 'median']:
        raise ValueError("agg must be 'mean' or 'median'")
    if error_type not in ['sem', 'sd']:
        raise ValueError("error_type must be 'sem' or 'sd'")

    common_mice = sorted(set(fluor_mouse_dfs.keys()) & set(deconv_mouse_dfs.keys()))
    rows = []

    for mouse in common_mice:
        fluor_df = fluor_mouse_dfs[mouse]
        deconv_df = deconv_mouse_dfs[mouse]

        if fluor_df is None or len(fluor_df) == 0:
            continue
        if deconv_df is None or len(deconv_df) == 0:
            continue

        if tau_col not in fluor_df.columns or tau_col not in deconv_df.columns:
            raise ValueError(f"Missing tau column '{tau_col}' for mouse {mouse}")
        if drop_col not in fluor_df.columns or drop_col not in deconv_df.columns:
            raise ValueError(f"Missing drop column '{drop_col}' for mouse {mouse}")

        fluor_tau = pd.to_numeric(fluor_df[tau_col], errors='coerce').to_numpy(dtype=float)
        deconv_tau = pd.to_numeric(deconv_df[tau_col], errors='coerce').to_numpy(dtype=float)
        fluor_drop = pd.to_numeric(fluor_df[drop_col], errors='coerce').to_numpy(dtype=float)
        deconv_drop = pd.to_numeric(deconv_df[drop_col], errors='coerce').to_numpy(dtype=float)

        fluor_tau = fluor_tau[np.isfinite(fluor_tau)]
        deconv_tau = deconv_tau[np.isfinite(deconv_tau)]
        fluor_drop = fluor_drop[np.isfinite(fluor_drop)]
        deconv_drop = deconv_drop[np.isfinite(deconv_drop)]

        if len(fluor_tau) == 0 or len(deconv_tau) == 0 or len(fluor_drop) == 0 or len(deconv_drop) == 0:
            continue

        if agg == 'mean':
            fluor_tau_val = np.nanmean(fluor_tau)
            deconv_tau_val = np.nanmean(deconv_tau)
            fluor_drop_val = np.nanmean(fluor_drop)
            deconv_drop_val = np.nanmean(deconv_drop)
        else:
            fluor_tau_val = np.nanmedian(fluor_tau)
            deconv_tau_val = np.nanmedian(deconv_tau)
            fluor_drop_val = np.nanmedian(fluor_drop)
            deconv_drop_val = np.nanmedian(deconv_drop)

        rows.append({
            'mouse': mouse,
            'fluor_tau': fluor_tau_val,
            'deconv_tau': deconv_tau_val,
            'fluor_drop': fluor_drop_val,
            'deconv_drop': deconv_drop_val,
            'n_fluor_neurons': len(fluor_tau),
            'n_deconv_neurons': len(deconv_tau),
        })

    mouse_summary_df = pd.DataFrame(rows)
    mouse_summary_df.to_csv(join(save_dir, f'{prefix}_table.csv'), index=False)

    with open(join(save_dir, f'{prefix}_table.pkl'), 'wb') as f:
        pickle.dump(mouse_summary_df, f)

    if len(mouse_summary_df) == 0:
        print('No mouse-level data to plot.')
        return mouse_summary_df

    def mean_and_err(x):
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        if len(x) == 0:
            return np.nan, np.nan
        mu = np.mean(x)
        if len(x) == 1:
            err = 0.0
        elif error_type == 'sd':
            err = np.std(x, ddof=1)
        else:
            err = np.std(x, ddof=1) / np.sqrt(len(x))
        return mu, err

    fluor_tau_vals = mouse_summary_df['fluor_tau'].to_numpy(dtype=float)
    deconv_tau_vals = mouse_summary_df['deconv_tau'].to_numpy(dtype=float)
    fluor_drop_vals = mouse_summary_df['fluor_drop'].to_numpy(dtype=float)
    deconv_drop_vals = mouse_summary_df['deconv_drop'].to_numpy(dtype=float)

    tau_means, tau_errs = zip(*(mean_and_err(arr) for arr in [fluor_tau_vals, deconv_tau_vals]))
    drop_means, drop_errs = zip(*(mean_and_err(arr) for arr in [fluor_drop_vals, deconv_drop_vals]))

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # fixed x positions
    x_tau = np.array([0, 1], dtype=float)
    x_drop = np.array([0, 1], dtype=float)

    # deterministic small offsets for each mouse, same offset used for both modalities
    n_mice = len(mouse_summary_df)
    if n_mice == 1:
        offsets = np.array([0.0])
    else:
        offsets = np.linspace(-jitter, jitter, n_mice)

    fluor_tau_x = x_tau[0] + offsets
    deconv_tau_x = x_tau[1] + offsets
    fluor_drop_x = x_drop[0] + offsets
    deconv_drop_x = x_drop[1] + offsets

    # tau panel
    axes[0].bar(x_tau, tau_means, yerr=tau_errs, capsize=5, width=0.6)
    if connect_pairs:
        for i in range(n_mice):
            axes[0].plot(
                [fluor_tau_x[i], deconv_tau_x[i]],
                [fluor_tau_vals[i], deconv_tau_vals[i]],
                '-',
                linewidth=1,
                alpha=0.7,
                color='gray'
            )
    # add mouse labels next to points
    for i, mouse in enumerate(mouse_summary_df['mouse']):
        axes[0].text(fluor_tau_x[i], fluor_tau_vals[i], mouse, fontsize=8, ha='right', va='bottom')
        axes[0].text(deconv_tau_x[i], deconv_tau_vals[i], mouse, fontsize=8, ha='left', va='bottom')
    axes[0].plot(fluor_tau_x, fluor_tau_vals, 'o')
    axes[0].plot(deconv_tau_x, deconv_tau_vals, 'o')
    axes[0].set_xticks(x_tau)
    axes[0].set_xticklabels([fluor_label, deconv_label])
    axes[0].set_ylabel('Tau_fast (s)')
    axes[0].set_title(f'{agg.capitalize()} across neurons, then mean across mice')

    # 2/3 panel
    axes[1].bar(x_drop, drop_means, yerr=drop_errs, capsize=5, width=0.6)
    if connect_pairs:
        for i in range(n_mice):
            axes[1].plot(
                [fluor_drop_x[i], deconv_drop_x[i]],
                [fluor_drop_vals[i], deconv_drop_vals[i]],
                '-',
                linewidth=1,
                alpha=0.7,
                color='gray'
            )
    # add mouse labels next to points
    for i, mouse in enumerate(mouse_summary_df['mouse']):
        axes[1].text(fluor_drop_x[i], fluor_drop_vals[i], mouse, fontsize=8, ha='right', va='bottom')
        axes[1].text(deconv_drop_x[i], deconv_drop_vals[i], mouse, fontsize=8, ha='left', va='bottom')
    axes[1].plot(fluor_drop_x, fluor_drop_vals, 'o')
    axes[1].plot(deconv_drop_x, deconv_drop_vals, 'o')
    axes[1].set_xticks(x_drop)
    axes[1].set_xticklabels([fluor_label, deconv_label])
    axes[1].set_ylabel('2/3-drop time (s)')
    axes[1].set_title(f'{agg.capitalize()} across neurons, then mean across mice')

    plt.tight_layout()
    plt.savefig(join(save_dir, f'{prefix}.png'))
    plt.savefig(join(save_dir, f'{prefix}.pdf'))
    plt.close(fig)

    return mouse_summary_df
def plot_odor_conclusion_across_mice(
        fluor_mouse_dfs,
        deconv_mouse_dfs,
        save_dir,
        prefix='odor_conclusion_across_mice',
        percent_col='median_odor_start_to_peak_percent',
        decrease_col='median_odor_start_decrease_percent',
        fluor_label='Fluorescence',
        deconv_label='Deconvolved',
        agg='mean',
        error_type='sem',
        jitter=0.04,
        connect_pairs=True
):
    """
    Final across-mice odor summary figure.

    For each mouse:
        collapse neurons -> one value per mouse
    Then plot:
        - two bars: fluorescence, deconvolved
        - error bars across mice
        - mouse dots
        - optional paired lines

    Panels:
        1) % of peak left at odor start
        2) % decrease from peak by odor start
    """
    import os
    import pickle
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from os.path import join

    os.makedirs(save_dir, exist_ok=True)

    if agg not in ['mean', 'median']:
        raise ValueError("agg must be 'mean' or 'median'")
    if error_type not in ['sem', 'sd']:
        raise ValueError("error_type must be 'sem' or 'sd'")

    common_mice = sorted(set(fluor_mouse_dfs.keys()) & set(deconv_mouse_dfs.keys()))
    rows = []

    for mouse in common_mice:
        fluor_df = fluor_mouse_dfs[mouse]
        deconv_df = deconv_mouse_dfs[mouse]

        if fluor_df is None or len(fluor_df) == 0:
            continue
        if deconv_df is None or len(deconv_df) == 0:
            continue

        if percent_col not in fluor_df.columns or percent_col not in deconv_df.columns:
            raise ValueError(f"Missing column '{percent_col}' for mouse {mouse}")
        if decrease_col not in fluor_df.columns or decrease_col not in deconv_df.columns:
            raise ValueError(f"Missing column '{decrease_col}' for mouse {mouse}")

        fluor_percent = pd.to_numeric(fluor_df[percent_col], errors='coerce').to_numpy(dtype=float)
        deconv_percent = pd.to_numeric(deconv_df[percent_col], errors='coerce').to_numpy(dtype=float)
        fluor_decrease = pd.to_numeric(fluor_df[decrease_col], errors='coerce').to_numpy(dtype=float)
        deconv_decrease = pd.to_numeric(deconv_df[decrease_col], errors='coerce').to_numpy(dtype=float)

        fluor_percent = fluor_percent[np.isfinite(fluor_percent)]
        deconv_percent = deconv_percent[np.isfinite(deconv_percent)]
        fluor_decrease = fluor_decrease[np.isfinite(fluor_decrease)]
        deconv_decrease = deconv_decrease[np.isfinite(deconv_decrease)]

        if len(fluor_percent) == 0 or len(deconv_percent) == 0 or len(fluor_decrease) == 0 or len(deconv_decrease) == 0:
            continue

        if agg == 'mean':
            fluor_percent_val = np.nanmean(fluor_percent)
            deconv_percent_val = np.nanmean(deconv_percent)
            fluor_decrease_val = np.nanmean(fluor_decrease)
            deconv_decrease_val = np.nanmean(deconv_decrease)
        else:
            fluor_percent_val = np.nanmedian(fluor_percent)
            deconv_percent_val = np.nanmedian(deconv_percent)
            fluor_decrease_val = np.nanmedian(fluor_decrease)
            deconv_decrease_val = np.nanmedian(deconv_decrease)

        rows.append({
            'mouse': mouse,
            'fluor_percent': fluor_percent_val,
            'deconv_percent': deconv_percent_val,
            'fluor_decrease': fluor_decrease_val,
            'deconv_decrease': deconv_decrease_val,
            'n_fluor_neurons': len(fluor_percent),
            'n_deconv_neurons': len(deconv_percent),
        })

    mouse_summary_df = pd.DataFrame(rows)
    mouse_summary_df.to_csv(join(save_dir, f'{prefix}_table.csv'), index=False)

    with open(join(save_dir, f'{prefix}_table.pkl'), 'wb') as f:
        pickle.dump(mouse_summary_df, f)

    if len(mouse_summary_df) == 0:
        print('No mouse-level odor data to plot.')
        return mouse_summary_df

    def mean_and_err(x):
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        if len(x) == 0:
            return np.nan, np.nan
        mu = np.mean(x)
        if len(x) == 1:
            err = 0.0
        elif error_type == 'sd':
            err = np.std(x, ddof=1)
        else:
            err = np.std(x, ddof=1) / np.sqrt(len(x))
        return mu, err

    fluor_percent_vals = mouse_summary_df['fluor_percent'].to_numpy(dtype=float)
    deconv_percent_vals = mouse_summary_df['deconv_percent'].to_numpy(dtype=float)
    fluor_decrease_vals = mouse_summary_df['fluor_decrease'].to_numpy(dtype=float)
    deconv_decrease_vals = mouse_summary_df['deconv_decrease'].to_numpy(dtype=float)

    percent_means, percent_errs = zip(*(mean_and_err(arr) for arr in [fluor_percent_vals, deconv_percent_vals]))
    decrease_means, decrease_errs = zip(*(mean_and_err(arr) for arr in [fluor_decrease_vals, deconv_decrease_vals]))

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    n_mice = len(mouse_summary_df)
    offsets = np.array([0.0]) if n_mice == 1 else np.linspace(-jitter, jitter, n_mice)

    # panel 1: % of peak left
    x1 = np.array([0, 1], dtype=float)
    axes[0].bar(x1, percent_means, yerr=percent_errs, capsize=5, width=0.6)

    fluor_x1 = x1[0] + offsets
    deconv_x1 = x1[1] + offsets

    if connect_pairs:
        for i in range(n_mice):
            axes[0].plot(
                [fluor_x1[i], deconv_x1[i]],
                [fluor_percent_vals[i], deconv_percent_vals[i]],
                '-',
                linewidth=1,
                alpha=0.7,
                color='gray'
            )
    # add mouse labels next to points
    for i, mouse in enumerate(mouse_summary_df['mouse']):
        axes[0].text(fluor_x1[i], fluor_percent_vals[i], mouse, fontsize=8, ha='right', va='bottom')
        axes[0].text(deconv_x1[i], deconv_percent_vals[i], mouse, fontsize=8, ha='left', va='bottom')

    axes[0].plot(fluor_x1, fluor_percent_vals, 'o')
    axes[0].plot(deconv_x1, deconv_percent_vals, 'o')
    axes[0].set_xticks(x1)
    axes[0].set_xticklabels([fluor_label, deconv_label])
    axes[0].set_ylabel('% of peak at odor start')
    axes[0].set_title(f'{agg.capitalize()} across neurons, then mean across mice')

    # panel 2: % decrease from peak
    x2 = np.array([0, 1], dtype=float)
    axes[1].bar(x2, decrease_means, yerr=decrease_errs, capsize=5, width=0.6)

    fluor_x2 = x2[0] + offsets
    deconv_x2 = x2[1] + offsets

    if connect_pairs:
        for i in range(n_mice):
            axes[1].plot(
                [fluor_x2[i], deconv_x2[i]],
                [fluor_decrease_vals[i], deconv_decrease_vals[i]],
                '-',
                linewidth=1,
                alpha=0.7,
                color='gray'
            )

    axes[1].plot(fluor_x2, fluor_decrease_vals, 'o')
    axes[1].plot(deconv_x2, deconv_decrease_vals, 'o')
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels([fluor_label, deconv_label])
    axes[1].set_ylabel('% decrease from peak by odor start')
    axes[1].set_title(f'{agg.capitalize()} across neurons, then mean across mice')

    plt.tight_layout()
    plt.savefig(join(save_dir, f'{prefix}.png'))
    plt.savefig(join(save_dir, f'{prefix}.pdf'))
    plt.close(fig)

    return mouse_summary_df


if __name__ == '__main__':
    # -----------------------------------
    # Paths: one folder per mouse with the saved per-neuron summary files
    # -----------------------------------
    mouse_dirs = {
        'mouse14': DATA_ROOT + r'\mouse14\sess11\ML\data\processed_data\opto_decay_trialwise_v3\decay_multisignal',
        'mouse09': DATA_ROOT + r'\mouse09\sess05\ML\data\processed_data\opto_decay_trialwise_v3\decay_multisignal',
        'mouse10': DATA_ROOT + r'\mouse10\sess07\ML\data\processed_data\opto_decay_trialwise_v3\decay_multisignal',
        'mouse05': DATA_ROOT + r'\mouse05\sess03\ML\data\processed_data\opto_decay_trialwise_v3\decay_multisignal',
        'mouse01': DATA_ROOT + r'\mouse01\sess01\ML\data\processed_data\opto_decay_trialwise_v3\decay_multisignal',

    }

    # Filenames inside each mouse folder
    fluor_filename = r'fluorescence\fluorescence_per_neuron.pkl'
    deconv_filename = r'deconvolved_spks\deconvolved_spks_per_neuron.pkl'

    # Output folder
    save_dir = RESULTS_ROOT + r'\decay_histograms_all_mice_V2'

    # Main tau column
    tau_col = 'median_tau_fast_s'   # tau_fast column
    # tau_col = 'median_tau_s'       # alternative column name (when tau_fast was saved as median_tau_s)

    drop_col = 'median_t_2third_drop_s'

    fluor_mouse_dfs, deconv_mouse_dfs = load_mouse_tables(
        mouse_dirs=mouse_dirs,
        fluor_filename=fluor_filename,
        deconv_filename=deconv_filename
    )

    fluor_all = pool_across_mice(fluor_mouse_dfs)
    deconv_all = pool_across_mice(deconv_mouse_dfs)
    mouse_summary_df = plot_average_per_mouse(
        fluor_mouse_dfs=fluor_mouse_dfs,
        deconv_mouse_dfs=deconv_mouse_dfs,
        save_dir=save_dir,
        prefix='average_per_mouse',
        tau_col=tau_col,
        drop_col=drop_col,
        fluor_label='Fluorescence',
        deconv_label='Deconvolved',
        agg='median',
        error_type='sem',
        connect_pairs=True
    )
    plot_pooled_histograms_from_saved(
        fluor_all=fluor_all,
        deconv_all=deconv_all,
        save_dir=save_dir,
        prefix='pooled_across_mice_taufast_and_2third',
        tau_col=tau_col,
        drop_col=drop_col,
        fluor_label='Fluorescence',
        deconv_label='Deconvolved',
        bins=20
    )

    odor_mouse_summary_df = plot_odor_conclusion_across_mice(
        fluor_mouse_dfs=fluor_mouse_dfs,
        deconv_mouse_dfs=deconv_mouse_dfs,
        save_dir=save_dir,
        prefix='odor_conclusion_across_mice',
        percent_col='median_odor_start_to_peak_percent',
        decrease_col='median_odor_start_decrease_percent',
        fluor_label='Fluorescence',
        deconv_label='Deconvolved',
        agg='median',  # or 'mean'
        error_type='sem',
        connect_pairs=True
    )
    print('Done.')
    print(f'Pooled fluorescence neurons: {len(fluor_all)}')
    print(f'Pooled deconvolved neurons: {len(deconv_all)}')