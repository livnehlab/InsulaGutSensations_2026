"""
Psychophysics: across-mice average of the running (sliding-window) classifier.

Loads classifier_results_by_freq_<comparison>.pkl (saved by running_classifier_psycho.py) from
each session folder, and saves in <out_dir>/<comparison>/:
  grand_average_classifier_results_<comparison>.pkl   - mean / SEM across mice for every entry
  combined_accuracy_results_{all, all_trials}.pkl     - (mean, SEM) of the combined accuracy
  per_mouse_results_before_avg.pkl                   - per-mouse combined curves (last entry only)
plus the plots (plot_comb_and_sep from running_classifier_psycho.py) and the average of each
mouse's best frequency. SEM across mice = std (ddof=0) / sqrt(n mice).
"""
import numpy as np
import os
import pickle
from os.path import join, basename, normpath
from typing import List, Dict, Tuple

from matplotlib import pyplot as plt

# ---------------------------------------------------------------------------
# Data locations (set these before running)
# ---------------------------------------------------------------------------
DATA_ROOT    = r'path\to\2p_data'  # folder with the imaging sessions (<mouse>/<session>/ML/...)
RESULTS_ROOT = r'path\to\results'  # folder where the across-mice results are saved

rec_hz = 15
pre_trial_length = 4 * rec_hz
###############################################################################
# 1. --------------------------------------------------------------------------
#    Helper to recursively merge & average the clf_results dictionaries
###############################################################################

def _stack_or_pad(arrays: List[np.ndarray]) -> np.ndarray:
    filtered = []
    for i, a in enumerate(arrays):
        if not isinstance(a, np.ndarray):
            print(f"WARNING: arrays[{i}] is not a numpy array: {a}")
            continue
        if a.ndim != 1:
            print(f"WARNING: arrays[{i}] is not 1D: shape={a.shape}")
            continue
        filtered.append(a)
    if not filtered:
        raise ValueError("No valid 1D arrays to stack or pad.")
    max_len = max(a.shape[0] for a in filtered)
    padded = []
    for a in filtered:
        if a.shape[0] < max_len:
            pad = np.full(max_len - a.shape[0], np.nan)
            a = np.concatenate([a, pad])
        padded.append(a)
    return np.vstack(padded)             # shape: [n_mice × time]

def _average_entry(entry_list: List[Dict]) -> Dict:
    """
    entry_list  : list of per-mouse entries for the same frequency
                  [{'comb': (avg,sem), 'sep': (avgC,avgF,semC,semF)}, ...]
    returns     : dictionary with averaged curves + SEM across mice
    """
    # -- combined -------------------------------------------------------------
    comb_curves  = []
    for e in entry_list:
        arr = e['comb'][0]
        # Skip if all values are NaN
        if np.isnan(arr).all():
            print("Skipping entry with all-NaN comb curve")
            continue
        comb_curves.append(arr)
    if not comb_curves:
        print("All comb curves are NaN or invalid for this frequency, skipping.")
        return None  # this frequency is skipped

    mat          = _stack_or_pad(comb_curves)
    mean_comb    = np.nanmean(mat, axis=0)
    sem_comb_m   = np.nanstd(mat,  axis=0) / np.sqrt(mat.shape[0])

    # -- separate -------------------------------------------------------------
    CR_curves, FA_curves = [], []
    for e in entry_list:
        arr_CR = e['sep'][0]
        arr_FA = e['sep'][1]
        if not np.isnan(arr_CR).all():
            CR_curves.append(arr_CR)
        else:
            print("Skipping entry with all-NaN CR curve")
        if not np.isnan(arr_FA).all():
            FA_curves.append(arr_FA)
        else:
            print("Skipping entry with all-NaN FA curve")
    if not CR_curves or not FA_curves:
        raise ValueError("No valid CR or FA curves to average (all were NaN)")

    CR_curves = _stack_or_pad(CR_curves)
    FA_curves = _stack_or_pad(FA_curves)

    mean_CR    = np.nanmean(CR_curves, axis=0)
    mean_FA    = np.nanmean(FA_curves, axis=0)
    sem_CR_m   = np.nanstd(CR_curves,  axis=0) / np.sqrt(CR_curves.shape[0])
    sem_FA_m   = np.nanstd(FA_curves,  axis=0) / np.sqrt(FA_curves.shape[0])

    return {
        'comb': (mean_comb, sem_comb_m),
        'sep' : (mean_CR,   mean_FA, sem_CR_m, sem_FA_m)
    }

###############################################################################
# 2. --------------------------------------------------------------------------
#    Public function: average_clf_results
###############################################################################
def average_clf_results(mouse_paths: List[str],
                        out_dir: str,
                        pickle_filename: str = "classifier_results_by_freq_CR_FA.pkl",
                        out_name: str = "grand_average.pkl") -> Tuple[Dict, List[Dict], str]:
    """
    Parameters
    ----------
    mouse_paths : list of folders that each contain the specified pickle file
    out_dir     : where to save the averaged pickle
    pickle_filename : name of the pickle file to load from each mouse folder
    out_name    : filename for the averaged pickle

    Returns
    -------
    grand_avg : dict with the same structure as individual results but averaged
    all_results : list of the raw dicts loaded per mouse
    pre_avg_path : path to the saved "before averaging" pickle for later stats
    """
    os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------------------------------------------
    # Load all pickles
    # ------------------------------------------------------------------------
    all_results = []
    loaded_paths = []
    mouse_ids = []
    for p in mouse_paths:
        pkl = join(p, pickle_filename)
        if not os.path.isfile(pkl):
            print(f"⚠️  Missing {pkl} - skipping this mouse")
            continue
        with open(pkl, "rb") as f:
            all_results.append(pickle.load(f))
        loaded_paths.append(pkl)
        mouse_ids.append(basename(normpath(p)))
        print(f"Loaded {pkl}")

    if len(all_results) == 0:
        raise RuntimeError("No valid pickle files found - aborting.")

    # ------------------------------------------------------------------------
    # Save the raw, per-mouse data before any averaging
    # Structure:
    # {
    #   "mouse_ids": [...],
    #   "source_pickles": [...],
    #   "per_freq": { freq_key: [entry_mouse1, entry_mouse2, ...] }
    # }
    # ------------------------------------------------------------------------
    all_freqs = set().union(*[r.keys() for r in all_results])
    per_freq_raw = {}
    for freq in all_freqs:
        per_freq_raw[freq] = [r[freq] for r in all_results if freq in r]




    # ------------------------------------------------------------------------
    # Collect per-frequency entries across mice and compute averages
    # ------------------------------------------------------------------------
    grand_avg: Dict = {}

    def freq_sort_key(x):
        # 'all' and 'all_trials' come first, then the frequencies in increasing order
        return (x not in ('all', 'all_trials'), x)

    for freq in sorted(all_freqs, key=freq_sort_key):
        entries = [r[freq] for r in all_results if freq in r]
        # take the "combined" array of each mouse
        # (the file is overwritten for every entry, so it keeps the last one)
        pre_avg_bundle = [entries[i]['comb'][0] for i in range(len(entries))]
        pre_avg_path = os.path.join(out_dir, "per_mouse_results_before_avg.pkl")
        with open(pre_avg_path, "wb") as f:
            pickle.dump(pre_avg_bundle, f)
        print(f"Saved pre-average bundle to {pre_avg_path}")
        avg_entry = _average_entry(entries)
        if avg_entry is None:
            print(f"⚠️  Skipping frequency {freq} (all entries were NaN or invalid)")
            continue
        grand_avg[freq] = avg_entry

        # Save combined curves and SEM for convenience
        if freq in ('all', 'all_trials'):
            ave_all = grand_avg[freq]['comb'][0]
            sem_all = grand_avg[freq]['comb'][1]
            combined_results_path = os.path.join(out_dir, f'combined_accuracy_results_{freq}.pkl')
            with open(combined_results_path, 'wb') as f:
                pickle.dump((ave_all, sem_all), f)

    # Save the averaged dict
    avg_path = os.path.join(out_dir, out_name)
    with open(avg_path, "wb") as f:
        pickle.dump(grand_avg, f)
    print(f"Saved grand average to {avg_path}")

    return grand_avg, all_results, pre_avg_path

# def average_clf_results(mouse_paths: List[str],
#                         out_dir: str,
#                         out_name: str = "grand_average.pkl") -> Dict:
#     """
#     Parameters
#     ----------
#     mouse_paths : list of folders that each contain
#                   'classifier_results_by_freq.pkl'
#     out_dir     : where to save the averaged pickle
#     out_name    : filename for the pickle
#
#     Returns
#     -------
#     grand_avg : dict with the same structure as individual results but averaged
#     """
#     # ------------------------------------------------------------------------
#     # Load all pickles
#     # ------------------------------------------------------------------------
#     all_results = []
#     for p in mouse_paths:
#         pkl = join(p, "classifier_results_by_freq_CR_FA.pkl")
#         if not os.path.isfile(pkl):
#             print(f"⚠️  Missing {pkl} – skipping this mouse")
#             continue
#         with open(pkl, "rb") as f:
#             all_results.append(pickle.load(f))
#         print(f"Loaded {pkl}")
#
#     if len(all_results) == 0:
#         raise RuntimeError("No valid pickle files found – aborting.")
#
#     # ------------------------------------------------------------------------
#     # Collect per-frequency entries across mice
#     # ------------------------------------------------------------------------
#     grand_avg: Dict = {}
#     all_freqs = set().union(*[r.keys() for r in all_results])
#
#     # --- CHANGED: sort so that 'all' and 'all_trials' come last ---
#     def freq_sort_key(x):
#         # 'all' and 'all_trials' should be last, otherwise sort alphabetically
#         return (x not in ('all', 'all_trials'), x)
#     for freq in sorted(all_freqs, key=freq_sort_key):
#         entries = [r[freq] for r in all_results if freq in r]
#         avg_entry = _average_entry(entries)
#         if avg_entry is None:
#             print(f"⚠️  Skipping frequency {freq} (all entries were NaN or invalid)")
#             continue
#         grand_avg[freq] = avg_entry
#
#         # --- CHANGED: handle both 'all' and 'all_trials' ---
#         if freq in ('all', 'all_trials'):
#             ave_all = grand_avg[freq]['comb'][0]
#             sem_all = grand_avg[freq]['comb'][1]
#             # save the combined curves and SEM
#             combined_results_path = os.path.join(out_dir, f'combined_accuracy_results_{freq}.pkl')
#             with open(combined_results_path, 'wb') as f:
#                 pickle.dump((ave_all, sem_all), f)
#
#     return grand_avg, all_results
###############################################################################
# 2b. -------------------------------------------------------------------------
#    best-frequency average across mice + plot + pickle
###############################################################################
def plot_best_freq_average(all_results: List[Dict],
                           save_dir: str,
                           pre_trial_len: int,
                           rec_hz: int,
                           fname_stub: str = "best_freq_before_avg") -> None:
    """
    For every mouse pick the frequency with the highest *peak* combined accuracy,
    then average those curves across mice and save both the numeric result and
    a figure (PNG + PDF).

    Parameters
    ----------
    all_results : list with one element per mouse (already loaded pickles).
    save_dir    : folder where outputs will be written.
    pre_trial_len, rec_hz : used only for the x-axis.
    fname_stub  : filename stem, e.g. 'best_freq_avg' → best_freq_avg.pkl / .png
    """
    # ---------------------------------------------------------------------
    # 1) collect the "best freq" curve for each mouse
    # ---------------------------------------------------------------------
    best_curves = []

    for mouse_idx, res in enumerate(all_results):
        best_freq, best_curve, best_peak = None, None, -np.inf
        for freq, entry in res.items():
            # skip the pooled entries 'all' and 'all_trials'
            if freq in ("all", "all_trials"):
                continue
            curve = entry["comb"][0]           # 1-D array: accuracy over time
            # --- CHECK ---
            if not isinstance(curve, np.ndarray) or curve.ndim != 1:
                print(f"Mouse {mouse_idx}, freq {freq}: curve is not 1D array, skipping. Got type {type(curve)}, shape {getattr(curve, 'shape', None)}")
                continue
            start = pre_trial_len + 1 * rec_hz  # int(pre_trial_len - 1.5 * rec_hz)
            stop = pre_trial_len + 5 * rec_hz
            local_mean = np.nanmean(curve[start:stop])  # mean 1-5 s after trial start
            peak = local_mean
            # peak  = np.nanmax(curve[start:stop])           # use highest single time-bin
            if peak > best_peak:
                best_peak   = peak
                best_freq   = freq
                best_curve  = curve
        if best_curve is not None:
            best_curves.append(best_curve)
            print(f"Mouse {mouse_idx} best freq: {best_freq} with peak {best_peak:.3f}")
        else:
            print(f"⚠️  Mouse {mouse_idx} had no valid frequencies")

    if len(best_curves) < 2:
        print("Not enough mice for best-freq average – nothing saved")
        return

    # ---------------------------------------------------------------------
    # 2) pad to equal length, then mean + SEM
    # ---------------------------------------------------------------------
    mat        = _stack_or_pad(best_curves)        # shape [n_mice × time]
    mean_curve = np.nanmean(mat, axis=0)
    sem_curve  = np.nanstd(mat,  axis=0) / np.sqrt(mat.shape[0])

    # ---------------------------------------------------------------------
    # 3) write numeric result
    # ---------------------------------------------------------------------
    out_pkl = os.path.join(save_dir, f"{fname_stub}.pkl")
    with open(out_pkl, "wb") as f:
        pickle.dump((mean_curve, sem_curve), f)
    print(f"✓ Saved best-freq average to {out_pkl}")

    # ---------------------------------------------------------------------
    # 4) make a quick plot
    # ---------------------------------------------------------------------
    t_bins = (np.arange(mean_curve.size) - pre_trial_len) / rec_hz  # seconds

    plt.figure(figsize=(4, 3))
    plt.plot(t_bins, mean_curve, lw=2, label="mean best-freq", color="black")
    plt.fill_between(t_bins,
                     mean_curve - sem_curve,
                     mean_curve + sem_curve,
                     alpha=0.25,
                     color="gray")
    plt.axvline(0, color="gray", ls="--", label="distract light onset")
    plt.axhline(0.5, color="gray", ls="--")
    plt.xlabel("time from laser onset [s]")
    plt.ylabel("classifier accuracy")
    plt.ylim(0, 1)
    plt.title("Average accuracy – best freq per mouse")
    plt.legend(loc="lower right")
    for ext in ("png", "pdf"):
        fpath = os.path.join(save_dir, f"{fname_stub}.{ext}")
        plt.savefig(fpath, bbox_inches="tight", dpi=300)
        print(f"✓ Figure saved: {fpath}")
    plt.close()

###############################################################################
# 3. --------------------------------------------------------------------------
#    Main: average each comparison across mice and plot it
###############################################################################
if __name__ == "__main__":
    # This script was run twice:
    #   Naive Bayes (as set here): base_out_dir = ...\running_classifier_psycho_NB and the *_NB.pkl files
    #   SVM: base_out_dir = RESULTS_ROOT + r"\running_classifier_psycho_new_v3" and the same file names
    #        without "_NB" (classifier_results_by_freq_CR_FA.pkl, ..._HIT_CR.pkl, ..._HIT_FA.pkl)
    mice_folders = [
        DATA_ROOT + r"\mouse14\sess12\ML\data\classifier_leading_z_no_sub_history_v2",
        DATA_ROOT + r"\mouse01\sess02\ML\data\classifier_leading_z_no_sub_history_v2",
        DATA_ROOT + r"\mouse05\sess04\ML\data\classifier_leading_z_no_sub_history_v2",
        DATA_ROOT + r"\mouse09\sess06\ML\data\classifier_leading_z_no_sub_history_v2",
        DATA_ROOT + r"\mouse10\sess09\ML\data\classifier_leading_z_no_sub_history_v2"
    ]

    base_out_dir = RESULTS_ROOT + r"\running_classifier_psycho_NB"
    
    # Define the three pickle file types to process
    pickle_configs = [
        {
            "filename": "classifier_results_by_freq_CR_FA_NB.pkl",
            "suffix": "CR_FA",
            "description": "CR vs FA"
        },
        {
            "filename": "classifier_results_by_freq_HIT_CR_NB.pkl",
            "suffix": "HIT_CR",
            "description": "HIT vs CR"
        },
        {
            "filename": "classifier_results_by_freq_HIT_FA_NB.pkl",
            "suffix": "HIT_FA", 
            "description": "HIT vs FA"
        }
    ]
    
    from running_classifier_psycho import plot_comb_and_sep   # running_classifier_psycho.py in the same folder
    
    # Process each pickle file type
    for config in pickle_configs:
        print(f"\n{'='*60}")
        print(f"Processing {config['description']} ({config['filename']})")
        print(f"{'='*60}")
        
        # Create separate output directory for each type
        out_dir = os.path.join(base_out_dir, config['suffix'])
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        
        try:
            avg_dict, all_dict, pre_avg_data = average_clf_results(
                mice_folders, 
                out_dir, 
                pickle_filename=config['filename'],
                out_name=f"grand_average_classifier_results_{config['suffix']}.pkl"
            )

            # save the ave results to a pickle file
            out_name = f"grand_average_classifier_results_{config['suffix']}.pkl"
            with open(join(out_dir, out_name), "wb") as f:
                pickle.dump(avg_dict, f)

            plot_comb_and_sep(avg_dict,
                              pre_trial_len=pre_trial_length,
                              rec_hz=rec_hz,
                              save_dir=out_dir,
                              base_name=f"grand_avg_comb_sep_{config['suffix']}")

            plot_best_freq_average(all_dict,
                                   save_dir=out_dir,
                                   pre_trial_len=pre_trial_length,
                                   rec_hz=rec_hz,
                                   fname_stub=f"best_freq_before_avg_{config['suffix']}")
                                   
            print(f"✓ Successfully processed {config['description']}")
            
        except Exception as e:
            print(f"❌ Error processing {config['description']}: {str(e)}")
            continue
