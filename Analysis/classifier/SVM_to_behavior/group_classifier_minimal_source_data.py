"""
Psychophysics: across-mice comparison of the behavior with the SVM classifier after removing
the Gaussian or the sigmoid neurons, with a minimal Excel source-data export.

Conditions (classifier results saved by SVM_sigmoid_gaussian_removal.py in
<mouse>/<session>/ML/data/processed_data/new_classifier_gassian_sigmoid_comparison/svm_cell_subset_comparison/):
- All cells
- No Gaussian
- No Sigmoid
- Random removal (mean over the random-removal repeats)

The psychometric curve of each session is computed from the MonkeyLogic .mat files in
<mouse>/<session>/ML (after removing the 23 priming trials) and fitted with a sigmoid to get
the 50% detection threshold. Each mouse's training frequency is the second tested frequency
below its threshold (or the only one, if just one is below).

Figures:
1) group psychometric curves + mean classifier curves, one figure per training frequency
2) example mouse: psychometric curve + all conditions at its threshold-based training frequency
3) threshold vs. delta (classifier accuracy - behavioral detection at the training frequency)
4) bar plot of the delta per condition, and 5) of the delta at test 15 Hz
6) paired per-mouse plot of the delta at test 15 Hz
The CSV tables and minimal_source_data.xlsx are saved in OUT_DIR.
"""

import os
from pathlib import Path
import pickle
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import matplotlib
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import pandas as pd

matplotlib.use('Qt5Agg')
plt.rcParams["font.family"] = "Arial"
plt.rcParams["pdf.fonttype"] = 42

# ---------------------------------------------------------------------------
# Data locations (set these before running)
# ---------------------------------------------------------------------------
DATA_ROOT    = r'path\to\2p_data'  # folder with the imaging sessions (<mouse>/<session>/ML/...)
RESULTS_ROOT = r'path\to\results'  # folder where the across-mice results are saved

# ===================== SETTINGS ===================== #
ROOT = Path(DATA_ROOT)
OUT_DIR = Path(RESULTS_ROOT + r'\group_classifier_vs_psy_diffs_v6_all_conditions_excel_export')
os.makedirs(OUT_DIR, exist_ok=True)

# Sessions (mouse, day)
MOUSE_DAYS = [
    ('mouse01', 'sess02'),
    ('mouse05', 'sess04'),
    ('mouse09', 'sess06'),
    ('mouse10', 'sess09'),
    ('mouse14', 'sess12'),
]

# Which training frequencies to show in the main mean-curves plot
TRAIN_FREQS_TO_PLOT = [0.5, 15]

# Single-mouse example panel
EXAMPLE_MOUSE = 'mouse14'
EXAMPLE_DAY = 'sess12'

N_PRIMING = 23  # priming trials removed from the behavior

# Results folder of each condition, relative to <ROOT>/<mouse>/<day>
CONDITION_ROOTS = {
    'All cells': (
        'ML', 'data', 'processed_data',
        'new_classifier_gassian_sigmoid_comparison',
        'svm_cell_subset_comparison',
        'all_cells'
    ),
    'No Gaussian': (
        'ML', 'data', 'processed_data',
        'new_classifier_gassian_sigmoid_comparison',
        'svm_cell_subset_comparison',
        'no_gaussian'
    ),
    'No Sigmoid': (
        'ML', 'data', 'processed_data',
        'new_classifier_gassian_sigmoid_comparison',
        'svm_cell_subset_comparison',
        'no_sigmoid'
    ),
    'Random removal': (
        'ML', 'data', 'processed_data',
        'new_classifier_gassian_sigmoid_comparison',
        'svm_cell_subset_comparison',
        'random_remove_avg_group_size'
    ),
}

CONDITION_COLORS = {
    'All cells': 'black',
    'No Gaussian': 'red',
    'No Sigmoid': 'blue',
    'Random removal': 'gray',
}

TRAIN_FREQ_LINESTYLES = {
    0.5: '-',
    15: '--',
}

TRAIN_FREQ_MARKERS = {
    0.5: 'o',
    15: 's',
}
# ===================================================== #

# Plotting order of the frequencies, and the order in which the classifier results are saved
psy_x_order = np.array([0, 0.5, 1, 3, 5, 7.5, 10, 15], dtype=float)
orig_order_8 = [0, 15, 10, 7.5, 5, 3, 1, 0.5]
orig_order_9 = [0, 151, 15, 10, 7.5, 5, 3, 1, 0.5]
frequency_map = {6: 0, 7: 15, 8: 10, 9: 7.5, 10: 5, 11: 3, 12: 1, 13: 0.5}  # condition code -> Hz

_RESULTS_CACHE = {}

# Minimal source data: values are in the same percent units as the plots.
source_group = []
source_group_summary = []
source_example = []
source_fit = []


# ===================== HELPERS ===================== #
def sigmoid(x, L, x0, k, b):
    return L / (1 + np.exp(-k * (x - x0))) + b


def load_mat_file(mat_path):
    """Returns the condition codes and the responses (TrialErrors) of one MonkeyLogic .mat file."""
    ml = scipy.io.loadmat(mat_path)
    cond = ml['TrialRecord'][0][0][7][0]
    resp = ml['TrialRecord'][0][0]['TrialErrors'][0]
    return cond, resp


def gather_behavior_trials(path_ml):
    conditions, responses = [], []
    for mat_path in Path(path_ml).glob('*.mat'):
        cond, resp = load_mat_file(mat_path)
        conditions.extend(cond)
        responses.extend(resp)
    return np.asarray(conditions), np.asarray(responses)


def build_psychometric_curve(conditions_played, responses):
    """Fraction of licked trials (Hit = 3 or FA = 5) per frequency, sorted by frequency."""
    response_dict = {f: {'licked': 0, 'total': 0} for f in frequency_map.values()}
    for cond, resp in zip(conditions_played, responses):
        freq = frequency_map[int(cond)]
        if resp in (3, 5):
            response_dict[freq]['licked'] += 1
        response_dict[freq]['total'] += 1

    freqs_sorted = sorted(response_dict.keys())
    frac_licked = [
        response_dict[f]['licked'] / response_dict[f]['total']
        if response_dict[f]['total'] else 0
        for f in freqs_sorted
    ]
    return np.array(freqs_sorted, dtype=float), np.array(frac_licked, dtype=float)


def estimate_threshold_from_fit(psy_x, psy_y_raw):
    """
    Fit sigmoid to raw lick fraction (0..1) and compute the frequency where detection is 50%.
    Detection% = (1 - y_raw) * 100
    """
    p0 = [max(psy_y_raw), np.median(psy_x), 1.0, min(psy_y_raw)]
    try:
        popt, _ = curve_fit(sigmoid, psy_x, psy_y_raw, p0=p0, method='trf', maxfev=10000)
    except Exception:
        return None, None, None

    x_fit = np.linspace(psy_x.min(), psy_x.max(), 400)
    y_fit_raw = sigmoid(x_fit, *popt)
    det_fit = (1 - y_fit_raw) * 100
    thr_idx = np.argmin(np.abs(det_fit - 50))
    threshold = float(x_fit[thr_idx])
    return threshold, x_fit, det_fit


def find_freq_two_below_or_one(threshold, available_freqs):
    """Second frequency below the threshold (or the only one, if just one is below); None if none."""
    if threshold is None:
        return None
    below_sorted = sorted([f for f in available_freqs if f < threshold])
    if len(below_sorted) >= 2:
        return below_sorted[-2]
    elif len(below_sorted) == 1:
        return below_sorted[-1]
    return None


def y_on_curve_at_x(x_grid, y_vals, x0):
    if x0 is None or len(x_grid) != len(y_vals):
        return None
    x_min, x_max = float(np.min(x_grid)), float(np.max(x_grid))
    xq = min(max(float(x0), x_min), x_max)
    f = interp1d(x_grid, y_vals, kind="linear", bounds_error=False, fill_value="extrapolate")
    return float(f(xq))


def reorder_accuracy_curve_for_plot(acc_curve):
    """Reorders a saved accuracy curve to psy_x_order, in percent (0 Hz shown as 1 - accuracy)."""
    acc_curve = np.asarray(acc_curve, dtype=float).copy()

    if len(acc_curve) == 8:
        source_order = orig_order_8
    elif len(acc_curve) == 9:
        source_order = orig_order_9
    else:
        raise ValueError(f'Unexpected accuracy curve length: {len(acc_curve)}')

    acc_curve[0] = 1 - acc_curve[0]
    index_map = [source_order.index(f) for f in psy_x_order]
    return acc_curve[index_map] * 100.0


def reorder_sem_curve_for_plot(sem_curve):
    """Reorders a saved SEM curve to psy_x_order, in percent."""
    sem_curve = np.asarray(sem_curve, dtype=float).copy()

    if len(sem_curve) == 8:
        source_order = orig_order_8
    elif len(sem_curve) == 9:
        source_order = orig_order_9
    else:
        raise ValueError(f'Unexpected SEM curve length: {len(sem_curve)}')

    index_map = [source_order.index(f) for f in psy_x_order]
    return sem_curve[index_map] * 100.0


def load_condition_results(mouse, day, condition_name):
    """Loads svm_results.pkl (or random_summary.pkl for the random removal) of one condition (cached)."""
    cache_key = (mouse, day, condition_name)
    if cache_key in _RESULTS_CACHE:
        return _RESULTS_CACHE[cache_key]

    rel_parts = CONDITION_ROOTS[condition_name]
    root = ROOT / mouse / day
    for part in rel_parts:
        root = root / part

    svm_file = root / 'svm_results.pkl'
    random_file = root / 'random_summary.pkl'

    if svm_file.exists():
        with svm_file.open('rb') as f:
            results = pickle.load(f)
    elif random_file.exists():
        with random_file.open('rb') as f:
            results = pickle.load(f)
    else:
        raise FileNotFoundError(f'No svm_results.pkl or random_summary.pkl in {root}')

    _RESULTS_CACHE[cache_key] = results
    return results


def find_matching_train_freq(results_dict, train_freq_hz):
    for key in results_dict.keys():
        if np.isclose(float(key), float(train_freq_hz)):
            return key
    return None


def load_classifier_curve(mouse, day, condition_name, train_freq_hz):
    """
    Returns:
        acc_curve_percent_reordered, sem_curve_percent_reordered
    """
    try:
        results_dict = load_condition_results(mouse, day, condition_name)
    except Exception as e:
        print(f'WARNING: could not load {condition_name} for {mouse} {day}: {e}')
        return None, None

    matched_key = find_matching_train_freq(results_dict, train_freq_hz)
    if matched_key is None:
        print(f'WARNING: train freq {train_freq_hz} not found for {condition_name}, {mouse} {day}')
        return None, None

    avg_curve = reorder_accuracy_curve_for_plot(results_dict[matched_key]['avg_accuracy'])

    sem_curve = results_dict[matched_key].get('sem_all', None)
    if sem_curve is not None:
        sem_curve = reorder_sem_curve_for_plot(sem_curve)

    return avg_curve, sem_curve


def build_session_behavior(mouse, day):
    """Psychometric curve, sigmoid fit, threshold and training frequency of one session."""
    path_ml = ROOT / mouse / day / 'ML'
    conds, resps = gather_behavior_trials(path_ml)

    conds = conds[N_PRIMING:]
    resps = resps[N_PRIMING:]

    psy_x, psy_y_raw = build_psychometric_curve(conds, resps)
    psy_det = (1 - psy_y_raw) * 100.0
    threshold, x_fit, det_fit = estimate_threshold_from_fit(psy_x, psy_y_raw)
    train_freq = find_freq_two_below_or_one(threshold, psy_x_order[1:])  # exclude 0 Hz

    return {
        'mouse': mouse,
        'day': day,
        'psy_x': psy_x,
        'psy_y_raw': psy_y_raw,
        'psy_det': psy_det,
        'threshold': threshold,
        'x_fit': x_fit,
        'det_fit': det_fit,
        'train_freq_below_threshold': train_freq,
    }


# ===================== BUILD BEHAVIOR CACHE ===================== #
session_behavior = {}
for mouse, day in MOUSE_DAYS:
    try:
        session_behavior[(mouse, day)] = build_session_behavior(mouse, day)
    except Exception as e:
        print(f'WARNING: could not build behavior for {mouse} {day}: {e}')

# ============================================================
# 1) GROUP PSYCHOMETRIC CURVES + MEAN CLASSIFIER CURVES
#    separate figure for each training frequency
# ============================================================

# build psychometric matrix once
psy_curves = []
for i, (mouse, day) in enumerate(MOUSE_DAYS):
    key = (mouse, day)
    if key not in session_behavior:
        continue

    beh = session_behavior[key]
    psy_curves.append(beh['psy_det'])
    for freq, value in zip(beh['psy_x'], beh['psy_det']):
        source_group.append(dict(mouse=mouse, day=day, condition='Behavior',
                                 train_hz=None, test_hz=freq, value_percent=value))

psy_mean = None
psy_sem = None
if len(psy_curves) > 0:
    psy_matrix = np.vstack(psy_curves)
    psy_mean = np.mean(psy_matrix, axis=0)
    psy_sem = np.std(psy_matrix, axis=0, ddof=1) / np.sqrt(psy_matrix.shape[0])
    for freq, mean, sem in zip(psy_x_order, psy_mean, psy_sem):
        source_group_summary.append(dict(condition='Behavior', train_hz=None,
                                         test_hz=freq, mean_percent=mean,
                                         sem_percent=sem, n_mice=psy_matrix.shape[0]))

for train_freq in TRAIN_FREQS_TO_PLOT:
    plt.figure(figsize=(15, 7))

    # faint psychometric per mouse
    for i, (mouse, day) in enumerate(MOUSE_DAYS):
        key = (mouse, day)
        if key not in session_behavior:
            continue

        beh = session_behavior[key]

        plt.plot(
            beh['psy_x'],
            beh['psy_det'],
            color='lightcoral',
            lw=1.5,
            alpha=0.35
        )
        plt.scatter(
            beh['psy_x'],
            beh['psy_det'],
            color='lightcoral',
            s=18,
            alpha=0.35
        )

    # mean psychometric
    if psy_mean is not None:
        plt.errorbar(
            psy_x_order,
            psy_mean,
            yerr=psy_sem,
            color='crimson',
            lw=3,
            marker='D',
            markersize=6,
            capsize=4,
            label='Behavior mean'
        )

    # classifier means for all conditions for this training freq only
    for condition_name in CONDITION_ROOTS.keys():
        curves = []

        for mouse, day in MOUSE_DAYS:
            acc_curve, _ = load_classifier_curve(mouse, day, condition_name, train_freq)
            if acc_curve is not None:
                curves.append(acc_curve)
                for freq, value in zip(psy_x_order, acc_curve):
                    source_group.append(dict(mouse=mouse, day=day, condition=condition_name,
                                             train_hz=train_freq, test_hz=freq, value_percent=value))

        if len(curves) == 0:
            continue

        curve_mat = np.vstack(curves)
        mean_curve = np.mean(curve_mat, axis=0)
        sem_curve = np.std(curve_mat, axis=0, ddof=1) / np.sqrt(curve_mat.shape[0])
        for freq, mean, sem in zip(psy_x_order, mean_curve, sem_curve):
            source_group_summary.append(dict(condition=condition_name, train_hz=train_freq,
                                             test_hz=freq, mean_percent=mean,
                                             sem_percent=sem, n_mice=curve_mat.shape[0]))

        plt.errorbar(
            psy_x_order,
            mean_curve,
            yerr=sem_curve,
            lw=2.5,
            color=CONDITION_COLORS[condition_name],
            linestyle='-',
            marker=TRAIN_FREQ_MARKERS.get(train_freq, 'o'),
            markersize=5,
            capsize=4,
            label=f'{condition_name} | train {train_freq:g} Hz'
        )

    plt.axhline(50, color='gray', linestyle='--', lw=1, alpha=0.5)
    plt.xlabel('Optogenetic stimulation frequency (Hz)')
    plt.ylabel('Detection / accuracy (%)')
    plt.xticks(psy_x_order, [f'{f:g} Hz' for f in psy_x_order])
    plt.yticks(np.arange(0, 101, 10))
    plt.title(f'Psychometric curves and mean classifier accuracy | train {train_freq:g} Hz')
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()

    freq_tag = str(train_freq).replace('.', 'p')
    plt.savefig(OUT_DIR / f'group_psychometric_and_classifier_train_{freq_tag}Hz.png', dpi=300)
    plt.savefig(OUT_DIR / f'group_psychometric_and_classifier_train_{freq_tag}Hz.pdf', dpi=300)
    plt.show()

# ============================================================
# 2) SINGLE MOUSE: PSYCHOMETRIC + ALL CONDITIONS AT THRESHOLD-BASED TRAIN FREQ
# ============================================================
example_key = (EXAMPLE_MOUSE, EXAMPLE_DAY)
if example_key in session_behavior:
    beh = session_behavior[example_key]
    threshold = beh['threshold']
    train_freq = beh['train_freq_below_threshold']
    for freq, value in zip(beh['psy_x'], beh['psy_det']):
        source_example.append(dict(mouse=EXAMPLE_MOUSE, day=EXAMPLE_DAY,
                                   condition='Behavior', train_hz=None, test_hz=freq,
                                   value_percent=value, sem_percent=None))
    if beh['x_fit'] is not None:
        source_fit = [dict(frequency_hz=f, detection_percent=v, threshold_hz=threshold)
                      for f, v in zip(beh['x_fit'], beh['det_fit'])]

    plt.figure(figsize=(9, 6))

    plt.scatter(
        beh['psy_x'],
        beh['psy_det'],
        s=55,
        color='crimson',
        label=f'{EXAMPLE_MOUSE} behavior (raw)'
    )

    if beh['x_fit'] is not None:
        plt.plot(
            beh['x_fit'],
            beh['det_fit'],
            lw=2.5,
            color='crimson',
            alpha=0.85,
            label='Behavior fit'
        )
        plt.axvline(
            threshold,
            color='crimson',
            linestyle=':',
            lw=1.5,
            label=f'Threshold ≈ {threshold:.2f} Hz'
        )

    if train_freq is not None:
        for condition_name in CONDITION_ROOTS.keys():
            acc_curve, sem_curve = load_classifier_curve(
                EXAMPLE_MOUSE, EXAMPLE_DAY, condition_name, train_freq
            )
            if acc_curve is None:
                continue
            for j, freq in enumerate(psy_x_order):
                source_example.append(dict(mouse=EXAMPLE_MOUSE, day=EXAMPLE_DAY,
                                           condition=condition_name, train_hz=train_freq,
                                           test_hz=freq, value_percent=acc_curve[j],
                                           sem_percent=sem_curve[j] if sem_curve is not None else None))

            plt.errorbar(
                psy_x_order,
                acc_curve,
                yerr=sem_curve,
                lw=2,
                color=CONDITION_COLORS[condition_name],
                capsize=4,
                marker='o',
                markersize=5,
                label=f'{condition_name} (train {train_freq:g} Hz)'
            )

    plt.axhline(50, color='gray', linestyle='--', lw=1)
    plt.xlabel('Optogenetic stimulation frequency (Hz)')
    plt.ylabel('Detection / accuracy (%)')
    plt.title(f'{EXAMPLE_MOUSE}: behavior and classifiers across conditions')
    plt.xticks(psy_x_order, [f'{f:g} Hz' for f in psy_x_order])
    plt.yticks(np.arange(0, 101, 10))
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f'{EXAMPLE_MOUSE}_psychometric_plus_all_conditions.png', dpi=300)
    plt.savefig(OUT_DIR / f'{EXAMPLE_MOUSE}_psychometric_plus_all_conditions.pdf', dpi=300)
    plt.show()


# ============================================================
# 3) BUILD DELTA TABLE FOR ALL CONDITIONS
#    delta = accuracy_at_train_freq - behavioral_detection_at_train_freq
# ============================================================
delta_records = []

for mouse, day in MOUSE_DAYS:
    key = (mouse, day)
    if key not in session_behavior:
        continue

    beh = session_behavior[key]
    threshold = beh['threshold']
    f_train = beh['train_freq_below_threshold']

    if threshold is None or f_train is None:
        continue

    # behavioral detection at training frequency from raw psychometric
    psy_x = beh['psy_x']
    psy_y_raw = beh['psy_y_raw']
    train_idx_psy = int(np.argmin(np.abs(psy_x - f_train)))
    beh_det_at_train = float((1.0 - psy_y_raw[train_idx_psy]) * 100.0)

    for condition_name in CONDITION_ROOTS.keys():
        acc_curve, sem_curve = load_classifier_curve(mouse, day, condition_name, f_train)
        if acc_curve is None:
            continue

        train_idx = int(np.argmin(np.abs(psy_x_order - f_train)))
        acc_at_train = float(acc_curve[train_idx])

        if sem_curve is not None and len(sem_curve) == len(acc_curve):
            delta_sem = float(sem_curve[train_idx])
        else:
            delta_sem = 0.0

        delta_val = acc_at_train - beh_det_at_train

        delta_records.append({
            'mouse': mouse,
            'day': day,
            'session': f'{mouse}_{day}',
            'condition': condition_name,
            'threshold': threshold,
            'train_freq': f_train,
            'behavior_detection_at_train': beh_det_at_train,
            'classifier_accuracy_at_train': acc_at_train,
            'delta': delta_val,
            'delta_sem': delta_sem
        })

        print(
            f"[DEBUG] {mouse} | {condition_name} | thr={threshold:.2f} Hz | "
            f"train={f_train:g} Hz | acc={acc_at_train:.2f}% | "
            f"beh={beh_det_at_train:.2f}% -> Δ={delta_val:.2f}% | SEM={delta_sem:.2f}%"
        )

df_delta = pd.DataFrame(delta_records)
df_delta.to_csv(OUT_DIR / 'delta_all_conditions_raw.csv', index=False)

if df_delta.empty:
    raise RuntimeError('No delta data could be computed.')

# ============================================================
# 3b) BUILD DELTA TABLE FOR ALL CONDITIONS AT TEST 15 Hz
#     delta_15 = accuracy_at_test_15Hz - behavioral_detection_at_15Hz
#     while keeping each mouse's own training frequency (2 below threshold)
# ============================================================
TEST_FREQ = 15.0

delta15_records = []

for mouse, day in MOUSE_DAYS:
    key = (mouse, day)
    if key not in session_behavior:
        continue

    beh = session_behavior[key]
    threshold = beh['threshold']
    f_train = beh['train_freq_below_threshold']

    if threshold is None or f_train is None:
        continue

    # behavioral detection at 15 Hz from raw psychometric
    psy_x = beh['psy_x']
    psy_y_raw = beh['psy_y_raw']

    if not np.any(np.isclose(psy_x, TEST_FREQ)):
        print(f'WARNING: {mouse} {day} has no 15 Hz point in psychometric curve')
        continue

    test_idx_psy = int(np.where(np.isclose(psy_x, TEST_FREQ))[0][0])
    beh_det_at_15 = float((1.0 - psy_y_raw[test_idx_psy]) * 100.0)

    for condition_name in CONDITION_ROOTS.keys():
        acc_curve, sem_curve = load_classifier_curve(mouse, day, condition_name, f_train)
        if acc_curve is None:
            continue

        # classifier accuracy at the 15 Hz TEST point
        test_idx = int(np.where(np.isclose(psy_x_order, TEST_FREQ))[0][0])
        acc_at_15 = float(acc_curve[test_idx])

        if sem_curve is not None and len(sem_curve) == len(acc_curve):
            delta15_sem = float(sem_curve[test_idx])
        else:
            delta15_sem = 0.0

        delta15_val = acc_at_15 - beh_det_at_15

        delta15_records.append({
            'mouse': mouse,
            'day': day,
            'session': f'{mouse}_{day}',
            'condition': condition_name,
            'threshold': threshold,
            'train_freq': f_train,
            'behavior_detection_at_15': beh_det_at_15,
            'classifier_accuracy_at_15': acc_at_15,
            'delta_15': delta15_val,
            'delta_15_sem': delta15_sem
        })

        print(
            f"[DEBUG 15Hz] {mouse} | {condition_name} | thr={threshold:.2f} Hz | "
            f"train={f_train:g} Hz | acc@15={acc_at_15:.2f}% | "
            f"beh@15={beh_det_at_15:.2f}% -> Δ15={delta15_val:.2f}% | SEM={delta15_sem:.2f}%"
        )

df_delta15 = pd.DataFrame(delta15_records)
df_delta15.to_csv(OUT_DIR / 'delta15_all_conditions_raw.csv', index=False)

if df_delta15.empty:
    raise RuntimeError('No 15 Hz delta data could be computed.')

# ============================================================
# 4) SCATTER: THRESHOLD VS DELTA FOR ALL CONDITIONS
# ============================================================
plt.figure(figsize=(9, 6))

cond_names = list(CONDITION_ROOTS.keys())
x_offsets = np.linspace(-0.18, 0.18, len(cond_names))

for cond_idx, condition_name in enumerate(cond_names):
    df_cond = df_delta[df_delta['condition'] == condition_name].copy()
    if df_cond.empty:
        continue

    x_vals = df_cond['threshold'].values + x_offsets[cond_idx]
    y_vals = df_cond['delta'].values
    y_errs = df_cond['delta_sem'].values

    plt.errorbar(
        x_vals,
        y_vals,
        yerr=y_errs,
        fmt='o',
        ms=7,
        capsize=5,
        color=CONDITION_COLORS[condition_name],
        ecolor=CONDITION_COLORS[condition_name],
        elinewidth=1.2,
        alpha=0.85,
        label=condition_name
    )

    for xv, yv, mouse, ftrain in zip(
        x_vals,
        y_vals,
        df_cond['mouse'].values,
        df_cond['train_freq'].values
    ):
        plt.annotate(
            f'{mouse} ({ftrain:g})',
            (xv, yv),
            xytext=(4, 2),
            textcoords='offset points',
            fontsize=7,
            color=CONDITION_COLORS[condition_name]
        )

plt.axhline(0, color='black', lw=1, linestyle='--')

# Add group mean ± SEM for each condition
x_band = [0, 15]

for condition_name in cond_names:
    df_cond = df_delta[df_delta['condition'] == condition_name].copy()

    if df_cond.empty:
        continue

    condition_mean = df_cond['delta'].mean()
    condition_sem = df_cond['delta'].std(ddof=1) / np.sqrt(len(df_cond))

    plt.axhline(
        condition_mean,
        color=CONDITION_COLORS[condition_name],
        lw=1.5,
        linestyle='--',
        alpha=0.8,
        label=f'{condition_name} mean'
    )

    plt.fill_between(
        x_band,
        condition_mean - condition_sem,
        condition_mean + condition_sem,
        color=CONDITION_COLORS[condition_name],
        alpha=0.10,
        label=f'{condition_name} SEM'
    )
plt.xlim(-0.5, 15.5)

xticks = [0, 1, 3, 5, 7.5, 10, 15]
plt.xticks(
    xticks,
    [f'{x:.1f}' for x in xticks],
    rotation=45,
    ha='right'
)
plt.xlabel('Behavioral threshold (Hz)')
plt.ylabel('Δ (accuracy - behavioral detection) (%)')
plt.title('Threshold vs delta across conditions')
plt.tight_layout()
plt.legend(fontsize=9)
plt.savefig(OUT_DIR / 'scatter_threshold_vs_delta_all_conditions.png', dpi=300)
plt.savefig(OUT_DIR / 'scatter_threshold_vs_delta_all_conditions.pdf', dpi=300)
plt.show()


# ============================================================
# 5) BAR PLOT: GROUP MEAN DELTA BY CONDITION + INDIVIDUAL MICE
#    with lines connecting the same mouse across conditions
# ============================================================
summary = (
    df_delta.groupby('condition')['delta']
    .agg(['mean', 'std', 'count'])
    .reset_index()
)
summary['sem'] = summary['std'] / np.sqrt(summary['count'])
summary.to_csv(OUT_DIR / 'delta_all_conditions_summary.csv', index=False)

plt.figure(figsize=(7, 5))
x = np.arange(len(cond_names))
bar_width = 0.65

plt.bar(
    x,
    [summary.loc[summary['condition'] == c, 'mean'].values[0] if c in summary['condition'].values else np.nan for c in cond_names],
    yerr=[summary.loc[summary['condition'] == c, 'sem'].values[0] if c in summary['condition'].values else np.nan for c in cond_names],
    capsize=6,
    width=bar_width,
    color=[CONDITION_COLORS[c] for c in cond_names],
    alpha=0.35,
    edgecolor='black',
    linewidth=1.2
)

rng = np.random.default_rng(0)

# store x positions per mouse per condition, so the same mouse can be connected across conditions
mouse_point_positions = {}

for i, condition_name in enumerate(cond_names):
    df_cond = df_delta[df_delta['condition'] == condition_name].copy()
    if df_cond.empty:
        continue

    jitter = rng.uniform(-0.08, 0.08, size=len(df_cond))
    x_vals = np.full(len(df_cond), x[i]) + jitter
    y_vals = df_cond['delta'].values

    plt.scatter(
        x_vals,
        y_vals,
        color=CONDITION_COLORS[condition_name],
        s=45,
        alpha=0.85,
        zorder=3
    )

    # save exact plotted x/y for each mouse
    for xv, yv, mouse in zip(x_vals, y_vals, df_cond['mouse'].values):
        if mouse not in mouse_point_positions:
            mouse_point_positions[mouse] = {}
        mouse_point_positions[mouse][condition_name] = (xv, yv)

# connect the same mouse across conditions
for mouse, cond_dict in mouse_point_positions.items():
    line_x = []
    line_y = []

    for condition_name in cond_names:
        if condition_name in cond_dict:
            xv, yv = cond_dict[condition_name]
            line_x.append(xv)
            line_y.append(yv)

    if len(line_x) >= 2:
        plt.plot(
            line_x,
            line_y,
            color='gray',
            alpha=0.5,
            linewidth=1.2,
            zorder=2
        )

plt.axhline(0, color='black', lw=1, linestyle='--')
plt.xticks(x, cond_names, rotation=15)
plt.ylabel('Δ (accuracy - behavioral detection) (%)')
plt.title('Group delta by condition')
plt.tight_layout()
plt.savefig(OUT_DIR / 'bar_delta_all_conditions.png', dpi=300)
plt.savefig(OUT_DIR / 'bar_delta_all_conditions.pdf', dpi=300)
plt.show()

# ============================================================
# 6) BAR PLOT: GROUP MEAN DELTA AT TEST 15 Hz BY CONDITION
#    using each mouse's own train freq (2 below threshold)
#    + individual mice + connecting lines
# ============================================================
summary15 = (
    df_delta15.groupby('condition')['delta_15']
    .agg(['mean', 'std', 'count'])
    .reset_index()
)
summary15['sem'] = summary15['std'] / np.sqrt(summary15['count'])
summary15.to_csv(OUT_DIR / 'delta15_all_conditions_summary.csv', index=False)

plt.figure(figsize=(7, 5))
x = np.arange(len(cond_names))
bar_width = 0.65

plt.bar(
    x,
    [summary15.loc[summary15['condition'] == c, 'mean'].values[0] if c in summary15['condition'].values else np.nan for c in cond_names],
    yerr=[summary15.loc[summary15['condition'] == c, 'sem'].values[0] if c in summary15['condition'].values else np.nan for c in cond_names],
    capsize=6,
    width=bar_width,
    color=[CONDITION_COLORS[c] for c in cond_names],
    alpha=0.35,
    edgecolor='black',
    linewidth=1.2
)

rng = np.random.default_rng(1)

# store x/y positions for connecting same mouse across conditions
mouse_point_positions_15 = {}

for i, condition_name in enumerate(cond_names):
    df_cond = df_delta15[df_delta15['condition'] == condition_name].copy()
    if df_cond.empty:
        continue

    jitter = rng.uniform(-0.08, 0.08, size=len(df_cond))
    x_vals = np.full(len(df_cond), x[i]) + jitter
    y_vals = df_cond['delta_15'].values

    plt.scatter(
        x_vals,
        y_vals,
        color=CONDITION_COLORS[condition_name],
        s=45,
        alpha=0.85,
        zorder=3
    )

    for xv, yv, mouse in zip(x_vals, y_vals, df_cond['mouse'].values):
        if mouse not in mouse_point_positions_15:
            mouse_point_positions_15[mouse] = {}
        mouse_point_positions_15[mouse][condition_name] = (xv, yv)

# connect same mouse across conditions
for mouse, cond_dict in mouse_point_positions_15.items():
    line_x = []
    line_y = []

    for condition_name in cond_names:
        if condition_name in cond_dict:
            xv, yv = cond_dict[condition_name]
            line_x.append(xv)
            line_y.append(yv)

    if len(line_x) >= 2:
        plt.plot(
            line_x,
            line_y,
            color='gray',
            alpha=0.5,
            linewidth=1.2,
            zorder=2
        )

plt.axhline(0, color='black', lw=1, linestyle='--')
plt.xticks(x, cond_names, rotation=15)
plt.ylabel('Δ at test 15 Hz (accuracy - behavioral detection) (%)')
plt.title('Group delta at test 15 Hz\n(train freq = 2 below threshold for each mouse)')
plt.tight_layout()
plt.savefig(OUT_DIR / 'bar_delta15_all_conditions.png', dpi=300)
plt.savefig(OUT_DIR / 'bar_delta15_all_conditions.pdf', dpi=300)
plt.show()

# ============================================================
# 7) PAIRED MOUSE PLOT: DELTA AT TEST 15 Hz
# ============================================================
plt.figure(figsize=(7, 5))

mouse_offsets = {}
for i, mouse in enumerate(sorted(df_delta15['mouse'].unique())):
    mouse_offsets[mouse] = i

for mouse in sorted(df_delta15['mouse'].unique()):
    df_mouse = df_delta15[df_delta15['mouse'] == mouse].copy()

    xs = []
    ys = []
    colors = []

    for i, condition_name in enumerate(cond_names):
        row = df_mouse[df_mouse['condition'] == condition_name]
        if row.empty:
            continue
        xs.append(i)
        ys.append(float(row['delta_15'].iloc[0]))
        colors.append(CONDITION_COLORS[condition_name])

    if len(xs) >= 2:
        plt.plot(xs, ys, color='lightgray', linewidth=1.5, zorder=1)

    for x0, y0, c in zip(xs, ys, colors):
        plt.scatter(x0, y0, color=c, s=55, zorder=3)

plt.axhline(0, color='black', lw=1, linestyle='--')
plt.xticks(np.arange(len(cond_names)), cond_names, rotation=15)
plt.ylabel('Δ at test 15 Hz (accuracy - behavioral detection) (%)')
plt.title('Per-mouse paired comparison at test 15 Hz')
plt.tight_layout()
plt.savefig(OUT_DIR / 'paired_delta15_all_conditions.png', dpi=300)
plt.savefig(OUT_DIR / 'paired_delta15_all_conditions.pdf', dpi=300)
plt.show()

# ===================== MINIMAL EXCEL SOURCE DATA ===================== #
# Group curves: individual mouse values used for the plotted mean +/- SEM.
# Delta sheets are shared by the scatter/bar/paired panels, avoiding duplicates.
# Sheet -> figure mapping (PNG and PDF contain the same values):
# group_curves + group_mean_SEM -> group_psychometric_and_classifier_train_*Hz
# example_points + example_fit -> {EXAMPLE_MOUSE}_psychometric_plus_all_conditions
# delta_points + delta_mean_SEM -> scatter_threshold_vs_delta_all_conditions,
#                                bar_delta_all_conditions
# delta15_points + delta15_mean_SEM -> bar_delta15_all_conditions,
#                                     paired_delta15_all_conditions
# Detection/accuracy and SEM are in percent; deltas are percentage points.
# Blank train_hz for Behavior means the values are shared by both training panels.
# Single-mouse classifier SEM is loaded from sem_all; underlying classifier-run
# values are not read by this script and cannot be recovered from mean/SEM alone.
source_delta = df_delta[['mouse', 'day', 'condition', 'threshold', 'train_freq',
                         'delta', 'delta_sem']].copy()
source_delta['scatter_x_hz'] = source_delta['threshold'] + source_delta['condition'].map(
    dict(zip(cond_names, x_offsets)))
source_delta['bar_x'] = [mouse_point_positions[m][c][0]
                         for m, c in zip(source_delta['mouse'], source_delta['condition'])]
source_delta15 = df_delta15[['mouse', 'day', 'condition', 'train_freq', 'delta_15']].copy()
source_delta15['bar_x'] = [mouse_point_positions_15[m][c][0]
                           for m, c in zip(source_delta15['mouse'], source_delta15['condition'])]
source_delta15['paired_x'] = source_delta15['condition'].map(dict(zip(cond_names, range(len(cond_names)))))

excel_path = OUT_DIR / 'minimal_source_data.xlsx'
with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
    pd.DataFrame(source_group).to_excel(writer, sheet_name='group_curves', index=False)
    pd.DataFrame(source_group_summary).to_excel(writer, sheet_name='group_mean_SEM', index=False)
    pd.DataFrame(source_example).to_excel(writer, sheet_name='example_points', index=False)
    pd.DataFrame(source_fit).to_excel(writer, sheet_name='example_fit', index=False)
    source_delta.to_excel(writer, sheet_name='delta_points', index=False)
    source_delta15.to_excel(writer, sheet_name='delta15_points', index=False)
    summary[['condition', 'mean', 'sem']].to_excel(writer, sheet_name='delta_mean_SEM', index=False)
    summary15[['condition', 'mean', 'sem']].to_excel(writer, sheet_name='delta15_mean_SEM', index=False)
print(f'Saved minimal Excel source data: {excel_path}')
