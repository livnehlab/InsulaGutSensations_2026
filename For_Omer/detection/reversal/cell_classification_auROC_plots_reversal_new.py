#!/usr/bin/env python3
"""
This script loads analysis data (pickled results), computes sets of neurons with
significant responses under various conditions (e.g., correct reject, hit, false alarm),
creates Venn diagrams and bar plots to compare these sets, and computes fractions of overlap.
"""

###############################################################################
#                              Imports                                        #
###############################################################################
import copy
import csv
import math
import os
import pickle
import random
import statistics as stats
from operator import itemgetter
from os.path import join

import h5py
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib_venn import venn2, venn3
import numpy as np
import pandas as pd
import scipy.io
from scipy.stats import wilcoxon, ttest_rel, ttest_ind, ranksums
import seaborn as sns
from sklearn.metrics import roc_auc_score

# Configure matplotlib backend and style
matplotlib.use('Qt5Agg')
matplotlib.rcParams['pdf.fonttype'] = 42

###############################################################################
#                           Plotting Functions                                #
###############################################################################
def plot_increase_decrease_bar(sig_neurons, total_neurons, Title, save_loc):
    """
    Creates a grouped bar plot for each event comparing the percentage of
    increased vs. decreased neurons.
    """
    event_names = list(sig_neurons.keys())
    n_events = len(event_names)

    inc_counts = []
    dec_counts = []
    for e in event_names:
        inc_counts.append((len(sig_neurons[e]['increase']) / total_neurons) * 100)
        dec_counts.append((len(sig_neurons[e]['decrease']) / total_neurons) * 100)

    x = np.arange(n_events)
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, inc_counts, width, label='Increase', color='tomato')
    ax.bar(x + width / 2, dec_counts, width, label='Decrease', color='cornflowerblue')

    ax.set_xticks(x)
    ax.set_xticklabels(event_names, rotation=45, ha='right')
    ax.set_ylabel('% Significant Neurons')
    ax.set_title(Title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(join(save_loc, Title))
    plt.show()


def plot_venn(set1, set2, total, labels, title, filename, save_loc,
              color1='gold', color2='lightpink'):
    """
    Plots a 2-set Venn diagram with percentage labels.
    """
    plt.figure()
    v = venn2([set1, set2], set_labels=labels, set_colors=(color1, color2))
    v.get_label_by_id('10').set_text(f"{(len(set1 - set2) / total * 100):.1f}%")
    v.get_label_by_id('01').set_text(f"{(len(set2 - set1) / total * 100):.1f}%")
    if len(set1 & set2) > 0:
        v.get_label_by_id('11').set_text(f"{(len(set1 & set2) / total * 100):.1f}%")
    plt.title(title)
    plt.savefig(join(save_loc, filename))
    plt.show()


def plot_venn3(set1, set2, set3, total, labels, title, filename, save_loc,
               color1='gold', color2='lightpink', color3='lightblue',
               inter_color1='orange', inter_color2='purple', inter_color3='green',
               center_color='red'):
    """
    Plots a 3-set Venn diagram with percentage labels and custom colors.
    """
    plt.figure()
    v = venn3([set1, set2, set3], set_labels=labels,
              set_colors=(color1, color2, color3))

    label_ids = ['100', '010', '001', '110', '101', '011', '111']
    set_operations = [
        len(set1 - set2 - set3),
        len(set2 - set1 - set3),
        len(set3 - set1 - set2),
        len(set1 & set2 - set3),
        len(set1 & set3 - set2),
        len(set2 & set3 - set1),
        len(set1 & set2 & set3)
    ]

    # Intersection colors mapping
    inter_colors = {
        '110': inter_color1,  # set1 & set2
        '101': inter_color2,  # set1 & set3
        '011': inter_color3,  # set2 & set3
        '111': center_color   # all three
    }

    for label_id, count in zip(label_ids, set_operations):
        label = v.get_label_by_id(label_id)
        if label and count > 0:
            label.set_text(f"{(count / total * 100):.1f}%")

    for region_id, color in inter_colors.items():
        patch = v.get_patch_by_id(region_id)
        if patch:
            patch.set_color(color)
            patch.set_alpha(0.5)

    plt.title(title)
    plt.savefig(join(save_loc, filename))
    plt.show()


###############################################################################
#                        Utility Functions                                    #
###############################################################################
# def exclude_repeated_significance(significant_neurons, events_order):
#     """
#     Removes neurons from later events if they already appeared as significant in earlier events.
#     """
#     for i in range(len(events_order) - 1):
#         e1 = events_order[i]
#         e2 = events_order[i + 1]
#
#         # For 'increase'
#         inc_e1 = set(significant_neurons[e1]['increase'])
#         inc_e2 = set(significant_neurons[e2]['increase'])
#         overlap_inc = inc_e1.intersection(inc_e2)
#         if overlap_inc:
#             new_e2_inc = [n for n in significant_neurons[e2]['increase'] if n not in overlap_inc]
#             significant_neurons[e2]['increase'] = np.array(new_e2_inc, dtype=int)
#
#         # For 'decrease'
#         dec_e1 = set(significant_neurons[e1]['decrease'])
#         dec_e2 = set(significant_neurons[e2]['decrease'])
#         overlap_dec = dec_e1.intersection(dec_e2)
#         if overlap_dec:
#             new_e2_dec = [n for n in significant_neurons[e2]['decrease'] if n not in overlap_dec]
#             significant_neurons[e2]['decrease'] = np.array(new_e2_dec, dtype=int)
#
#     return significant_neurons
#

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


def three_set_fractions(set1, set2, set3, n_total):
    """
    Computes the fractions of neurons (percentages of n_total) for only one set, pairwise intersections, and the three-set intersection.
    """
    if n_total == 0:
        return {
            "set1_only": 0.0, "set2_only": 0.0, "set3_only": 0.0,
            "set1_set2": 0.0, "set1_set3": 0.0, "set2_set3": 0.0,
            "all_three": 0.0
        }

    s1_only_count = len(set1 - set2 - set3)
    s2_only_count = len(set2 - set1 - set3)
    s3_only_count = len(set3 - set1 - set2)
    s1_s2_count = len((set1 & set2) - set3)
    s1_s3_count = len((set1 & set3) - set2)
    s2_s3_count = len((set2 & set3) - set1)
    all3_count = len(set1 & set2 & set3)

    return {
        "set1_only": (s1_only_count / n_total) * 100.0,
        "set2_only": (s2_only_count / n_total) * 100.0,
        "set3_only": (s3_only_count / n_total) * 100.0,
        "set1_set2": (s1_s2_count / n_total) * 100.0,
        "set1_set3": (s1_s3_count / n_total) * 100.0,
        "set2_set3": (s2_s3_count / n_total) * 100.0,
        "all_three": (all3_count / n_total) * 100.0
    }


###############################################################################
#                         Data Processing Functions                           #
###############################################################################
def load_data(save_loc):
    """
    Loads the required pickled data files from the given directory.
    """
    data = {}
    with open(join(save_loc, 'auROC_results_hit.pkl'), 'rb') as f:
        data['auROC_results_hit'] = pickle.load(f)
    with open(join(save_loc, 'auROC_results_miss.pkl'), 'rb') as f:
        data['auROC_results_miss'] = pickle.load(f)
    with open(join(save_loc, 'res_all_light.pkl'), 'rb') as f:
        data['res_all_light'] = pickle.load(f)
    with open(join(save_loc, 'res_all_light_vs_lightCR.pkl'), 'rb') as f:
        data['res_all_light_vs_lightCR'] = pickle.load(f)
    with open(join(save_loc, 'res_CR_vs_lightHit.pkl'), 'rb') as f:
        data['res_hit_vs_lightCR'] = pickle.load(f)
    with open(join(save_loc, 'res_reward_vs_lightHit.pkl'), 'rb') as f:
        data['res_reward_vs_lightHit'] = pickle.load(f)
    with open(join(save_loc, 'res_lick_on_vs_lightHit.pkl'), 'rb') as f:
        data['res_lick_on_vs_lightHit'] = pickle.load(f)
    return data


def compute_significant_neurons(data):
    """
    Computes dictionaries of significant neurons for various conditions.
    """
    res_all_light = data['res_all_light']
    auROC_results_hit = data['auROC_results_hit']
    res_hit_vs_lightCR = data['res_hit_vs_lightCR']
    res_all_light_vs_lightCR = data['res_all_light_vs_lightCR']
    res_reward_vs_lightCR = data['res_reward_vs_lightHit']
    res_lick_on_vs_lightCR = data['res_lick_on_vs_lightHit']

    sig_all_light = {
        'light_on': {'increase': [], 'decrease': []}
    }

    sig_corr_reject = {
        'light_on': {'increase': [], 'decrease': []},
        'light_off': {'increase': [], 'decrease': []},
        'lick_onset': {'increase': [], 'decrease': []},
        'lick_offset': {'increase': [], 'decrease': []},
        'reward': {'increase': [], 'decrease': []}
    }
    sig_hit = {
        'light_on': {'increase': [], 'decrease': []},
        'light_off': {'increase': [], 'decrease': []},
        'lick_onset': {'increase': [], 'decrease': []},
        'lick_offset': {'increase': [], 'decrease': []},
        'reward': {'increase': [], 'decrease': []}
    }
    sig_fa = {
        'light_on': {'increase': [], 'decrease': []},
        'lick_onset': {'increase': [], 'decrease': []},
        'lick_offset': {'increase': [], 'decrease': []},
        'reward': {'increase': [], 'decrease': []}
    }
    sig_miss = {
        'light_on': {'increase': [], 'decrease': []},
        'lick_onset': {'increase': [], 'decrease': []},
        'lick_offset': {'increase': [], 'decrease': []},
        'reward': {'increase': [], 'decrease': []}
    }
    sig_hit_vs_cr = {
        'light_on': {'increase': [], 'decrease': []}
    }
    sig_reward_vs_cr = {
        'light_on': {'increase': [], 'decrease': []}
    }
    sig_lick_vs_cr = {
        'light_on': {'increase': [], 'decrease': []}
    }

    sig_all_light_vs_hit = {'light_on': {'increase': [], 'decrease': []}}
    sig_reward_vs_hit = {'light_on': {'increase': [], 'decrease': []}}
    sig_lick_vs_hit = {'light_on': {'increase': [], 'decrease': []}}
    # All light
    for i in range(len(res_all_light['final_auroc'])):
        if res_all_light['classification'][i] == 'above_95':
            sig_all_light['light_on']['increase'].append(i)
        elif res_all_light['classification'][i] == 'below_5':
            sig_all_light['light_on']['decrease'].append(i)

    for i in range(len(res_all_light_vs_lightCR['final_auroc'])):
        if res_all_light_vs_lightCR['classification'][i] == 'above_95':
            sig_all_light_vs_hit['light_on']['increase'].append(i)
        elif res_all_light_vs_lightCR['classification'][i] == 'below_5':
            sig_all_light_vs_hit['light_on']['decrease'].append(i)

    # Hit - Light off
    for i in range(len(auROC_results_hit['light_off']['auroc'])):
        if auROC_results_hit['light_off']['p_value'][i] == 'above_95':
            sig_hit['light_off']['increase'].append(i)
        elif auROC_results_hit['light_off']['p_value'][i] == 'below_5':
            sig_hit['light_off']['decrease'].append(i)
    # Hit - Lick onset
    for i in range(len(auROC_results_hit['lick_onset']['auroc'])):
        if auROC_results_hit['lick_onset']['p_value'][i] == 'above_95':
            sig_hit['lick_onset']['increase'].append(i)
        elif auROC_results_hit['lick_onset']['p_value'][i] == 'below_5':
              sig_hit['lick_onset']['decrease'].append(i)

    # Hit - Lick offset
    for i in range(len(auROC_results_hit['lick_offset']['auroc'])):
        if auROC_results_hit['lick_offset']['p_value'][i] == 'above_95':
            sig_hit['lick_offset']['increase'].append(i)
        elif auROC_results_hit['lick_offset']['p_value'][i] == 'below_5':
              sig_hit['lick_offset']['decrease'].append(i)

    # Hit - Reward
    for i in range(len(auROC_results_hit['reward']['auroc'])):
        if auROC_results_hit['reward']['p_value'][i] == 'above_95':
            sig_hit['reward']['increase'].append(i)
        elif auROC_results_hit['reward']['p_value'][i] == 'below_5':
              sig_hit['reward']['decrease'].append(i)

    # Hit - reward vs Hit (Light on)
    for i in range(len(res_reward_vs_lightCR['final_auroc'])):
        if res_reward_vs_lightCR['final_auroc'][i] > 0.5 and res_reward_vs_lightCR['classification'][i] == 'above_95':
            sig_reward_vs_cr['light_on']['increase'].append(i)
        elif res_reward_vs_lightCR['final_auroc'][i] < 0.5 and res_reward_vs_lightCR['classification'][i] == 'below_5':
            sig_reward_vs_cr['light_on']['decrease'].append(i)

    # Hit - lick onset vs Hit (Light on)
    for i in range(len(res_lick_on_vs_lightCR['final_auroc'])):
        if res_lick_on_vs_lightCR['classification'][i] == 'above_95':
            sig_lick_vs_cr['light_on']['increase'].append(i)
        elif res_lick_on_vs_lightCR['classification'][i] == 'below_5':
            sig_lick_vs_cr['light_on']['decrease'].append(i)
    # Hit - CR vs Hit (Light on)
    for i in range(len(res_hit_vs_lightCR['final_auroc'])):
        if res_hit_vs_lightCR['classification'][i] == 'above_95':
            sig_hit_vs_cr['light_on']['increase'].append(i)
        elif res_hit_vs_lightCR['classification'][i] == 'below_5':
            sig_hit_vs_cr['light_on']['decrease'].append(i)


    return {
        'all_light': sig_all_light,
        'correct_reject': sig_corr_reject,
        'hit': sig_hit,
        'false_alarm': sig_fa,
        'miss': sig_miss,
        'hit_vs_cr': sig_hit_vs_cr,
        'reward_vs_cr': sig_reward_vs_cr,
        'lick_vs_cr': sig_lick_vs_cr,
        'all_light_vs_hit': sig_all_light_vs_hit,
        'reward_vs_hit': sig_reward_vs_hit,
        'lick_vs_hit': sig_lick_vs_hit
    }


def save_significant_neurons(significant_neurons, save_loc):
    """
    Saves the computed significant neurons dictionaries as pickle files.
    """
    with open(join(save_loc, 'significant_neurons_all_light.pkl'), 'wb') as f:
        pickle.dump(significant_neurons['all_light'], f)
    with open(join(save_loc, 'significant_neurons_correct_reject.pkl'), 'wb') as f:
        pickle.dump(significant_neurons['correct_reject'], f)
    with open(join(save_loc, 'significant_neurons_hit.pkl'), 'wb') as f:
        pickle.dump(significant_neurons['hit'], f)
    with open(join(save_loc, 'significant_neurons_false_alarm.pkl'), 'wb') as f:
        pickle.dump(significant_neurons['false_alarm'], f)
    with open(join(save_loc, 'significant_neurons_miss.pkl'), 'wb') as f:
        pickle.dump(significant_neurons['miss'], f)


def create_venn_sets(significant_neurons):
    """
    Extracts sets of neuron indices from the computed significant neuron dictionaries.
    """
    sig_all_light = significant_neurons['all_light']
    sig_hit = significant_neurons['hit']
    sig_CR = significant_neurons['correct_reject']
    sig_fa = significant_neurons['false_alarm']
    sig_hit_vs_cr = significant_neurons['hit_vs_cr']
    sig_miss = significant_neurons['miss']
    sig_reward_vs_cr = significant_neurons['reward_vs_cr']
    sig_lick_vs_cr = significant_neurons['lick_vs_cr']
    sig_all_light_vs_hit = significant_neurons['all_light_vs_hit']
    sig_reward_vs_hit = significant_neurons['reward_vs_hit']
    sig_lick_vs_hit = significant_neurons['lick_vs_hit']

    venn_sets = {}
    # All light sets
    venn_sets['all_light_on_increase'] = set(sig_all_light['light_on']['increase'])
    venn_sets['all_light_on_decrease'] = set(sig_all_light['light_on']['decrease'])
    # Hit condition sets
    venn_sets['hit_lick_on_increase'] = set(sig_hit['lick_onset']['increase'])
    venn_sets['hit_lick_on_decrease'] = set(sig_hit['lick_onset']['decrease'])
    venn_sets['hit_lick_off_increase'] = set(sig_hit['lick_offset']['increase'])
    venn_sets['hit_lick_off_decrease'] = set(sig_hit['lick_offset']['decrease'])
    venn_sets['hit_light_on_increase'] = set(sig_hit['light_on']['increase'])
    venn_sets['hit_light_on_decrease'] = set(sig_hit['light_on']['decrease'])
    venn_sets['hit_light_off_increase'] = set(sig_hit['light_off']['increase'])
    venn_sets['hit_light_off_decrease'] = set(sig_hit['light_off']['decrease'])
    venn_sets['hit_reward_increase'] = set(sig_hit['reward']['increase'])
    venn_sets['hit_reward_decrease'] = set(sig_hit['reward']['decrease'])
    venn_sets['hit_vs_cr_increase'] = set(sig_hit_vs_cr['light_on']['increase'])
    venn_sets['hit_vs_cr_decrease'] = set(sig_hit_vs_cr['light_on']['decrease'])
    venn_sets['reward_vs_cr_increase'] = set(sig_reward_vs_cr['light_on']['increase'])
    venn_sets['reward_vs_cr_decrease'] = set(sig_reward_vs_cr['light_on']['decrease'])
    venn_sets['lick_vs_cr_increase'] = set(sig_lick_vs_cr['light_on']['increase'])
    venn_sets['lick_vs_cr_decrease'] = set(sig_lick_vs_cr['light_on']['decrease'])
    # Correct Reject (CR) sets
    venn_sets['cr_light_on_increase'] = set(sig_CR['light_on']['increase'])
    venn_sets['cr_light_on_decrease'] = set(sig_CR['light_on']['decrease'])
    # False Alarm (FA) sets
    venn_sets['fa_lick_on_increase'] = set(sig_fa['lick_onset']['increase'])
    venn_sets['fa_lick_on_decrease'] = set(sig_fa['lick_onset']['decrease'])
    # Miss sets
    venn_sets['miss_light_on_increase'] = set(sig_miss['light_on']['increase'])
    venn_sets['miss_light_on_decrease'] = set(sig_miss['light_on']['decrease'])
    # Comparisons between All light and Hit
    venn_sets['all_light_vs_hit_increase'] = set(sig_all_light_vs_hit['light_on']['increase'])
    venn_sets['all_light_vs_hit_decrease'] = set(sig_all_light_vs_hit['light_on']['decrease'])
    venn_sets['reward_vs_hit_increase'] = set(sig_reward_vs_hit['light_on']['increase'])
    venn_sets['reward_vs_hit_decrease'] = set(sig_reward_vs_hit['light_on']['decrease'])
    venn_sets['lick_vs_hit_increase'] = set(sig_lick_vs_hit['light_on']['increase'])
    venn_sets['lick_vs_hit_decrease'] = set(sig_lick_vs_hit['light_on']['decrease'])

    return venn_sets


def process_overlap(venn_sets, res_hit_subsample):
    """
    Processes overlaps/differences between sets to determine detection and stimulation cells.
    """
    # For excitatory neurons (increase)
    if len(venn_sets['all_light_on_increase']) > 0:
        real_light = set([i for i in venn_sets['all_light_on_increase'] if i in venn_sets['hit_vs_cr_increase']]) # take only the stimulation cells from CR
        venn_sets['all_light_on_increase'] = real_light
    if len(venn_sets['hit_light_on_increase']) > 0:
        real_light_lim = set([i for i in venn_sets['hit_light_on_increase'] if i in venn_sets['hit_vs_cr_increase']])
        venn_sets['hit_light_on_increase'] = real_light_lim
    miss_not_hit = venn_sets['miss_light_on_increase'] - venn_sets['miss_light_on_increase']
    hit_not_miss = sorted(list(venn_sets['hit_light_on_increase'] - venn_sets['miss_light_on_increase']))
    hit_not_miss_dist = [res_hit_subsample['all_distributions'][i] for i in hit_not_miss]
    median_dist = np.array([np.median(dist) for dist in hit_not_miss_dist])
    real_diff = np.where(median_dist > 0.59)[0]
    detection_cells = [hit_not_miss[i] for i in real_diff]
    detection_cells = set(detection_cells)
    not_diff = np.array([hit_not_miss[i] for i in range(len(hit_not_miss))
                         if i not in real_diff])
    venn_sets['miss_light_on_increase'] = venn_sets['miss_light_on_increase'].union(not_diff)
    venn_sets['miss_light_on_increase'] = venn_sets['miss_light_on_increase'] - miss_not_hit
    hit_and_miss = venn_sets['hit_light_on_increase'].intersection(venn_sets['miss_light_on_increase'])

    # For inhibitory neurons (decrease)
    if len(venn_sets['all_light_on_decrease']) > 0:
        real_light_inh = set([i for i in venn_sets['all_light_on_decrease'] if i in venn_sets['hit_vs_cr_decrease']])
        venn_sets['all_light_on_decrease'] = real_light_inh
    if len(venn_sets['hit_light_on_decrease']) > 0:
        real_light_inh_lim = set([i for i in venn_sets['hit_light_on_decrease'] if i in venn_sets['hit_vs_cr_decrease']])
        venn_sets['hit_light_on_decrease'] = real_light_inh_lim
    miss_not_hit_inhibitory = venn_sets['miss_light_on_decrease'] - venn_sets['hit_light_on_decrease']
    hit_not_miss_inhibitory = sorted(list(venn_sets['hit_light_on_decrease'] - venn_sets['miss_light_on_decrease']))
    hit_not_miss_inhibitory_dist = [res_hit_subsample['all_distributions'][i] for i in hit_not_miss_inhibitory]
    median_dist_inhibitory = np.array([np.median(dist) for dist in hit_not_miss_inhibitory_dist])
    real_diff_inhibitory = np.where(median_dist_inhibitory < 0.41)[0]
    detection_cells_inhibitory = [hit_not_miss_inhibitory[i] for i in real_diff_inhibitory]
    detection_cells_inhibitory = set(detection_cells_inhibitory)
    not_diff_inhibitory = np.array([hit_not_miss_inhibitory[i] for i in range(len(hit_not_miss_inhibitory))
                                    if i not in real_diff_inhibitory])
    venn_sets['miss_light_on_decrease'] = venn_sets['miss_light_on_decrease'].union(not_diff_inhibitory)
    venn_sets['miss_light_on_decrease'] = venn_sets['miss_light_on_decrease'] - miss_not_hit_inhibitory
    hit_and_miss_inhibitory = venn_sets['hit_light_on_decrease'].intersection(venn_sets['miss_light_on_decrease'])

    # choice probability for each neuron
    # significant_neurons are the neurons that are in real_diff and real_diff_inhibitory
    significant_neurons = real_light.union(real_light_inh)
    significant_neurons = np.asarray(list(significant_neurons))
    cp_dist = [res_hit_subsample['all_distributions'][i] for i in significant_neurons]
    median_dist_cp = np.array([np.median(dist) for dist in cp_dist])


    return detection_cells, detection_cells_inhibitory, hit_and_miss, hit_and_miss_inhibitory, median_dist_cp, significant_neurons

def process_significant_neurons(venn_sets):
    """
    find the significant neurons
    """
    # Set up keys based on neuron type
    all_light_increase_key = 'all_light_on_increase'
    all_light_decrease_key = 'all_light_on_decrease'
    all_light_vs_hit_increase_key = 'all_light_vs_hit_increase'
    all_light_vs_hit_decrease_key = 'all_light_vs_hit_decrease'
    # Filter light-responsive cells to only include those in CR vs hit
    if len(venn_sets[all_light_increase_key]) > 0:
        real_light_increase = set([i for i in venn_sets[all_light_increase_key] if i in venn_sets[all_light_vs_hit_increase_key]])
        venn_sets[all_light_increase_key] = real_light_increase
    if len(venn_sets[all_light_decrease_key]) > 0:
        real_light_decrease = set([i for i in venn_sets[all_light_decrease_key] if i in venn_sets[all_light_vs_hit_decrease_key]])
        venn_sets[all_light_decrease_key] = real_light_decrease

    all_real_light = real_light_increase.union(real_light_decrease)

    return all_real_light, real_light_increase, real_light_decrease

def save_neuron_indices(save_loc, detection_cells, detection_cells_inhibitory, hit_and_miss, hit_and_miss_inhibitory):
    """
    Saves detected and stimulation neuron indices to text files.
    """
    with open(join(save_loc, 'detection_cells_excitation.txt'), 'w') as f:
        for item in detection_cells:
            f.write("%s\n" % item)
    with open(join(save_loc, 'detection_cells_inhibition.txt'), 'w') as f:
        for item in detection_cells_inhibitory:
            f.write("%s\n" % item)
    with open(join(save_loc, 'stimulation_cells.txt'), 'w') as f:
        for item in hit_and_miss:
            f.write("%s\n" % item)
    with open(join(save_loc, 'stimulation_cells_inhibitory.txt'), 'w') as f:
        for item in hit_and_miss_inhibitory:
            f.write("%s\n" % item)


def combine_hit_neurons(venn_sets, save_loc):
    """
    Combines hit neurons from lick onset and reward, then saves the results.
    """
    venn_sets['hit_reward_increase'] = set([i for i in venn_sets['hit_reward_increase'] if
                       i in venn_sets['reward_vs_cr_increase']])
    venn_sets['hit_reward_decrease'] = set([i for i in venn_sets['hit_reward_decrease'] if
                        i in venn_sets['reward_vs_cr_decrease']])
    venn_sets['hit_lick_on_increase'] = set([i for i in venn_sets['hit_lick_on_increase'] if
                        i in venn_sets['lick_vs_cr_increase']])
    venn_sets['hit_lick_on_decrease'] = set([i for i in venn_sets['hit_lick_on_decrease'] if
                            i in venn_sets['lick_vs_cr_decrease']])
    venn_sets['hit_reward_increase'] = venn_sets['hit_reward_increase'].union(venn_sets['hit_lick_on_increase'])
    venn_sets['hit_reward_decrease'] = venn_sets['hit_reward_decrease'].union(venn_sets['hit_lick_on_decrease'])
    with open(join(save_loc, 'significant_neurons_hit_reward_increase.pkl'), 'wb') as f:
        pickle.dump(venn_sets['hit_reward_increase'], f)
    with open(join(save_loc, 'significant_neurons_hit_reward_decrease.pkl'), 'wb') as f:
        pickle.dump(venn_sets['hit_reward_decrease'], f)

    # combine the hit reward neurons of increase and decrease
    significant_reward_neurons = venn_sets['hit_reward_increase'].union(venn_sets['hit_reward_decrease'])
    significant_reward_neurons = np.asarray(list(significant_reward_neurons))
    return significant_reward_neurons
def plot_venn_diagrams(venn_sets, total_neurons, save_loc):
    """
    Calls the plotting functions to generate 2-set and 3-set Venn diagrams.
    """


    # 2-set Venn diagrams
    plot_venn(
        venn_sets['hit_lick_on_increase'], venn_sets['all_light_on_increase'],
        total_neurons,
        labels=('Hit: Lick Onset', 'Light On'),
        title='Hit (Lick On, Increase) vs. Light On',
        filename='Hit_lickOn_vs_lightOn_increase.png',
        save_loc=save_loc
    )

    plot_venn(
        venn_sets['hit_lick_on_decrease'], venn_sets['all_light_on_decrease'],
        total_neurons,
        labels=('Hit: Lick Onset', 'Light On'),
        title='Hit (Lick On, Decrease) vs. Light On',
        filename='Hit_lickOn_vs_lightOn_decrease.png',
        save_loc=save_loc
    )

    plot_venn(
        venn_sets['hit_lick_off_increase'], venn_sets['all_light_on_increase'],
        total_neurons,
        labels=('Hit: Lick Off', 'Light On'),
        title='Hit (Lick Off, Increase) vs. Light On',
        filename='Hit_lickOff_vs_lightOn_increase.png',
        save_loc=save_loc
    )

    plot_venn(
        venn_sets['hit_lick_off_decrease'], venn_sets['all_light_on_decrease'],
        total_neurons,
        labels=('Hit: Lick Off', 'Light On'),
        title='Hit (Lick Off, Decrease) vs. Light On',
        filename='Hit_lickOff_vs_lightOn_decrease.png',
        save_loc=save_loc
    )

    if "all_light_all" in venn_sets and "all_reward_all" in venn_sets:
        plot_venn(
            venn_sets["all_reward_all"], venn_sets["all_light_all"],
            total_neurons,
            labels=("All Reward", "All Light"),
            title="All Reward vs. All Light",
            filename="AllReward_vs_AllLight.png",
            save_loc=save_loc
        )


def compute_and_save_fractions(venn_sets, res_all_light, save_loc, day):
    n_total = len(res_all_light['final_auroc'])
    fractions_2set = {}

    # existing entries
    fractions_2set["Hit(reward)_LightOn_Increase"] = two_set_fractions(
        venn_sets['hit_reward_increase'], venn_sets['all_light_on_increase'], n_total)
    fractions_2set["Hit(reward)_LightOn_Decrease"] = two_set_fractions(
        venn_sets['hit_reward_decrease'], venn_sets['all_light_on_decrease'], n_total)
    fractions_2set["Hit(LickOff)_LightOn_Increase"] = two_set_fractions(
        venn_sets['hit_lick_off_increase'], venn_sets['all_light_on_increase'], n_total)
    fractions_2set["Hit(LickOff)_LightOn_Decrease"] = two_set_fractions(
        venn_sets['hit_lick_off_decrease'], venn_sets['all_light_on_decrease'], n_total)

    # NEW: top-level "All" so plotting code finds it
    if "all_light_all" in venn_sets and "all_reward_all" in venn_sets:
        fractions_2set["Hit(reward)_LightOn_All"] = two_set_fractions(
            venn_sets["all_reward_all"], venn_sets["all_light_all"], n_total
        )

    fractions_dict = {"2set": fractions_2set}

    # optional: keep counts under a collapsed section if you want
    if "all_light_all" in venn_sets and "all_reward_all" in venn_sets:
        fractions_dict["collapsed"] = {
            "counts": {
                "all_light": len(venn_sets["all_light_all"]),
                "all_reward": len(venn_sets["all_reward_all"]),
                "all_both": len(venn_sets["all_light_all"] & venn_sets["all_reward_all"]),
                "n_neurons": n_total
            }
        }

    frac_filename = f"comparison_fractions_{day}.pkl"
    with open(join(save_loc, frac_filename), 'wb') as f:
        pickle.dump(fractions_dict, f)
    print(f"Saved fraction data to {frac_filename}")

def build_collapsed_sets(sig_light_neurons, sig_neurons_reward):
    """
    Build direction-agnostic sets:
      - all_light: any significant light neuron (inc or dec)
      - all_reward: any significant reward neuron (inc or dec, after your filtering/union)
      - all_both: intersection of the above two
    """
    all_light_set = set(map(int, np.atleast_1d(sig_light_neurons).tolist()))
    all_reward_set = set(map(int, np.atleast_1d(sig_neurons_reward).tolist()))
    all_both_set = all_light_set & all_reward_set
    return all_light_set, all_reward_set, all_both_set
###############################################################################
#                                Main Code                                    #
###############################################################################
def main(day, load_loc, save_loc):

    # Load the pickled data
    data = load_data(load_loc)

    # Compute significant neurons for each condition
    significant_neurons = compute_significant_neurons(data)
    save_significant_neurons(significant_neurons, save_loc)

    # Create sets for Venn diagram comparisons
    venn_sets = create_venn_sets(significant_neurons)

    # Process overlaps/differences between sets
    # detection_cells, detection_cells_inhibitory, hit_and_miss, hit_and_miss_inhibitory, CP_all, sig_light_neurons = process_overlap(venn_sets, data['res_hit_subsample'])
    sig_light_neurons, sig_light_increase, sig_light_decrease = process_significant_neurons(venn_sets)
    sig_light_neurons = np.array(list(sig_light_neurons), dtype=int).flatten()
    sig_light_increase = np.array(list(sig_light_increase), dtype=int).flatten()
    sig_light_decrease = np.array(list(sig_light_decrease), dtype=int).flatten()
    with open(join(save_loc, 'significant_light_neurons.pkl'), 'wb') as f:
        pickle.dump(sig_light_neurons, f)
    with open(join(save_loc, 'significant_light_neurons_increase.pkl'), 'wb') as f:
        pickle.dump(sig_light_increase, f)
    with open(join(save_loc, 'significant_light_neurons_decrease.pkl'), 'wb') as f:
        pickle.dump(sig_light_decrease, f)

    # Save AUROC arrays you already extract
    res_all_light = data['res_all_light']['final_auroc']
    with open(join(save_loc, 'auroc_all_light.pkl'), 'wb') as f:
        pickle.dump(res_all_light, f)
    reward_hit_auroc = data['auROC_results_hit']['reward']['auroc']
    with open(join(save_loc, 'auroc_all_reward.pkl'), 'wb') as f:
        pickle.dump(reward_hit_auroc, f)

    # Combine hit condition neurons and save
    sig_neurons_reward = combine_hit_neurons(venn_sets, save_loc)
    with open(join(save_loc, 'significant_neurons_reward.pkl'), 'wb') as f:
        pickle.dump(sig_neurons_reward, f)

    # Both (for completeness you already save it)
    sig_neurons_both = np.intersect1d(sig_light_neurons, sig_neurons_reward)
    with open(join(save_loc, 'significant_neurons_both_light_reward.pkl'), 'wb') as f:
        pickle.dump(sig_neurons_both, f)

    # Build collapsed sets and expose them to venn_sets
    all_light_set, all_reward_set, all_both_set = build_collapsed_sets(
        sig_light_neurons, sig_neurons_reward
    )
    venn_sets["all_light_all"] = all_light_set
    venn_sets["all_reward_all"] = all_reward_set
    venn_sets["all_both_all"] = all_both_set

    # Plot Venns, now includes All vs All if present
    total_neurons = len(data['res_all_light']['final_auroc'])
    plot_venn_diagrams(venn_sets, total_neurons, save_loc)

    # Fractions, now includes collapsed if present
    compute_and_save_fractions(venn_sets, data['res_all_light'], save_loc, day)

if __name__ == '__main__':
    # Define session/day and file locations
    day = '240926'
    load_loc = r'Z:\2p_data\OI152\{}\ML\data\processed_data\auROC'.format(day)
    save_loc = r'Z:\2p_data\OI152\{}\ML\data\processed_data\auROC_v2'.format(day)
    os.makedirs(save_loc, exist_ok=True)
    main(day, load_loc, save_loc)
