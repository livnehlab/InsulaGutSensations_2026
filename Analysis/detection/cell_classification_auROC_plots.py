#!/usr/bin/env python3
"""
Detection session: loads the pickled auROC results (cell_classification_auROC_V2.py),
computes the sets of neurons with significant responses (light: all light trials vs.
baseline and vs. distractor light only; reward / lick: hit trials), saves the light- and
reward-responsive neuron indices, plots Venn diagrams of the overlaps and saves the overlap
fractions (comparison_fractions_<day>.pkl), used by plot_stacked_bars_across_mice.py and
plot_reward_vs_light_simple.py.
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
from matplotlib_venn import venn2, venn3
import numpy as np
matplotlib.use('Qt5Agg')
matplotlib.rcParams['pdf.fonttype'] = 42


# ---------------------------------------------------------------------------
# Data locations (set these before running)
# ---------------------------------------------------------------------------
DATA_ROOT = r'path\to\2p_data'  # folder with the imaging sessions (<mouse>/<session>/ML/...)


###############################################################################
#                           Plotting Functions                                #
###############################################################################

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

###############################################################################
#                        Utility Functions                                    #
###############################################################################


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
    with open(join(save_loc, 'res_all_light_vs_lightHit.pkl'), 'rb') as f:
        data['res_all_light_vs_lightHit'] = pickle.load(f)
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
    res_all_light_vs_lightHit = data['res_all_light_vs_lightHit']
    auROC_results_hit = data['auROC_results_hit']
    res_reward_vs_lightHit = data['res_reward_vs_lightHit']
    res_lick_on_vs_lightHit = data['res_lick_on_vs_lightHit']

    sig_all_light = {
        'light_on': {'increase': [], 'decrease': []}
    }
    sig_all_light_vs_hit = {
        'light_on': {'increase': [], 'decrease': []}
    }
    sig_hit = {
        'light_on': {'increase': [], 'decrease': []},
        'lick_onset': {'increase': [], 'decrease': []},
        'lick_offset': {'increase': [], 'decrease': []},
        'reward': {'increase': [], 'decrease': []}
    }
    sig_reward_vs_hit = {
        'light_on': {'increase': [], 'decrease': []}
    }
    sig_lick_vs_hit = {
        'light_on': {'increase': [], 'decrease': []}
    }

    # All light
    for i in range(len(res_all_light['final_auroc'])):
        if res_all_light['classification'][i] == 'above_95':
            sig_all_light['light_on']['increase'].append(i)
        elif res_all_light['classification'][i] == 'below_5':
            sig_all_light['light_on']['decrease'].append(i)


    # All light vs Hit (Light on)
    for i in range(len(res_all_light_vs_lightHit['final_auroc'])):
        if res_all_light_vs_lightHit['classification'][i] == 'above_95':
            sig_all_light_vs_hit['light_on']['increase'].append(i)
        elif res_all_light_vs_lightHit['classification'][i] == 'below_5':
            sig_all_light_vs_hit['light_on']['decrease'].append(i)

    # Hit - reward vs. distractor light (hit trials)
    for i in range(len(res_reward_vs_lightHit['final_auroc'])):
        if res_reward_vs_lightHit['classification'][i] == 'above_95':
            sig_reward_vs_hit['light_on']['increase'].append(i)
        elif res_reward_vs_lightHit['classification'][i] == 'below_5':
            sig_reward_vs_hit['light_on']['decrease'].append(i)

    # Hit - lick onset vs. distractor light (hit trials)
    for i in range(len(res_lick_on_vs_lightHit['final_auroc'])):
        if res_lick_on_vs_lightHit['classification'][i] == 'above_95':
            sig_lick_vs_hit['light_on']['increase'].append(i)
        elif res_lick_on_vs_lightHit['classification'][i] == 'below_5':
            sig_lick_vs_hit['light_on']['decrease'].append(i)


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

    return {
        'all_light': sig_all_light,
        'hit': sig_hit,
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
    with open(join(save_loc, 'significant_neurons_hit.pkl'), 'wb') as f:
        pickle.dump(significant_neurons['hit'], f)


def create_venn_sets(significant_neurons):
    """
    Extracts sets of neuron indices from the computed significant neuron dictionaries.
    """
    sig_all_light = significant_neurons['all_light']
    sig_hit = significant_neurons['hit']
    sig_all_light_vs_hit = significant_neurons['all_light_vs_hit']
    sig_reward_vs_hit = significant_neurons['reward_vs_hit']
    sig_lick_vs_hit = significant_neurons['lick_vs_hit']

    venn_sets = {}
    # All light sets
    venn_sets['all_light_on_increase'] = set(sig_all_light['light_on']['increase'])
    venn_sets['all_light_on_decrease'] = set(sig_all_light['light_on']['decrease'])
    venn_sets['all_light_vs_hit_increase'] = set(sig_all_light_vs_hit['light_on']['increase'])
    venn_sets['all_light_vs_hit_decrease'] = set(sig_all_light_vs_hit['light_on']['decrease'])
    # Hit condition sets
    venn_sets['hit_lick_on_increase'] = set(sig_hit['lick_onset']['increase'])
    venn_sets['hit_lick_on_decrease'] = set(sig_hit['lick_onset']['decrease'])
    venn_sets['hit_lick_off_increase'] = set(sig_hit['lick_offset']['increase'])
    venn_sets['hit_lick_off_decrease'] = set(sig_hit['lick_offset']['decrease'])
    venn_sets['hit_light_on_increase'] = set(sig_hit['light_on']['increase'])
    venn_sets['hit_light_on_decrease'] = set(sig_hit['light_on']['decrease'])
    venn_sets['hit_reward_increase'] = set(sig_hit['reward']['increase'])
    venn_sets['hit_reward_decrease'] = set(sig_hit['reward']['decrease'])
    # Comparison sets
    venn_sets['reward_vs_hit_increase'] = set(sig_reward_vs_hit['light_on']['increase'])
    venn_sets['reward_vs_hit_decrease'] = set(sig_reward_vs_hit['light_on']['decrease'])
    venn_sets['lick_vs_hit_increase'] = set(sig_lick_vs_hit['light_on']['increase'])
    venn_sets['lick_vs_hit_decrease'] = set(sig_lick_vs_hit['light_on']['decrease'])

    return venn_sets


def process_significant_neurons(venn_sets):
    """
    Light-responsive neurons: significant for light vs. baseline and also for
    light vs. distractor light only (hit trials). Returns (all, increase, decrease).
    """
    # Set up keys based on neuron type
    all_light_increase_key = 'all_light_on_increase'
    all_light_decrease_key = 'all_light_on_decrease'
    all_light_vs_hit_increase_key = 'all_light_vs_hit_increase'
    all_light_vs_hit_decrease_key = 'all_light_vs_hit_decrease'
    # Keep light-responsive cells that are also significant for light vs. distractor light only
    if len(venn_sets[all_light_increase_key]) > 0:
        real_light_increase = set([i for i in venn_sets[all_light_increase_key] if i in venn_sets[all_light_vs_hit_increase_key]])
        venn_sets[all_light_increase_key] = real_light_increase
    if len(venn_sets[all_light_decrease_key]) > 0:
        real_light_decrease = set([i for i in venn_sets[all_light_decrease_key] if i in venn_sets[all_light_vs_hit_decrease_key]])
        venn_sets[all_light_decrease_key] = real_light_decrease

    all_real_light = real_light_increase.union(real_light_decrease)

    return all_real_light, real_light_increase, real_light_decrease


def combine_hit_neurons(venn_sets, save_loc):
    """
    Reward-responsive neurons: hit-trial reward (or lick-onset) neurons that are also
    significant vs. the distractor light; the reward and lick-onset sets are combined
    and saved.
    """
    venn_sets['hit_reward_increase'] = set([i for i in venn_sets['hit_reward_increase'] if
                      i in venn_sets['reward_vs_hit_increase']])
    venn_sets['hit_reward_decrease'] = set([i for i in venn_sets['hit_reward_decrease'] if
                        i in venn_sets['reward_vs_hit_decrease']])
    venn_sets['hit_lick_on_increase'] = set([i for i in venn_sets['hit_lick_on_increase'] if
                        i in venn_sets['lick_vs_hit_increase']])
    venn_sets['hit_lick_on_decrease'] = set([i for i in venn_sets['hit_lick_on_decrease'] if
                            i in venn_sets['lick_vs_hit_decrease']])
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

def build_collapsed_sets(sig_light_neurons, sig_neurons_reward):
    """
    Build direction-agnostic sets:
      - all_light: any significant light neuron (inc or dec)
      - all_reward: any significant reward neuron (inc or dec, after the filtering/union above)
      - all_both: intersection of the above two
    """
    all_light_set = set(map(int, np.atleast_1d(sig_light_neurons).tolist()))
    all_reward_set = set(map(int, np.atleast_1d(sig_neurons_reward).tolist()))
    all_both_set = all_light_set & all_reward_set
    return all_light_set, all_reward_set, all_both_set
def compute_and_save_fractions(venn_sets, res_all_light, save_loc, day):
    n_total = len(res_all_light['final_auroc'])
    fractions_2set = {}

    # direction-specific comparisons
    fractions_2set["Hit(reward)_LightOn_Increase"] = two_set_fractions(
        venn_sets['hit_reward_increase'], venn_sets['all_light_on_increase'], n_total)
    fractions_2set["Hit(reward)_LightOn_Decrease"] = two_set_fractions(
        venn_sets['hit_reward_decrease'], venn_sets['all_light_on_decrease'], n_total)
    fractions_2set["Hit(LickOff)_LightOn_Increase"] = two_set_fractions(
        venn_sets['hit_lick_off_increase'], venn_sets['all_light_on_increase'], n_total)
    fractions_2set["Hit(LickOff)_LightOn_Decrease"] = two_set_fractions(
        venn_sets['hit_lick_off_decrease'], venn_sets['all_light_on_decrease'], n_total)

    # direction-agnostic comparison (all light vs. all reward)
    if "all_light_all" in venn_sets and "all_reward_all" in venn_sets:
        fractions_2set["Hit(reward)_LightOn_All"] = two_set_fractions(
            venn_sets["all_reward_all"], venn_sets["all_light_all"], n_total
        )

    fractions_dict = {"2set": fractions_2set}

    # counts of the direction-agnostic sets
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
###############################################################################
#                                Main Code                                    #
###############################################################################
def main(day, save_loc):
    # Load the pickled data
    data = load_data(save_loc)

    # Compute significant neurons for each condition
    significant_neurons = compute_significant_neurons(data)

    # Create sets for Venn diagram comparisons
    venn_sets = create_venn_sets(significant_neurons)

    # Process overlaps/differences between sets
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

    # Save the auROC arrays
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

    # Neurons responsive to both light and reward
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

    # Plot Venn diagrams (including all light vs. all reward)
    total_neurons = len(data['res_all_light']['final_auroc'])
    plot_venn_diagrams(venn_sets, total_neurons, save_loc)

    # Overlap fractions (including all light vs. all reward)
    compute_and_save_fractions(venn_sets, data['res_all_light'], save_loc, day)


if __name__ == '__main__':
    # Define session/day and file locations
    day = 'sess14'
    save_loc = DATA_ROOT + r'\mouse16\{}\ML\data\processed_data\auROC_z_no_sub_history'.format(day)
    main(day, save_loc)
