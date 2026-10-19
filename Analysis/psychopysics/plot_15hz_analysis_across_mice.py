import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
from scipy.stats import sem, ttest_1samp, wilcoxon
import matplotlib.patches as mpatches
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib parameters
plt.rcParams["font.family"] = "Arial"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["figure.dpi"] = 300


def load_15hz_analysis_data(mouse_dirs: List[str], analysis_filename: str = "non_15hz_to_15hz_analysis_{day}.pkl") -> pd.DataFrame:
    """
    Load 15Hz Analysis data from multiple mouse directories.
    
    Parameters:
    -----------
    mouse_dirs : List[str]
        List of paths to mouse data directories
    analysis_filename : str
        Template filename for the Analysis files (with {day} placeholder)
    
    Returns:
    --------
    pd.DataFrame
        Combined dataframe with all mouse data
    """
    all_data = []
    
    for mouse_dir in mouse_dirs:
        mouse_path = Path(mouse_dir)
        
        # Extract mouse ID from the full path (e.g., OI208 from Z:\2p_data\OI208\250227\...)
        path_parts = mouse_path.parts
        mouse_id = None
        for part in path_parts:
            if part.startswith('OI') and len(part) <= 6:  # Look for mouse ID pattern like OI208
                mouse_id = part
                break
        
        if mouse_id is None:
            mouse_id = mouse_path.name  # Fallback to directory name
            print(f"Warning: Could not extract mouse ID from path {mouse_dir}, using {mouse_id}")
        
        # Look for Analysis files in the directory
        analysis_files = list(mouse_path.glob("non_15hz_to_15hz_analysis_*.pkl"))
        
        if not analysis_files:
            print(f"No 15Hz Analysis files found for mouse {mouse_id} in {mouse_dir}")
            continue
            
        for analysis_file in analysis_files:
            try:
                with open(analysis_file, 'rb') as f:
                    data = pickle.load(f)
                
                # Extract day from filename
                day = analysis_file.stem.split('_')[-1]
                
                # Create row for this mouse/day
                row_data = {
                    'mouse': mouse_id,
                    'day': day,
                    'total_non_15hz_responsive': data['summary']['total_non_15hz_responsive'],
                    'total_15hz_responsive': data['summary']['total_15hz_responsive'],
                    'total_overlap': data['summary']['total_overlap'],
                    'percentage_non_15hz_also_15hz': data['summary']['percentage_non_15hz_also_15hz']
                }
                
                # Add detailed breakdown by frequency
                for freq, freq_data in data['detailed_breakdown'].items():
                    row_data[f'freq_{freq}_responsive'] = freq_data['responsive_to_freq']
                    row_data[f'freq_{freq}_also_15hz'] = freq_data['also_responsive_to_15hz']
                    row_data[f'freq_{freq}_percentage'] = freq_data['percentage']
                
                all_data.append(row_data)
                print(f"Loaded data for mouse {mouse_id}, day {day}")
                
            except Exception as e:
                print(f"Error loading {analysis_file}: {e}")
                continue
    
    if not all_data:
        raise ValueError("No data could be loaded from any mouse directory")
    
    df = pd.DataFrame(all_data)
    
    # Debug information
    print(f"\nDataFrame shape: {df.shape}")
    print(f"Unique mice: {df['mouse'].unique()}")
    print(f"Available columns: {[col for col in df.columns if 'freq_' in col]}")
    
    return df


def plot_overall_percentage_across_mice(df: pd.DataFrame, save_dir: str) -> None:
    """
    Plot the overall percentage of non-15Hz responsive cells that are also responsive to 15Hz across mice.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Individual mouse data points and mean
    mice = df['mouse'].unique()
    percentages = df.groupby('mouse')['percentage_non_15hz_also_15hz'].mean()
    
    # Plot 1: Bar plot with individual points
    mean_pct = percentages.mean()
    sem_pct = percentages.sem()
    
    # Handle case where SEM is NaN (only one mouse)
    if pd.isna(sem_pct) or len(percentages) == 1:
        sem_pct = 0
        yerr_val = None
    else:
        yerr_val = [sem_pct]
    
    ax1.bar(['Non-15Hz → 15Hz'], [mean_pct], yerr=yerr_val, 
            color='lightblue', edgecolor='black', alpha=0.7, capsize=5)
    
    # Add individual mouse points
    x_pos = np.zeros(len(percentages))
    ax1.scatter(x_pos, percentages.values, color='red', s=60, alpha=0.8, zorder=3)
    
    # Add mouse labels
    for i, (mouse, pct) in enumerate(percentages.items()):
        ax1.annotate(mouse, (x_pos[i], pct), xytext=(5, 5), 
                    textcoords='offset points', fontsize=8, alpha=0.7)
    
    ax1.set_ylabel('Percentage (%)')
    ax1.set_title('Percentage of Non-15Hz Responsive Cells\nAlso Responsive to 15Hz')
    ax1.set_ylim(0, max(percentages.max() * 1.1, 100))
    ax1.grid(True, alpha=0.3)
    
    # Add statistics
    n_mice = len(percentages)
    ax1.text(0.02, 0.98, f'n = {n_mice} mice\nMean ± SEM: {mean_pct:.1f} ± {sem_pct:.1f}%', 
             transform=ax1.transAxes, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot 2: Distribution of percentages
    ax2.hist(percentages.values, bins=min(10, len(percentages)), 
             color='lightblue', edgecolor='black', alpha=0.7)
    ax2.axvline(mean_pct, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_pct:.1f}%')
    ax2.set_xlabel('Percentage (%)')
    ax2.set_ylabel('Number of Mice')
    ax2.set_title('Distribution of Percentages Across Mice')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'overall_15hz_overlap_percentage.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, 'overall_15hz_overlap_percentage.pdf'), bbox_inches='tight')
    plt.show()


def plot_frequency_breakdown_across_mice(df: pd.DataFrame, save_dir: str) -> None:
    """
    Plot the percentage breakdown by individual frequencies across mice.
    """
    # Extract frequency columns
    freq_cols = [col for col in df.columns if col.startswith('freq_') and col.endswith('_percentage')]
    frequencies = [float(col.split('_')[1]) for col in freq_cols]
    frequencies.sort()
    
    # Prepare data for plotting
    freq_data = {}
    available_frequencies = []
    
    for freq in frequencies:
        if freq == 1.0 or freq == 3.0 or freq == 5.0 or freq == 10.0:
            freq = int(freq)  # Convert to int for cleaner labels
        col_name = f'freq_{freq}_percentage'
        if col_name in df.columns:
            freq_data[freq] = df.groupby('mouse')[col_name].mean()
            available_frequencies.append(freq)
        else:
            print(f"Warning: Column {col_name} not found in data")
    
    if not available_frequencies:
        print("Error: No frequency data available for plotting")
        return
    
    # Use only available frequencies for plotting
    frequencies = available_frequencies
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Bar plot with error bars
    freq_means = [freq_data[freq].mean() for freq in frequencies]
    freq_sems = [freq_data[freq].sem() for freq in frequencies]
    
    # Handle NaN SEMs (when only one mouse has data for a frequency)
    freq_sems_clean = [sem if not pd.isna(sem) else 0 for sem in freq_sems]
    
    bars = ax1.bar(range(len(frequencies)), freq_means, yerr=freq_sems_clean,
                   color='lightcoral', edgecolor='black', alpha=0.7, capsize=5)
    
    # Add individual mouse lines
    mice = df['mouse'].unique()
    colors = plt.cm.Set3(np.linspace(0, 1, len(mice)))
    
    for i, mouse in enumerate(mice):
        mouse_values = [freq_data[freq][mouse] if mouse in freq_data[freq].index else np.nan 
                       for freq in frequencies]
        ax1.plot(range(len(frequencies)), mouse_values, 'o-', 
                color=colors[i], alpha=0.6, linewidth=1, markersize=4, label=mouse)
    
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Percentage Also Responsive to 15Hz (%)')
    ax1.set_title('Frequency-Specific Overlap with 15Hz Responses')
    ax1.set_xticks(range(len(frequencies)))
    ax1.set_xticklabels([str(f) for f in frequencies])
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Heatmap of individual mice
    heatmap_data = []
    mouse_labels = []
    
    for mouse in mice:
        mouse_row = []
        for freq in frequencies:
            if mouse in freq_data[freq].index:
                mouse_row.append(freq_data[freq][mouse])
            else:
                mouse_row.append(np.nan)
        heatmap_data.append(mouse_row)
        mouse_labels.append(mouse)
    
    heatmap_data = np.array(heatmap_data)
    
    im = ax2.imshow(heatmap_data, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=100)
    ax2.set_xticks(range(len(frequencies)))
    ax2.set_xticklabels([str(f) for f in frequencies])
    ax2.set_yticks(range(len(mice)))
    ax2.set_yticklabels(mouse_labels)
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Mouse')
    ax2.set_title('Heatmap: Percentage Also Responsive to 15Hz by Mouse and Frequency')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label('Percentage (%)')
    
    # Add text annotations
    for i in range(len(mice)):
        for j in range(len(frequencies)):
            if not np.isnan(heatmap_data[i, j]):
                text = ax2.text(j, i, f'{heatmap_data[i, j]:.0f}', 
                               ha="center", va="center", color="black", fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'frequency_breakdown_15hz_overlap.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, 'frequency_breakdown_15hz_overlap.pdf'), bbox_inches='tight')
    plt.show()


def plot_cell_counts_across_mice(df: pd.DataFrame, save_dir: str) -> None:
    """
    Plot the absolute cell counts across mice.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    # Metrics to plot
    metrics = [
        ('total_non_15hz_responsive', 'Non-15Hz Responsive Cells'),
        ('total_15hz_responsive', '15Hz Responsive Cells'),
        ('total_overlap', 'Cells Responsive to Both'),
        ('percentage_non_15hz_also_15hz', 'Percentage Overlap (%)')
    ]
    
    mice = df['mouse'].unique()
    
    for idx, (metric, title) in enumerate(metrics):
        ax = axes[idx]
        
        # Get data for this metric
        mouse_data = df.groupby('mouse')[metric].mean()
        
        # Bar plot
        bars = ax.bar(range(len(mice)), mouse_data.values, 
                     color='lightgreen', edgecolor='black', alpha=0.7)
        
        # Add value labels on bars
        for i, (mouse, value) in enumerate(mouse_data.items()):
            ax.text(i, value + max(mouse_data.values) * 0.01, f'{value:.1f}', 
                   ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Mouse')
        ax.set_ylabel('Count' if 'Percentage' not in title else 'Percentage (%)')
        ax.set_title(title)
        ax.set_xticks(range(len(mice)))
        ax.set_xticklabels(mice, rotation=45)
        ax.grid(True, alpha=0.3)
        
        # Add summary statistics
        mean_val = mouse_data.mean()
        sem_val = mouse_data.sem()
        ax.text(0.02, 0.98, f'Mean ± SEM:\n{mean_val:.1f} ± {sem_val:.1f}', 
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'cell_counts_across_mice.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, 'cell_counts_across_mice.pdf'), bbox_inches='tight')
    plt.show()


def create_summary_statistics(df: pd.DataFrame, save_dir: str) -> None:
    """
    Create and save summary statistics table.
    """
    # Overall statistics
    overall_stats = df.groupby('mouse').agg({
        'total_non_15hz_responsive': 'mean',
        'total_15hz_responsive': 'mean', 
        'total_overlap': 'mean',
        'percentage_non_15hz_also_15hz': 'mean'
    }).round(2)
    
    # Add summary row
    summary_row = pd.DataFrame({
        'total_non_15hz_responsive': [overall_stats['total_non_15hz_responsive'].mean()],
        'total_15hz_responsive': [overall_stats['total_15hz_responsive'].mean()],
        'total_overlap': [overall_stats['total_overlap'].mean()],
        'percentage_non_15hz_also_15hz': [overall_stats['percentage_non_15hz_also_15hz'].mean()]
    }, index=['MEAN_ALL_MICE'])
    
    overall_stats = pd.concat([overall_stats, summary_row])
    
    # Frequency-specific statistics
    freq_cols = [col for col in df.columns if col.startswith('freq_') and col.endswith('_percentage')]
    frequencies = sorted([float(col.split('_')[1]) for col in freq_cols])
    
    freq_stats = pd.DataFrame()
    for freq in frequencies:
        col_name = f'freq_{freq}_percentage'
        if col_name in df.columns:
            freq_stats[f'{freq}Hz'] = df.groupby('mouse')[col_name].mean()
    
    # Add frequency summary
    freq_summary = pd.DataFrame(freq_stats.mean()).T
    freq_summary.index = ['MEAN_ALL_MICE']
    freq_stats = pd.concat([freq_stats, freq_summary])
    
    # Save to files
    overall_stats.to_csv(os.path.join(save_dir, 'overall_15hz_statistics.csv'))
    freq_stats.to_csv(os.path.join(save_dir, 'frequency_specific_15hz_statistics.csv'))
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY STATISTICS - 15Hz OVERLAP ANALYSIS")
    print("="*60)
    print(f"Number of mice: {len(df['mouse'].unique())}")
    print(f"Total sessions: {len(df)}")
    print("\nOverall Statistics (Mean ± SEM across mice):")
    
    for col in ['total_non_15hz_responsive', 'total_15hz_responsive', 'total_overlap', 'percentage_non_15hz_also_15hz']:
        mouse_means = df.groupby('mouse')[col].mean()
        mean_val = mouse_means.mean()
        sem_val = mouse_means.sem()
        # Handle NaN SEM (when only one mouse)
        if pd.isna(sem_val):
            sem_val = 0.0
        print(f"  {col}: {mean_val:.2f} ± {sem_val:.2f}")
    
    print("\nFrequency-specific overlap percentages (Mean ± SEM across mice):")
    for freq in frequencies:
        col_name = f'freq_{freq}_percentage'
        if col_name in df.columns:
            mouse_means = df.groupby('mouse')[col_name].mean()
            mean_val = mouse_means.mean()
            sem_val = mouse_means.sem()
            # Handle NaN SEM (when only one mouse)
            if pd.isna(sem_val):
                sem_val = 0.0
            print(f"  {freq} Hz: {mean_val:.2f} ± {sem_val:.2f}%")


def perform_statistical_tests(df: pd.DataFrame, save_dir: str) -> None:
    """
    Perform statistical tests on the data.
    """
    results = []
    
    # Test if overall percentage is significantly different from chance (50%)
    mouse_percentages = df.groupby('mouse')['percentage_non_15hz_also_15hz'].mean()
    
    # Check if we have enough data for statistical tests
    if len(mouse_percentages) < 2:
        print(f"Warning: Only {len(mouse_percentages)} mouse(es) available. Statistical tests require at least 2 mice.")
        results.append({
            'test': 'Overall percentage vs 50%',
            'statistic': 'N/A',
            'p_value': 'N/A',
            'significant': 'N/A',
            'interpretation': f"Insufficient data for statistical testing (n={len(mouse_percentages)} mice). Mean percentage: {mouse_percentages.mean():.1f}%"
        })
    else:
        # One-sample t-test against 50%
        try:
            t_stat, p_val = ttest_1samp(mouse_percentages.values, 50)
            results.append({
                'test': 'Overall percentage vs 50% (t-test)',
                'statistic': t_stat,
                'p_value': p_val,
                'significant': p_val < 0.05,
                'interpretation': f"Mean percentage ({mouse_percentages.mean():.1f}%) is {'significantly' if p_val < 0.05 else 'not significantly'} different from 50%"
            })
        except Exception as e:
            results.append({
                'test': 'Overall percentage vs 50% (t-test)',
                'statistic': 'Error',
                'p_value': 'Error',
                'significant': 'Error',
                'interpretation': f"Error in t-test: {str(e)}"
            })
        
        # Wilcoxon signed-rank test against 50% (non-parametric)
        try:
            w_stat, w_p_val = wilcoxon(mouse_percentages.values - 50)
            results.append({
                'test': 'Overall percentage vs 50% (Wilcoxon)',
                'statistic': w_stat,
                'p_value': w_p_val,
                'significant': w_p_val < 0.05,
                'interpretation': f"Mean percentage ({mouse_percentages.mean():.1f}%) is {'significantly' if w_p_val < 0.05 else 'not significantly'} different from 50% (non-parametric)"
            })
        except Exception as e:
            results.append({
                'test': 'Overall percentage vs 50% (Wilcoxon)',
                'statistic': 'Error',
                'p_value': 'Error',
                'significant': 'Error',
                'interpretation': f"Error in Wilcoxon test: {str(e)}"
            })
    
    # Test frequency-specific percentages
    freq_cols = [col for col in df.columns if col.startswith('freq_') and col.endswith('_percentage')]
    frequencies = sorted([float(col.split('_')[1]) for col in freq_cols])
    
    for freq in frequencies:
        col_name = f'freq_{freq}_percentage'
        if col_name in df.columns:
            freq_percentages = df.groupby('mouse')[col_name].mean().dropna()
            if len(freq_percentages) >= 2:  # Need at least 2 mice for meaningful test
                try:
                    t_stat, p_val = ttest_1samp(freq_percentages.values, 50)
                    results.append({
                        'test': f'{freq} Hz vs 50% (t-test)',
                        'statistic': t_stat,
                        'p_value': p_val,
                        'significant': p_val < 0.05,
                        'interpretation': f"{freq} Hz mean ({freq_percentages.mean():.1f}%) is {'significantly' if p_val < 0.05 else 'not significantly'} different from 50%"
                    })
                except Exception as e:
                    results.append({
                        'test': f'{freq} Hz vs 50% (t-test)',
                        'statistic': 'Error',
                        'p_value': 'Error',
                        'significant': 'Error',
                        'interpretation': f"Error in {freq} Hz t-test: {str(e)}"
                    })
            else:
                results.append({
                    'test': f'{freq} Hz vs 50% (t-test)',
                    'statistic': 'N/A',
                    'p_value': 'N/A',
                    'significant': 'N/A',
                    'interpretation': f"Insufficient data for {freq} Hz testing (n={len(freq_percentages)} mice)"
                })
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(save_dir, 'statistical_test_results.csv'), index=False)
    
    # Print results
    print("\n" + "="*60)
    print("STATISTICAL TEST RESULTS")
    print("="*60)
    for result in results:
        print(f"\n{result['test']}:")
        print(f"  Statistic: {result['statistic']:.4f}")
        print(f"  P-value: {result['p_value']:.4f}")
        print(f"  Significant: {result['significant']}")
        print(f"  {result['interpretation']}")


def main():
    """
    Main function to run the 15Hz Analysis across mice.
    """
    # Configuration - UPDATE THESE PATHS FOR YOUR DATA
    mouse_dirs = [
        r'Z:\2p_data\OI208\250227\ML\data\processed_data\auROC_freq_psycho_15hz',
        r'Z:\2p_data\OI151\240827\ML\data\processed_data\auROC_freq_psycho_15hz',
        r'Z:\2p_data\OI152\241003\ML\data\processed_data\auROC_freq_psycho_15hz',
        r'Z:\2p_data\OI124\240610\ML\data\processed_data\auROC_freq_psycho_15hz',
        r'Z:\2p_data\OI116\240529\ML\data\processed_data\auROC_freq_psycho_15hz',
        # Add more mouse directories here
    ]
    
    output_dir = r'Z:\Stavsht\Omer_OXTR\15hz_analysis_across_mice'
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading 15Hz Analysis data from multiple mice...")
    
    try:
        # Load data
        df = load_15hz_analysis_data(mouse_dirs)
        print(f"Successfully loaded data from {len(df['mouse'].unique())} mice")
        
        # Create plots
        print("\nCreating plots...")
        plot_overall_percentage_across_mice(df, output_dir)
        plot_frequency_breakdown_across_mice(df, output_dir)
        plot_cell_counts_across_mice(df, output_dir)
        
        # Create summary statistics
        print("\nGenerating summary statistics...")
        create_summary_statistics(df, output_dir)
        
        # Perform statistical tests
        print("\nPerforming statistical tests...")
        perform_statistical_tests(df, output_dir)
        
        print(f"\nAnalysis complete! Results saved to: {output_dir}")
        
    except Exception as e:
        print(f"Error during Analysis: {e}")
        raise


if __name__ == '__main__':
    main()
