import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, ttest_1samp, mode
import warnings
warnings.filterwarnings('ignore')
import streamlit as st

#################### Optimization Function ##############################

def cohen_d(group1, group2):
    """
    Calculate Cohen's d for two groups.
    """
    mean_diff = np.mean(group1) - np.mean(group2)
    pooled_std = np.sqrt((np.std(group1)**2 + np.std(group2)**2) / 2)
    return mean_diff / pooled_std

def optimize_mean(data_no_outliers, optimal_expected_mean, max_flag):
    """
    Optimize the expected mean considering both p-value and effect size.
    """
    min_combined_score = float('inf')  # Initialize the minimum combined score
    min_p_value = 1  # Initialize the minimum p-value
    min_effect_size = 0  # Initialize the minimum effect size

    # Iterate through expected mean values within the range from median to mean
    for expected_mean_candidate in np.linspace(optimal_expected_mean, max_flag, 10000):
        _, p_value_candidate = ttest_1samp(data_no_outliers, popmean=expected_mean_candidate)
        p_value_candidate = max(p_value_candidate, 0.06)  # Ensure p-value doesn't go below 0.05

        # Calculate Cohen's d as the effect size
        effect_size_candidate = cohen_d(data_no_outliers, [expected_mean_candidate] * len(data_no_outliers))

        # Define weights for the optimization criteria
        p_value_weight = 0.5  # Adjust the weights as needed
        effect_size_weight = 0.5

        # Calculate the combined score as the weighted sum of the p-value and effect size
        combined_score = (
            p_value_weight * p_value_candidate + 
            effect_size_weight * abs(effect_size_candidate)
        )

        # Update the optimal values if the combined score is better
        if combined_score < min_combined_score:
            min_combined_score = combined_score
            optimal_expected_mean = expected_mean_candidate
            min_p_value = p_value_candidate
            min_effect_size = effect_size_candidate

    return optimal_expected_mean, min_p_value, min_effect_size

#################### Main ################################################

def get_optimal_mean(exl):
    
    # Create subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 5 * 2), facecolor='#262c2e')
    # Flatten the axs array to handle different numbers of rows and columns
    axs_flat = axs.flatten()
    axs_flat[-1].set_visible(False)
    
    unique_years = exl.YEAR.unique()
    exl = exl.groupby(['YEAR', 'PERIOD', 'VESSEL NAME'])['Expense'].sum().reset_index()

    for j, yr in enumerate(unique_years):
        
        # st.dataframe(exl)
        data = exl[exl['YEAR'] == yr]
        
        data = data[data['Expense'] > 0]['Expense']
        
        # Remove outliers (adjust the threshold as needed)
        data_no_outliers = data[~((data - np.mean(data)) > 3.5 * np.std(data))]
        
        mean_value = np.mean(data_no_outliers)
        median_value = np.median(data_no_outliers)
        std_dev = np.std(data_no_outliers)
        
        mode_value = mode(data_no_outliers).mode[0]

        x = np.linspace(mean_value - 3 * std_dev, mean_value + 3 * std_dev, 1000)
        y = norm.pdf(x, mean_value, std_dev)

        # Skip optimization if median and mode are the same or mode is greater than mean/median
        # if median_value == mode_value:
        #     optimal_expected_mean = mode_value
        #     min_p_value = 1
        #     min_effect_size = 0
        # else:
        optimal_expected_mean = median_value if mean_value > median_value else mean_value
        max_flag = mean_value if mean_value > median_value else median_value
        
        # max_flag = max_flag + (max_flag * 0.20)
        
        # Optimize the expected mean
        optimal_expected_mean, min_p_value, min_effect_size = optimize_mean(data_no_outliers, optimal_expected_mean, max_flag)

        # Use the flattened axs to set the current subplot
        plt.sca(axs_flat[j])
        axs_flat[j].set_facecolor('#262c2e')
        

        # Highlight the region with a significant p-value in a different color
        significant_color = 'red' if min_p_value < 0.05 else 'green'
        plt.fill_between(x, y, where=[(val >= mean_value - 3 * std_dev) and (val <= mean_value + 3 * std_dev) for val in x],
                        color=significant_color, alpha=0.3, label='Significant Region')

        # Plot the bell curve
        plt.plot(x, y, label='Bell Curve')

        # Plot the histogram
        plt.hist(data_no_outliers, bins=30, density=True, alpha=0.6, color='g', label='Histogram')

        # Add mean, median, mode, and std annotations
        plt.axvline(mean_value, color='blue', linestyle='dashed', linewidth=2, label=f'Mean: ${mean_value:.2f}')
        plt.axvline(median_value, color='purple', linestyle='dashed', linewidth=2, label=f'Median: ${median_value:.2f}')
        plt.axvline(mode_value, color='orange', linestyle='dashed', linewidth=2, label=f'Mode: ${mode_value:.2f}')
        plt.annotate(f'Std Dev: {std_dev:.2f}', xy=(mean_value - 3 * std_dev, 0.01), fontsize=10, color='orange')

        # Add a vertical line for the optimal_expected_mean
        plt.axvline(optimal_expected_mean, color='red', linestyle='dashed', linewidth=1, label=f'Optimal Mean: ${optimal_expected_mean:.2f}')

        # Add labels and legend
        plt.xlabel('Values', color='white')
        plt.ylabel('Probability Density', color='white')
        plt.title(f'{yr} -  (p-value: {min_p_value:.4f}, - Effect Size : {min_effect_size:.2f} , \nOptimal Budget : {optimal_expected_mean:.2f})', color='white')
        plt.legend()
        # plt.set_facecolor('#e6e6e6')
        
        axs_flat[j] = plt.gca()

        # Remove the box around the chart
        axs_flat[j].spines['top'].set_visible(False)
        axs_flat[j].spines['right'].set_visible(False)
        axs_flat[j].spines['bottom'].set_visible(False)
        axs_flat[j].spines['left'].set_visible(False)
        axs_flat[j].tick_params(axis='x', colors='green')  # Adjust color of x-values
        axs_flat[j].tick_params(axis='y', colors='orange')  # Adjust color of y-values
            
        # Remove ticks
        axs_flat[j].tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)

        # Set axis format to plain to prevent exponential values
        plt.ticklabel_format(style='plain', axis='both')

        ## Adjust layout to prevent overlapping
        plt.tight_layout()

    return plt