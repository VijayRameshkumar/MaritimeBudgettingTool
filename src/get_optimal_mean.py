import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, ttest_1samp, mode
import warnings
warnings.filterwarnings('ignore')

#################### Main ################################################

def cohen_d(group1, group2):
    """
    Calculate Cohen's d for two groups.
    """
    mean_diff = np.mean(group1) - np.mean(group2)
    pooled_std = np.sqrt((np.std(group1)**2 + np.std(group2)**2) / 2)
    return mean_diff / pooled_std


def get_optimal_mean(exl):

    vessel_optimal_mean = dict()

    # Get the number of unique vessels
    unique_vessels = exl['Vessels'].unique()
    num_vessels = len(unique_vessels)

    # Set the number of columns for the subplot
    num_columns = 2

    for i, ship in enumerate(unique_vessels):
        ship_data = exl[exl.Vessels == ship]
        unique_cats = ship_data.Categories.unique()
        num_cats = len(unique_cats)
        
        # Calculate the number of rows needed for subplots
        num_rows = int(np.ceil(num_cats / num_columns))

        # Create subplots
        fig, axs = plt.subplots(num_rows, num_columns, figsize=(15, 5 * num_rows))
        # Flatten the axs array to handle different numbers of rows and columns
        axs_flat = axs.flatten()
        
        for i, cat in enumerate(unique_cats):
            
            data = ship_data[ship_data['Categories'] == cat]['Expense']
            
            # Remove outliers (adjust the threshold as needed)
            data_no_outliers = data[~((data - np.mean(data)) > 3.5 * np.std(data))]
            
            mean_value = np.mean(data_no_outliers)
            median_value = np.median(data_no_outliers)
            std_dev = np.std(data_no_outliers)

            # Calculate mode, mean, and standard deviation
            mode_value = mode(data_no_outliers).mode[0]

            # Create a range of values for x-axis
            x = np.linspace(mean_value - 3 * std_dev, mean_value + 3 * std_dev, 1000)

            # Generate the bell curve using the probability density function (pdf)
            y = norm.pdf(x, mean_value, std_dev)
            
            max_flag = median_value
            if mean_value > median_value:
                optimal_expected_mean = median_value
                max_flag = mean_value
            else:
                optimal_expected_mean = mean_value
                max_flag = median_value

            # # Initialize variables for optimization
            # optimal_expected_mean = median_value
            p_value = 1  # Initialize p-value to a value greater than 0.05

            # Iterate through expected mean values within the range from median to mean
            for expected_mean_candidate in np.linspace(optimal_expected_mean, max_flag, 800):
                _, p_value_candidate = ttest_1samp(data_no_outliers, popmean=expected_mean_candidate)
                p_value_candidate = round(p_value_candidate, 2)

                # If the current p-value is greater than 0.05, update the optimal expected mean and break the loop
                if round(p_value_candidate, 2) > 0.05:
                    optimal_expected_mean = expected_mean_candidate
                
            # Calculate Cohen's d as the effect size
            effect_size = cohen_d(data_no_outliers, [optimal_expected_mean] * len(data_no_outliers))

            vessel_optimal_mean[ship + "_" + cat] = f'${str(round(optimal_expected_mean, 2))}'

            # Use the flattened axs to set the current subplot
            plt.sca(axs_flat[i])

            # Highlight the region with a significant p-value in a different color
            significant_color = 'red' if p_value_candidate < 0.05 else 'green'
            plt.fill_between(x, y, where=[(val >= mean_value - 3 * std_dev) and (val <= mean_value + 3 * std_dev) for val in x],
                            color=significant_color, alpha=0.3, label='Significant Region')

            # Plot the bell curve
            plt.plot(x, y, label='Bell Curve')

            # Plot the histogram
            plt.hist(data_no_outliers, bins=30, density=True, alpha=0.6, color='g', label='Histogram')

            # Add mean, median, mode, and std annotations
            plt.axvline(mean_value, color='blue', linestyle='dashed', linewidth=2, label=f'Mean: ${mean_value:.2f}')
            plt.axvline(median_value, color='purple', linestyle='dashed', linewidth=2, label=f'Median: ${median_value:.2f}')
            # plt.axvline(effect_size, color='purple', linestyle='dashed', linewidth=2, label=f'Effect Size: ${effect_size:.2f}')
            plt.axvline(mode_value, color='orange', linestyle='dashed', linewidth=2, label=f'Mode: ${mode_value:.2f}')
            plt.annotate(f'Std Dev: {std_dev:.2f}', xy=(mean_value - 3 * std_dev, 0.01), fontsize=10, color='orange')

            # Add a vertical line for the optimal_expected_mean
            plt.axvline(optimal_expected_mean, color='red', linestyle='dashed', linewidth=2, label=f'Optimal Mean: ${optimal_expected_mean:.2f}')

            # Add labels and legend
            plt.xlabel('Values')
            plt.ylabel('Probability Density')
            plt.title(f'{ship}\n{cat} -  (p-value: {p_value_candidate:.4f}, - Effect Size : {effect_size:.2f} , Optimal Budget : {optimal_expected_mean:.2f})')
            plt.legend()

        # Adjust layout to prevent overlapping
        plt.tight_layout()

        plt.savefig(f'output/{ship}.png')

        # Show the plot
        plt.show()
        plt.close()
    return vessel_optimal_mean



