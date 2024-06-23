import numpy as np
from scipy.stats import ttest_1samp, norm, mode
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


def get_optimal_mean(data):
            
    # Remove outliers (adjust the threshold as needed)
    data_no_outliers = data[~((data - np.mean(data)) > 3.5 * np.std(data))]
    
    mean_value = np.mean(data_no_outliers)
    median_value = np.median(data_no_outliers)

    # Calculate mode, mean, and standard deviation
    mode_value = mode(data_no_outliers).mode[0]
    
    max_flag = median_value
    if mean_value > median_value:
        optimal_expected_mean = median_value
        max_flag = mean_value
    else:
        optimal_expected_mean = mean_value
        max_flag = median_value

    # Initialize variables for optimization
    p_value = 1  # Initialize p-value to a value greater than 0.05

    # If the effect size is more than 0.5, find the optimal expected mean where the effect size is 0.4
    if cohen_d(data_no_outliers, [optimal_expected_mean] * len(data_no_outliers)) > 0.5:
        # Iterate through expected mean values within the range from median to mean
        for expected_mean_candidate in np.linspace(optimal_expected_mean, max_flag, 800):
            _, p_value_candidate = ttest_1samp(data_no_outliers, popmean=expected_mean_candidate)
            p_value_candidate = round(p_value_candidate, 2)

            # If the current p-value is greater than 0.05 and the effect size is less than 0.5,
            # update the optimal expected mean and break the loop
            if round(p_value_candidate, 2) > 0.05 and cohen_d(data_no_outliers, [expected_mean_candidate] * len(data_no_outliers)) < 0.5:
                optimal_expected_mean = expected_mean_candidate
                break

    return optimal_expected_mean