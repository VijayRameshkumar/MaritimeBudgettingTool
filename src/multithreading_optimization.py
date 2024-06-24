import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp, mode, norm, yeojohnson
from concurrent.futures import ThreadPoolExecutor
import streamlit as st
# import multiprocessing as mp
# from multiprocessing import Pool

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
    min_combined_score = float('inf') # Initialize the minimum combined score
    min_p_value = 1  # Initialize the minimum p-value
    min_effect_size = 0  # Initialize the minimum effect size

    # Iterate through expected mean values within the range from median to mean
    for expected_mean_candidate in np.linspace(optimal_expected_mean, max_flag, 1000):
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

def calculate_combined_score(p_value, effect_size):
    """
    Calculate the combined score as the weighted sum of the p-value and effect size.
    """
    p_value_weight = 0.5
    effect_size_weight = 0.5

    combined_score = p_value_weight * p_value + effect_size_weight * abs(effect_size)
    return combined_score

@st.cache_data
def get_cat_optimal_mean(exl):
    """
    Calculate the optimal mean for each category in parallel.
    """
    exl = exl.groupby(['YEAR', 'PERIOD', 'VESSEL NAME', 'CATEGORIES'])['Expense'].sum().reset_index()
    exl = exl.loc[exl['Expense'] > 0]
    exl = exl.groupby(['YEAR', 'CATEGORIES'])['Expense'].median().reset_index()
    cats = exl['CATEGORIES'].unique()
    optimal_means = dict()

    # Function to optimize mean for a single category
    def optimize_category(cat):
        data = exl[exl['CATEGORIES'] == cat]
        # st.dataframe(data)
        if cat == 'Administrative Expenses':
            data = data['Expense']
        else:
            data = data[data['Expense'] != 0]['Expense']
        
        transformed_data = data[~((data - np.mean(data)) > 3.5 * np.std(data))]
        
        # Apply Yeo-Johnson transformation
        # transformed_data, lambda_value = yeojohnson(data_no_outliers)

        mean_value = np.mean(transformed_data)
        median_value = np.quantile(transformed_data, 0.75)
        std_dev = np.std(transformed_data)
        
        #mode_value = mode(transformed_data).mode[0]

        x = np.linspace(mean_value - 3 * std_dev, mean_value + 3 * std_dev, 1000)
        y = norm.pdf(x, mean_value, std_dev)

        optimal_expected_mean = median_value if mean_value > median_value else mean_value
        max_flag = mean_value if mean_value > median_value else median_value

        # Optimize the expected mean
        optimal_expected_mean, min_p_value, min_effect_size = optimize_mean(transformed_data, optimal_expected_mean,
                                                                            max_flag)

        # Reverse Yeo-Johnson transformation to obtain optimal budget in original scale
        # optimal_budget = (optimal_expected_mean * std_dev) + mean_value
        
        # Store optimal mean in dictionary
        optimal_means[cat] = optimal_expected_mean
        # optimal_means[cat+"_"+"median"] = median_value
        
        return optimal_means
    
    for cat in cats:
        optimize_category(cat)
        
    df = pd.DataFrame.from_dict(optimal_means, orient='index', columns=['Stats Model - Optimal Budget'])
    df.index.name = 'CATEGORIES'
    df = df.reset_index()

    return df

@st.cache_data
def get_subcat_optimal_mean(exl):
    """
    Calculate the optimal mean for each category without using parallel processing.
    """
    exl = exl.groupby(['YEAR', 'PERIOD', 'VESSEL NAME', 'CATEGORIES', 'ACCOUNT_CODE', 'SUB_CATEGORIES'])['Expense'].sum().reset_index()
    exl = exl.loc[exl['Expense'] != 0]
    exl = exl.groupby(['YEAR', 'CATEGORIES', 'ACCOUNT_CODE', 'SUB_CATEGORIES'])['Expense'].median().reset_index()
    sub_cats = exl['SUB_CATEGORIES'].unique()
    optimal_means = dict()

    # Function to optimize mean for a single category
    def optimize_category(cat, ac_code, subcat):
        data = exl[(exl['CATEGORIES'] == cat) & (exl['ACCOUNT_CODE'] == ac_code) & (exl['SUB_CATEGORIES'] == subcat)]
        data = data['Expense']
        transformed_data = data[~((data - np.mean(data)) > 3.5 * np.std(data))]

        mean_value = np.mean(transformed_data)
        median_value = np.quantile(transformed_data, 0.75)
        std_dev = np.std(transformed_data)
        
        # mode_value = mode(transformed_data).mode[0]

        x = np.linspace(mean_value - 3 * std_dev, mean_value + 3 * std_dev, 1000)
        y = norm.pdf(x, mean_value, std_dev)

        optimal_expected_mean = median_value if mean_value > median_value else mean_value
        max_flag = mean_value if mean_value > median_value else median_value

        # Optimize the expected mean
        optimal_expected_mean, min_p_value, min_effect_size = optimize_mean(transformed_data, optimal_expected_mean, max_flag)

        # Store optimal mean in dictionary
        optimal_means["('{}', '{}', '{}')".format(cat, ac_code, subcat)] = optimal_expected_mean
        return optimal_means
    
    # Iterate through categories and optimize each one
    for _, row in exl.iterrows():
        optimal_means.update(optimize_category(row['CATEGORIES'], row['ACCOUNT_CODE'], row['SUB_CATEGORIES']))
        
    df = pd.DataFrame.from_dict(optimal_means, orient='index', columns=['Stats Model - Optimal Budget'])
    df.index.name = 'SUB CATEGORIES'
    df = df.reset_index()
    return df

@st.cache_data
def calculate_geometric_mean(exl, level='CATEGORIES'):
    """
    Calculate the geometric mean for each category or subcategory.
    Args:
        exl (DataFrame): Input DataFrame containing expense data.
        level (str): Level at which to calculate the geometric mean ('CATEGORIES' or 'SUB_CATEGORIES').
    Returns:
        DataFrame: DataFrame containing the geometric mean for each category or subcategory.
    """
    # Group data by specified level and calculate geometric mean
    if level == 'CATEGORIES':
        # Calculate geometric mean for each category
        grouped_data = exl.groupby(['YEAR', 'CATEGORIES'])['Expense'].apply(lambda x: np.exp(np.mean(np.log(x + 1)))) - 1
    elif level == 'SUB_CATEGORIES':
        # Calculate geometric mean for each subcategory
        grouped_data = exl.groupby(['YEAR', 'CATEGORIES', 'ACCOUNT_CODE', 'SUB_CATEGORIES'])['Expense'].apply(lambda x: np.exp(np.mean(np.log(x + 1)))) - 1
    else:
        raise ValueError("Invalid level. Use 'CATEGORIES' or 'SUB_CATEGORIES'.")

    # Convert series to DataFrame
    df = grouped_data.reset_index(name='Geometric Mean')
    
    return df

@st.cache_data
def get_event_cat_optimal_mean(exl):
    """
    Calculate the optimal mean for each category in parallel.
    """
    # st.dataframe(exl)
    exl = exl.loc[exl['EXPENSE'] > 0]
    cats = exl['CATEGORIES'].unique()
    optimal_means = dict()

    # Function to optimize mean for a single category
    def optimize_category(cat):
        data = exl[exl['CATEGORIES'] == cat]
        if cat == 'Administrative Expenses':
            data = data['EXPENSE']
        else:
            data = data[data['EXPENSE'] != 0]['EXPENSE']
        
        transformed_data = data[~((data - np.mean(data)) > 3.5 * np.std(data))]
        
        # Apply Yeo-Johnson transformation
        # transformed_data, lambda_value = yeojohnson(data_no_outliers)

        mean_value = np.mean(transformed_data)
        median_value = np.quantile(transformed_data, 0.75)
        std_dev = np.std(transformed_data)
        
        # mode_value = mode(transformed_data).mode[0]

        x = np.linspace(mean_value - 3 * std_dev, mean_value + 3 * std_dev, 1000)
        y = norm.pdf(x, mean_value, std_dev)

        optimal_expected_mean = median_value if mean_value > median_value else mean_value
        max_flag = mean_value if mean_value > median_value else median_value

        # Optimize the expected mean
        optimal_expected_mean, min_p_value, min_effect_size = optimize_mean(transformed_data, optimal_expected_mean,
                                                                            max_flag)

        # Reverse Yeo-Johnson transformation to obtain optimal budget in original scale
        # optimal_budget = (optimal_expected_mean * std_dev) + mean_value
        
        # Store optimal mean in dictionary
        optimal_means[cat] = optimal_expected_mean
        # optimal_means[cat+"_"+"median"] = median_value
        
        return optimal_means
    
    for cat in cats:
        optimize_category(cat)
        
    df = pd.DataFrame.from_dict(optimal_means, orient='index', columns=['Stats Model - Optimal Budget'])
    df.index.name = 'CATEGORIES'
    df = df.reset_index()

    return df

@st.cache_data
def get_event_subcat_optimal_mean(exl):
    """
    Calculate the optimal mean for each category without using parallel processing.
    """
    # exl = exl.groupby(['YEAR', 'PERIOD', 'VESSEL NAME', 'CATEGORIES', 'ACCOUNT_CODE', 'SUB_CATEGORIES'])['Expense'].sum().reset_index()
    exl = exl.loc[exl['EXPENSE'] != 0]
    # exl = exl.groupby(['CATEGORIES', 'ACCOUNT_CODE', 'SUB_CATEGORIES'])['EXPENSE'].median().reset_index()
    # sub_cats = exl['SUB_CATEGORIES'].unique()
    sub_cats = exl.groupby(['CATEGORIES', 'ACCOUNT_CODE']).size().reset_index()
    optimal_means = dict()

    # Function to optimize mean for a single category
    def optimize_category(cat, ac_code):
        data = exl[(exl['CATEGORIES'] == cat) & (exl['ACCOUNT_CODE'] == ac_code)]
        data = data[data['EXPENSE'] > 0]
        data = data['EXPENSE']
        transformed_data = data[~((data - np.mean(data)) > 1 * np.std(data))]

        mean_value = np.mean(transformed_data)
        median_value = np.quantile(transformed_data, 0.75)
        std_dev = np.std(transformed_data)
        
        # mode_value = mode(transformed_data).mode[0]

        x = np.linspace(mean_value - 3 * std_dev, mean_value + 3 * std_dev, 1000)
        y = norm.pdf(x, mean_value, std_dev)

        optimal_expected_mean = median_value if mean_value > median_value else mean_value
        max_flag = mean_value if mean_value > median_value else median_value

        # Optimize the expected mean
        optimal_expected_mean, min_p_value, min_effect_size = optimize_mean(transformed_data, optimal_expected_mean, max_flag)
        
        # Store optimal mean in dictionary
        optimal_means["{};{}".format(cat, ac_code)] = optimal_expected_mean
        return optimal_means
    
    for _, row in sub_cats.iterrows():
        optimal_means.update(optimize_category(row['CATEGORIES'], row['ACCOUNT_CODE']))
        
    df = pd.DataFrame.from_dict(optimal_means, orient='index', columns=['Stats Model - Optimal Budget'])
    df.index.name = 'SUB CATEGORIES'
    df = df.reset_index()
    return df
