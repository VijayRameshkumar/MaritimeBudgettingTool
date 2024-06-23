import datetime
import numpy as np

def calculate_yearly_growth_rate(dates, expenses):
    if len(dates) != len(expenses):
        raise ValueError("Lengths of dates and expenses must be the same.")

    yearly_data = {}
    for date, expense in zip(dates, expenses):
        year = date.year
        if year not in yearly_data:
            yearly_data[year] = []
        yearly_data[year].append(expense)

    yearly_growth_rate = {}
    for year, expenses_list in yearly_data.items():
        start_expense = expenses_list[0]
        end_expense = expenses_list[-1]
        
        growth_rate = ((end_expense - start_expense) / start_expense) * 100
        
        # Check for division by zero or start_expense being -inf
        if np.isinf(growth_rate):
            growth_rate = 0
        
        yearly_growth_rate[year] = growth_rate
    
    return yearly_growth_rate

