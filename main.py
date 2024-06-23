import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd
import json
import base64
import calendar
import datetime

from src import new_modified
from src import whole_year
from src.get_data import get_expense_data

from datetime import timedelta
from src.multithreading_optimization import get_cat_optimal_mean, get_subcat_optimal_mean, get_event_cat_optimal_mean, get_event_subcat_optimal_mean
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode, GridUpdateMode
from scipy.interpolate import CubicSpline
# from src.snowflake_connect import get_expense_data

# Set page layout
st.set_page_config(layout="wide")

### Read Data
vessel_particulars = pd.read_excel('VESSEL_PARTICULARS.xlsx')
# last_3_years = pd.read_csv('2021_2022_2023_expense.csv', skiprows=1).rename(columns={'AMOUNT_USD': 'Expense'})

@st.cache_data
def get_data():
    return get_expense_data()

last_3_years = get_data()

# Sidebar navigation
page = st.sidebar.radio("Navigation", ["1. Report Page", "2. Trend Analysis"])

with st.container():
    # col1, col2, col3, col4, col5 = st.columns([1, 2, 1, 2, 2])  # Added another column for the selected vessels dropdown
    col000, col00 = st.columns([1, 2])

    logo_path = "./Synergy_icon.jpg"
    col000.image(logo_path)
    col00.markdown('<h1 style="font-family: Times New Roman; font-size: 35px; text-align: left; margin-top: 1em; color: white;"><u>BUDGET ANALYSIS TOOL v1.0</u></h1>', unsafe_allow_html=True)
    # col00.markdown("<h1 style='text-align: left; margin-top: 1em; color: white; '><u>Budget Analysis Tool v1.0</u></h1>", unsafe_allow_html=True)
    # logo_path = "./Syn-DD.png"
    # col0.image(logo_path, width=250)
    st.markdown("---")

# st.write(last_3_years.info())

# last_3_years = get_expense_data()
# Merge last_3_years with vessel_particulars to include DATE column
merged_df = pd.merge(last_3_years, vessel_particulars[['COST_CENTER', 'VESSEL NAME', 'BUILD YEAR']], on='COST_CENTER', how='left')
# Add VESSEL AGE to merged_df
merged_df['VESSEL AGE'] = pd.to_datetime(merged_df['DATE']).dt.year - merged_df['BUILD YEAR']
last_3_years['VESSEL AGE'] = pd.to_datetime(merged_df['DATE']).dt.year - merged_df['BUILD YEAR']
last_3_years['CATEGORIES'] = last_3_years['CATEGORIES'].str.upper()
merged_df['CATEGORIES'] = merged_df['CATEGORIES'].str.upper()

# Filter DataFrame based on slicer values
@st.cache_data
def filter_dataframe(vessel_type, vessel_subtype, vessel_age_start, vessel_age_end):
    filtered_df = vessel_particulars[
        (vessel_particulars['VESSEL TYPE'] == vessel_type) &
        (vessel_particulars['VESSEL SUBTYPE'].isin(vessel_subtype))]
    
    selected_vessels = merged_df[merged_df['COST_CENTER'].isin(filtered_df['COST_CENTER'].unique())].reset_index(drop=True)
    selected_vessels = selected_vessels[selected_vessels['VESSEL AGE'].between(vessel_age_start, vessel_age_end)]
    selected_vessels = selected_vessels.groupby(['PERIOD', 'COST_CENTER', 'VESSEL NAME', 'CATEGORIES', 'ACCOUNT_CODE', 'SUB_CATEGORIES'])['Expense'].sum().reset_index()
    return selected_vessels

@st.cache_data
def get_cat_quartiles(filtered_result1):
        
    # Calculate percentiles for each month
    filtered_result1 = filtered_result1[filtered_result1.Expense > 0]
    monthly_data = filtered_result1.groupby(['PERIOD', 'VESSEL NAME', 'CATEGORIES'])['Expense'].sum()
    
    monthly_data = monthly_data.groupby(['PERIOD', 'CATEGORIES']).agg(
        q1=lambda x: np.quantile(x, 0.25),
        q2=lambda x: np.quantile(x, 0.50),
        median=lambda x: np.quantile(x, 0.63),
        q3=lambda x: np.quantile(x, 0.75)
        )
    
    # Extract dates and percentiles for plotting
    dates = monthly_data.index
    percentiles = monthly_data.reset_index()
    
    percentiles = percentiles.groupby(['CATEGORIES']).agg(
        median_50perc_population=('q2', lambda x: np.quantile(x, 0.50)),
        optimal_63perc_population=('median', lambda x: np.quantile(x, 0.63)),
        top_75perc_population=('q3', lambda x: np.quantile(x, 0.75))
    )
    
    return percentiles

@st.cache_data
def get_subcat_quartiles(filtered_result1):
    
    filtered_result1 = filtered_result1[filtered_result1.Expense > 0]
    monthly_data = filtered_result1.groupby(['PERIOD', 'VESSEL NAME', 'CATEGORIES', 'ACCOUNT_CODE', 'SUB_CATEGORIES'])['Expense'].sum()
    # Calculate percentiles for each month
    monthly_data = filtered_result1.groupby(['PERIOD', 'CATEGORIES', 'ACCOUNT_CODE', 'SUB_CATEGORIES'])['Expense'].agg(
        q1=lambda x: np.quantile(x, 0.25),
        q2=lambda x: np.quantile(x, 0.50),
        median=lambda x: np.quantile(x, 0.63),
        q3=lambda x: np.quantile(x, 0.75)
    )
    
    # Extract dates and percentiles for plotting
    # dates = monthly_data.index
    percentiles = monthly_data.reset_index()
    
    percentiles = percentiles.groupby(['CATEGORIES', 'ACCOUNT_CODE', 'SUB_CATEGORIES']).agg(
        median_50perc_population=('q2', lambda x: np.quantile(x, 0.50)),
        optimal_63perc_population=('median', lambda x: np.quantile(x, 0.63)),
        top_75perc_population=('q3', lambda x: np.quantile(x, 0.75))
    )
    
    return percentiles

def generate_segments(dates_series, interval_years=2, interval_months=6):
    segments = []
    dates_series.sort_values(inplace=True)
    current_date = dates_series.iloc[0]  # Start with the minimum date
    for date in dates_series.iloc[1:]:
        segment_end = current_date + timedelta(
            days=(interval_years * 365.25 + interval_months * 30.44) - 1
        )
        if date > segment_end:
            segments.append((current_date, segment_end))
            current_date = date
    segments.append((current_date, dates_series.iloc[-1]))  # Last segment
    return segments

def func(x):
    dates = pd.to_datetime(x['DATE']).copy()
    segments = generate_segments(dates)
    return segments

def get_aggregation(segments, df_dd, flag='cat'):
    results = []
    
    if flag == 'subcat':
        for x in segments.reset_index(name='segment')[['VESSEL NAME', 'segment', 'CATEGORIES', 'ACCOUNT_CODE', 'SUB_CATEGORIES']].values:
            for dt in x[1]:
                temp = df_dd[df_dd['VESSEL NAME'] == x[0]]
                temp['DATE'] = pd.to_datetime(temp['DATE'])
                expense = temp[(temp['DATE'] >= x[1][0][0]) & (temp['DATE'] == x[1][0][1])]['Expense'].sum()
                results.append((x[0], x[1][0], expense, x[2], x[3], x[4]))
    else:
        for x in segments.reset_index(name='segment')[['VESSEL NAME', 'segment', 'CATEGORIES']].values:
            for dt in x[1]:
                temp = df_dd[df_dd['VESSEL NAME'] == x[0]]
                temp['DATE'] = pd.to_datetime(temp['DATE'])
                expense = temp[(temp['DATE'] >= x[1][0][0]) & (temp['DATE'] == x[1][0][1])]['Expense'].sum()
                results.append((x[0], x[1][0], expense, x[2]))
    return results

@st.cache_data
def get_dd_cat(DF_DD):
    cost_centers = []
    expenses = []
    
    df_dd_cat = DF_DD.groupby(['VESSEL NAME', 'PERIOD', 'CATEGORIES']).Expense.sum().reset_index()
    df_dd_cat['DATE'] = df_dd_cat['PERIOD'].astype('str').apply(lambda x: f"{x[:4]}-{x[4:]}-01" if x else None)
    # df_dd_cat.groupby(['COST_CENTER', 'CATEGORIES', 'PERIOD', 'DATE'])['AMOUNT_USD'].sum().reset_index()
    cat_seg = df_dd_cat.groupby(['VESSEL NAME', 'CATEGORIES']).apply(func)

    for rec in cat_seg.reset_index(name='daterange').itertuples():
        cc = rec[1]
        for dd in rec[3]:
            temp = df_dd_cat[(df_dd_cat['VESSEL NAME'] == cc) & (df_dd_cat.DATE >= pd.to_datetime(dd[0]).strftime("%Y-%m-%d")) & (df_dd_cat.DATE <= pd.to_datetime(dd[1]).strftime("%Y-%m-%d"))]
            cost_centers.append(cc)
            expenses.append(temp.Expense.sum())
            
    cat_seg_event = pd.DataFrame()
    cat_seg_event['VESSEL NAME'] = pd.Series(cost_centers)
    cat_seg_event['EXPENSE'] = pd.Series(expenses)
    
    # st.write(f"CAT event {len(cost_centers), len(expenses), df_dd_cat.shape, cat_seg.shape}")
    
    # filtered_df = cat_seg_event
    filtered_df = cat_seg_event[cat_seg_event.EXPENSE > 0.00]
    q1 = int(filtered_df['EXPENSE'].quantile(0.25))
    q2 = int(filtered_df['EXPENSE'].quantile(0.50))  # This is the median
    q3 = int(filtered_df['EXPENSE'].quantile(0.75))
    

    # Create a DataFrame with quartile values
    return filtered_df, pd.DataFrame({'Quartile': ['CATEGORIES', 'median_50perc_population', 'optimal_63perc_population', 'top_75perc_population'],
                                'Value': [rec[2], q1, q2, q3]}).set_index('Quartile').T.reset_index(drop=True)
@st.cache_data
def get_dd_subcat(DF_DD):
    cost_centers = []
    expenses = []
    ac_codes=[]
    sub_cats = []

    df_dd_cat = DF_DD.groupby(['VESSEL NAME', 'PERIOD', 'CATEGORIES', 'ACCOUNT_CODE', 'SUB_CATEGORIES']).Expense.sum().reset_index()
    df_dd_cat['DATE'] = df_dd_cat['PERIOD'].astype('str').apply(lambda x: f"{x[:4]}-{x[4:]}-01" if x else None)
    # df_dd_cat.groupby(['COST_CENTER', 'CATEGORIES', 'PERIOD', 'DATE'])['AMOUNT_USD'].sum().reset_index()
    cat_seg = df_dd_cat.groupby(['VESSEL NAME', 'CATEGORIES', 'ACCOUNT_CODE', 'SUB_CATEGORIES']).apply(func)

    for rec in cat_seg.reset_index(name='daterange').itertuples():
        cc = rec[1]
        for dd in rec[5]:
            temp = df_dd_cat[(df_dd_cat['VESSEL NAME'] == cc) & (df_dd_cat.DATE >= pd.to_datetime(dd[0]).strftime("%Y-%m-%d")) & (df_dd_cat.DATE <= pd.to_datetime(dd[1]).strftime("%Y-%m-%d"))]
            cost_centers.append(cc)
            expenses.append(temp.Expense.sum())
            ac_codes.append(rec[3])
            sub_cats.append(rec[4])
            
    subcat_seg_event = pd.DataFrame()
    subcat_seg_event['VESSEL NAME'] = pd.Series(cost_centers)
    subcat_seg_event['CATEGORIES'] = rec[2]
    subcat_seg_event['ACCOUNT_CODE'] = pd.Series(ac_codes)
    subcat_seg_event['SUB_CATEGORIES'] = pd.Series(sub_cats)
    subcat_seg_event['EXPENSE'] = pd.Series(expenses)
    
    filtered_df = subcat_seg_event[subcat_seg_event.EXPENSE > 0.00]
    # filtered_df = subcat_seg_event

    subcat_df_pd = filtered_df.groupby(['CATEGORIES', 'ACCOUNT_CODE', 'SUB_CATEGORIES']).agg(
        median_50perc_population=('EXPENSE', lambda x: np.quantile(x, 0.50)),
        optimal_63perc_population=('EXPENSE', lambda x: np.quantile(x, 0.63)),
        top_75perc_population=('EXPENSE', lambda x: np.quantile(x, 0.75))
        )

    return filtered_df, subcat_df_pd

def get_pd_data(df_pd):
    cat_df_pd_ = df_pd.groupby(['VESSEL NAME', 'CATEGORIES']).Expense.median()

    subcat_df_pd_ = df_pd.groupby(['VESSEL NAME', 'CATEGORIES', 'ACCOUNT_CODE', 'SUB_CATEGORIES']).Expense.median()
    
    cat_df_pd = cat_df_pd_.groupby(['CATEGORIES']).agg(
        median_50perc_population=lambda x: np.quantile(x, 0.50),
        optimal_63perc_population=lambda x: np.quantile(x, 0.63),
        top_75perc_population=lambda x: np.quantile(x, 0.75)
        ).astype(int)
    
    subcat_df_pd = subcat_df_pd_.groupby(['CATEGORIES', 'ACCOUNT_CODE', 'SUB_CATEGORIES']).agg(
        median_50perc_population=lambda x: np.quantile(x, 0.50),
        optimal_63perc_population=lambda x: np.quantile(x, 0.63),
        top_75perc_population=lambda x: np.quantile(x, 0.75)
        ).astype(int)
    
    return cat_df_pd_, subcat_df_pd_, cat_df_pd, subcat_df_pd

def plotly_monthly_quartiles(data):
    try:
        monthly_data = data.groupby(['PERIOD', 'VESSEL NAME'])['Expense'].sum()
        # monthly_data = data.groupby(['PERIOD', 'CATEGORIES'])['Expense'].median()
    
        percentiles = monthly_data.groupby(['PERIOD']).agg(
            q1=lambda x: np.quantile(x, 0.25),
            q2=lambda x: np.quantile(x, 0.50),
            median=lambda x: np.quantile(x, 0.63),
            q3=lambda x: np.quantile(x, 0.75)
            ).reset_index()
        
        # # Extract dates and percentiles for plotting
        dates = pd.to_datetime(percentiles['PERIOD'], format='%Y%m').dt.date
        
        # st.dataframe(percentiles, use_container_width=True)

        # Create figure with custom size
        fig = go.Figure()

        # Plot quartiles as a shaded area
        fig.add_trace(go.Scatter(x=dates, y=percentiles['q3'], line=dict(color='#1a6933', shape='spline'), mode='lines', name='75% population'))
        fig.add_trace(go.Scatter(x=dates, y=percentiles['median'], line=dict(color='#cc2345', shape='spline'), mode='lines', name='median'))
        fig.add_trace(go.Scatter(x=dates, y=percentiles['q2'], line=dict(color='#ab871d', shape='spline'), mode='lines', name='50% population'))

        # Scatter plot for quartile points
        fig.add_trace(go.Scatter(x=dates, y=percentiles['q2'], mode='markers', marker=dict(color='white', size=3), name='q1 points', showlegend=False))
        fig.add_trace(go.Scatter(x=dates, y=percentiles['q3'], mode='markers', marker=dict(color='white', size=3), name='q3 points', showlegend=False))
        fig.add_trace(go.Scatter(x=dates, y=percentiles['median'], mode='markers', marker=dict(color='white', size=4), name='median points', showlegend=False))

        # Plot additional line for overall median
        q1_median = percentiles['q2'].median()
        overall_median = percentiles['median'].median()
        q3_median = percentiles['q3'].median()
        
        # fig.add_trace(go.Scatter(x=dates, y=[overall_median] * len(dates), line=dict(color='blue', dash='dash', width=1.5), mode='lines', name='overall median'))
        # fig.add_trace(go.Scatter(x=dates, y=[q1_median] * len(dates), line=dict(color='rgba(255,99,71,1)', dash='dash', width=1.5), mode='lines', name='q1_median'))
        # fig.add_trace(go.Scatter(x=dates, y=[q3_median] * len(dates), line=dict(color='rgba(135,206,250,1)', dash='dash', width=1.5), mode='lines', name='q3_median'))
        # Calculate the y-coordinate for the upper and lower text annotations
        
        upper_text_y = q3_median + 0.3 * (q3_median-overall_median) # Adjust the percentage as needed
        lower_text_y = q1_median - 0.3 * (overall_median-q1_median)  # Adjust the percentage as needed
        
        # st.write(lower_text_y, overall_median, upper_text_y)

        # Text label
        fig.add_trace(go.Scatter(x=[dates[0]], y=[overall_median], text=[f'{overall_median:.2f}'], mode='text', name='median text', textfont=dict(color='#de2669'), textposition="middle left", showlegend=False))
        fig.add_trace(go.Scatter(x=[dates[0]], y=[lower_text_y], text=[f'{q1_median:.2f}'], mode='text', name='q2_median text', textfont=dict(color='#ab871d'), textposition="bottom left", showlegend=False))
        fig.add_trace(go.Scatter(x=[dates[0]], y=[upper_text_y], text=[f'{q3_median:.2f}'], mode='text', name='q3_median text', textfont=dict(color='#1a6933'), textposition="top left", showlegend=False))
        
        fig.update_layout(
            title={          
                'text': 'Monthly Expenses / Per Ship - Per Month',
                'x': 0.3,
                'y': 0.95,
                'font': {
                    'color': '#FFFFFF',  # Set the title color to white
                    'size': 20
                    }
                },
            xaxis=dict(title='Month', color='black', showgrid=False, zeroline=False),  # Light grey grid color for x-axis
            yaxis=dict(title='Expense', color='black', showgrid=False, zeroline=False),
            xaxis_tickangle=-45,
            showlegend=True,
            height=500,
            width=1120,
            paper_bgcolor='#262c2e',  # Dark grey background color
            plot_bgcolor='#262c2e',      # Dark grey plot background color
            legend=dict(
                x=0.5,  # Horizontal position in the range [0, 1], where 0 is leftmost and 1 is rightmost
                y=1.1,  # Vertical position in the range [0, 1], where 0 is bottom and 1 is top
                orientation="h",  # Horizontal legend orientation
            )
        )

        return fig

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None
    
def plotly_yearly_quartiles(data):
    try:
        data['YEAR'] = data['YEAR'].astype('str')
        data = data.groupby(['YEAR', 'PERIOD', 'VESSEL NAME'])['Expense'].sum().reset_index()
        
        percentiles = data.groupby(['YEAR'])['Expense'].agg(
            optimal_q1=lambda x: np.quantile(x, 0.25),
            optimal_q2=lambda x: np.quantile(x, 0.50),
            optimal_median=lambda x: np.quantile(x, 0.63),
            optimal_q3=lambda x: np.quantile(x, 0.75)
            )
        
        dates = percentiles.index

        # Create figure with custom size
        fig = go.Figure()

        # Plot quartiles as a shaded area
        fig.add_trace(go.Scatter(x=dates, y=percentiles['optimal_q3'], line=dict(color='#1a6933', shape='spline'), mode='lines', name='75% population'))
        fig.add_trace(go.Scatter(x=dates, y=percentiles['optimal_median'], line=dict(color='#cc2345', shape='spline'), mode='lines', name='median'))
        fig.add_trace(go.Scatter(x=dates, y=percentiles['optimal_q2'], line=dict(color='#ab871d', shape='spline'), mode='lines', name='50% population'))

        # Scatter plot for quartile points
        fig.add_trace(go.Scatter(x=dates, y=percentiles['optimal_q2'], mode='markers', marker=dict(color='white', size=3), name='q2 points', showlegend=False))
        fig.add_trace(go.Scatter(x=dates, y=percentiles['optimal_q3'], mode='markers', marker=dict(color='white', size=3), name='q3 points', showlegend=False))
        fig.add_trace(go.Scatter(x=dates, y=percentiles['optimal_median'], mode='markers', marker=dict(color='white', size=4), name='median points', showlegend=False))

        # Plot additional line for overall median
        q1_median = percentiles['optimal_q2'].median()
        overall_median = percentiles['optimal_median'].median()
        q3_median = percentiles['optimal_q3'].median()
        
        
        fig.add_trace(go.Scatter(x=dates, y=[overall_median] * len(dates), line=dict(color='#de2669', dash='dash', width=0.5), mode='lines', name='overall median', showlegend=False))
        fig.add_trace(go.Scatter(x=dates, y=[q1_median] * len(dates), line=dict(color='#ab871d', dash='dash', width=0.5), mode='lines', name='q1_median', showlegend=False))
        fig.add_trace(go.Scatter(x=dates, y=[q3_median] * len(dates), line=dict(color='#1a6933', dash='dash', width=0.5), mode='lines', name='q3_median', showlegend=False))
        # Calculate the y-coordinate for the upper and lower text annotations
        
        # upper_text_y = q3_median + 0.1 * (q3_median-overall_median) # Adjust the percentage as needed
        # lower_text_y = q1_median - 0.1 * (overall_median-q1_median)
        
        # st.write(lower_text_y, overall_median, upper_text_y)

        # Text label
        fig.add_trace(go.Scatter(x=[dates[0]], y=[overall_median], text=[f'{overall_median:.2f}'], mode='text', name='overall median text', textfont=dict(color='#de2669'), textposition="middle left", showlegend=False))
        fig.add_trace(go.Scatter(x=[dates[0]], y=[q1_median], text=[f'{q1_median:.2f}'], mode='text', name='q1_median text', textfont=dict(color='#ab871d'), textposition="bottom left", showlegend=False))
        fig.add_trace(go.Scatter(x=[dates[0]], y=[q3_median], text=[f'{q3_median:.2f}'], mode='text', name='q3_median text', textfont=dict(color='#1a6933'), textposition="top left", showlegend=False))
        
        fig.update_layout(
            title={          
                'text': 'Yearly Expenses / Per Ship - Per Month',
                'x': 0.3,
                'y': 0.95,
                'font': {
                    'color': '#FFFFFF',  # Set the title color to white
                    'size': 20
                    }
                },
            xaxis=dict(title='Month', color='black', showgrid=False, zeroline=False),  # Light grey grid color for x-axis
            yaxis=dict(title='Expense', color='black', showgrid=False, zeroline=False),
            xaxis_tickangle=-45,
            showlegend=True,
            paper_bgcolor='#262c2e',  # Dark grey background color
            plot_bgcolor='#262c2e',     # Dark grey plot background color
            legend=dict(
                x=0.5,  # Horizontal position in the range [0, 1], where 0 is leftmost and 1 is rightmost
                y=1.1,  # Vertical position in the range [0, 1], where 0 is bottom and 1 is top
                orientation="h",  # Horizontal legend orientation
            )
        )
        
        return fig

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None

@st.cache_data
def det_cat_data(fr):
    df = get_cat_quartiles(fr)
    return df

@st.cache_data
def det_subcat_data(fr):
    df = get_subcat_quartiles(fr)
    return df
@st.cache_data
def call_get_subcat_optimal_mean(fr):
    x = get_subcat_optimal_mean(fr)
    return x

@st.cache_data
def call_get_cat_optimal_mean(fr):
    x = get_cat_optimal_mean(fr)
    return x

# Function to download DataFrame as CSV
# @st.cache_data
def download_csv(df):
    csv = df.to_csv()
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="data.csv">Download CSV File</a>'
    st.markdown(href, unsafe_allow_html=True)

def return_grid_options(data, flag='Cat'):
    
    if flag == 'subcat':
        gridOptions = {                
            # Enable Master / Detail
            "masterDetail": True,
            "rowSelection": "disabled",
            "detailRowHeight": 500,
            # Configure column definitions
            "columnDefs": [
                            {
                                "field": "Header", "minWidth": 400,
                                "cellRenderer": "agGroupCellRenderer",
                                "checkboxSelection": False,
                            },
                            {"field": "median_50perc_population", "minWidth": 150},
                            {"field": "optimal_63perc_population", "minWidth": 150},
                            {"field": "top_75perc_population", "minWidth": 150},
                            {"field": "Stats Model - Optimal Budget", "minWidth": 150}
                            
                        ],
            "defaultColDef": {
                "flex": 1,
            },
            # Provide Detail Cell Renderer Params
            "detailCellRendererParams": {
                # Provide the Grid Options to use on the Detail Grid
                "detailGridOptions": {
                    "rowSelection": "disable",
                    "suppressRowClickSelection": True,
                    "enableRangeSelection": True,
                    "pagination": False,
                    "paginationAutoPageSize": True,
                    "columnDefs": [
                                    {"field": "order", "minWidth": 50},
                                    {"field": "CATEGORIES", "minWidth": 400},
                                    {"field": "median_50perc_population", "minWidth": 150},
                                    {"field": "optimal_63perc_population", "minWidth": 150},
                                    {"field": "top_75perc_population", "minWidth": 150},
                                    {"field": "Stats Model - Optimal Budget"}
                                ],
                    "defaultColDef": {
                        "sortable": True,
                        "flex": 1,
                    },
                },
                # Function to get the rows for each Detail Grid
                "getDetailRowData": JsCode(
                    """function (params) {
                        params.successCallback(params.data.records);
                    }"""
                ),
            },
            # Provide the row data
            "rowData": data
            }
        
    elif flag == 'event':
        gridOptions = {                
            # Enable Master / Detail
            "masterDetail": True,
            "rowSelection": "disabled",
            "detailRowHeight": 500,
            # Configure column definitions
            "columnDefs": [
                            {
                                "field": "CATEGORIES", "minWidth": 400,
                                "cellRenderer": "agGroupCellRenderer",
                                "checkboxSelection": False,
                            },
                            {"field": "median_50perc_population", "minWidth": 150},
                            {"field": "optimal_63perc_population", "minWidth": 150},
                            {"field": "top_75perc_population", "minWidth": 150},
                            {"field": "Stats Model - Optimal Budget", "minWidth": 150}
                            
                        ],
            "defaultColDef": {
                "flex": 1,
            },
            # Provide Detail Cell Renderer Params
            "detailCellRendererParams": {
                # Provide the Grid Options to use on the Detail Grid
                "detailGridOptions": {
                    "rowSelection": "disable",
                    "suppressRowClickSelection": True,
                    "enableRangeSelection": True,
                    "pagination": False,
                    "paginationAutoPageSize": True,
                    "columnDefs": [
                                    {"field": "order", "minWidth": 50},
                                    {"field": "SUBCATEGORIES", "minWidth": 400},
                                    {"field": "median_50perc_population", "minWidth": 150},
                                    {"field": "optimal_63perc_population", "minWidth": 150},
                                    {"field": "top_75perc_population", "minWidth": 150},
                                    {"field": "Stats Model - Optimal Budget"}
                                ],
                    "defaultColDef": {
                        "sortable": True,
                        "flex": 1,
                    },
                },
                # Function to get the rows for each Detail Grid
                "getDetailRowData": JsCode(
                    """function (params) {
                        params.successCallback(params.data.records);
                    }"""
                ),
            },
            # Provide the row data
            "rowData": data
            }      
        
    else:
        gridOptions = {                
            # Enable Master / Detail
            "masterDetail": True,
            "rowSelection": "disabled",
            "detailRowHeight": 500,
            # Configure column definitions
            "columnDefs": [
                            {
                                "field": "Header",
                                "cellRenderer": "agGroupCellRenderer",
                                "checkboxSelection": False,
                            },
                            {"field": "median_50perc_population", "minWidth": 150},
                            {"field": "optimal_63perc_population", "minWidth": 150},
                            {"field": "top_75perc_population", "minWidth": 150},
                            {"field": "Stats Model - Optimal Budget", "minWidth": 150}
                        ],
            "defaultColDef": {
                "flex": 1,
            },
            # Provide Detail Cell Renderer Params
            "detailCellRendererParams": {
                # Provide the Grid Options to use on the Detail Grid
                "detailGridOptions": {
                    "rowSelection": "disable",
                    "suppressRowClickSelection": True,
                    "enableRangeSelection": True,
                    "pagination": False,
                    "paginationAutoPageSize": True,
                    "columnDefs": [
                                    {"field": "order", "minWidth": 50},
                                    {"field": "CATEGORIES", "minWidth": 150},
                                    {"field": "median_50perc_population", "minWidth": 150},
                                    {"field": "optimal_63perc_population", "minWidth": 150},
                                    {"field": "top_75perc_population", "minWidth": 150},
                                    {"field": "Stats Model - Optimal Budget", "minWidth": 150}
                                ],
                    "defaultColDef": {
                        "sortable": True,
                        "flex": 1,
                    },
                },
                # Function to get the rows for each Detail Grid
                "getDetailRowData": JsCode(
                    """function (params) {
                        params.successCallback(params.data.records);
                    }"""
                ),
            },
            # Provide the row data
            "rowData": data
            }
          
    return gridOptions

def get_json_data(cat_df, flag='cat'):
    
    if flag == 'cat':
        cat_df['median_50perc_population'] = cat_df['median_50perc_population'].astype(int)
        cat_df['optimal_63perc_population'] = cat_df['optimal_63perc_population'].astype(int)
        cat_df['top_75perc_population'] = cat_df['top_75perc_population'].astype(int)
        
        grp_tot_cat = cat_df.groupby(['Header']).sum().reset_index()
        
        condition = grp_tot_cat["Header"].isin(['Total OPEX', 'OPEX/DAY'])
        
        if condition.shape[0] > 0:
            grp_tot_cat.loc[condition, 'Stats Model - Optimal Budget'] = None
            grp_tot_cat.loc[condition, 'median_50perc_population'] = None
            grp_tot_cat.loc[condition, 'optimal_63perc_population'] = None
            grp_tot_cat.loc[condition, 'top_75perc_population'] = None
        
        grp_tot_cat['order'] = grp_tot_cat['Header'].apply(lambda x: order.index(x) if x in order else len(order))
        grp_tot_cat = grp_tot_cat.sort_values(by='order', ascending=True)
        grp_tot_cat = grp_tot_cat.set_index('Header')        
        grp_tot_cat = grp_tot_cat.sort_values(by='order', ascending=True)
        
        grp_tot_cat = grp_tot_cat.T.to_json()
        
        ## cate vise sum
        json_data = {}
        
        # Group by 'Header' column
        grouped = cat_df.groupby('Header')
        
        # Iterate over groups
        for group_name, group_data in grouped:
            group_json = group_data.to_dict(orient='records')
            json_data[group_name] = group_json
            # json_output = json.dumps(json_data, indent=4)
        
        grp_tot_cat = json.loads(grp_tot_cat)
        for key in grp_tot_cat.keys():
            grp_tot_cat[key].update({"records": json_data[key]})
            grp_tot_cat[key].update({"Header": key})                
            
        
        data = []
        for key, value in grp_tot_cat.items():
            data.append(grp_tot_cat[key])
    else:
        grp_tot_cat = cat_df.reset_index().groupby(['Header']).sum().reset_index()
        grp_tot_cat['order'] = grp_tot_cat['Header'].apply(lambda x: order.index(x) if x in order else len(order))
        grp_tot_cat = grp_tot_cat.sort_values(by='order', ascending=True)
        # grp_tot_cat.drop('order', axis=1, inplace=True)
        grp_tot_cat = grp_tot_cat.set_index('Header')
        grp_tot_cat = grp_tot_cat.astype('int')
        
        grp_tot_cat = grp_tot_cat.T.to_json()
        
        ## cate vise sum
        json_data = {}
        cat_df = cat_df.reset_index()
        # Concatenate the columns into a new column 'CATEGORIES'
        cat_df['CATEGORIES'] = cat_df['CATEGORIES'] + ': (' + cat_df['ACCOUNT_CODE'] + ', ' + cat_df['SUB_CATEGORIES'] + ')'
        cat_df = cat_df.drop(columns=['ACCOUNT_CODE', 'SUB_CATEGORIES'])
        
        # Group by 'Header' column
        grouped = cat_df.groupby('Header')
        
        # Iterate over groups
        for group_name, group_data in grouped:
            group_json = group_data.to_dict(orient='records')
            json_data[group_name] = group_json
            json_output = json.dumps(json_data, indent=4)
        
        grp_tot_cat = json.loads(grp_tot_cat)
        for key in grp_tot_cat.keys():
            grp_tot_cat[key].update({"records": json_data[key]})
            grp_tot_cat[key].update({"Header": key})                
            
        
        data = []
        for key, value in grp_tot_cat.items():
            data.append(grp_tot_cat[key])
    # st.json(data)
    return data

# Slicers
with st.container():
    # col1, col2, col3, col4, col5 = st.columns([1, 2, 1, 2, 2])  # Added another column for the selected vessels dropdown
    col1, col2, col3 = st.columns([1, 2, 1])
    # Vessel Type
    vessel_type_options = vessel_particulars['VESSEL TYPE'].unique()
    vessel_type = col1.selectbox('Vessel Type', vessel_type_options)

    # Vessel Subtype
    vessel_subtype_options = vessel_particulars[vessel_particulars['VESSEL TYPE'] == vessel_type]['VESSEL SUBTYPE'].unique()
    vessel_subtype = col2.multiselect('Vessel Subtype', vessel_subtype_options, default=vessel_subtype_options[0])

    # Vessel Age Range
    default_age_start, default_age_end = 0, 20
    vessel_age_start, vessel_age_end = col3.slider("Vessel Age Range",
                                                   min_value=default_age_start,
                                                   max_value=default_age_end,
                                                   value=(default_age_start, default_age_end))

    # Trigger query and fetch results
    filtered_result1 = filter_dataframe(vessel_type, vessel_subtype, vessel_age_start, vessel_age_end)
    
    # Assuming df is your DataFrame
    filtered_result1 = filtered_result1[~filtered_result1['ACCOUNT_CODE'].str.contains('.*-999')]
    filtered_result1 = filtered_result1[~filtered_result1['ACCOUNT_CODE'].str.contains('.*-998')]

with st.container():
    
    col4, col5, col6 = st.columns([2, 1, 1])
    
    # Categories
    vessel_cat_options = filtered_result1['CATEGORIES'].unique()
    vessel_cat_options = ['Select All'] + vessel_cat_options.tolist()
    vessel_cat = col4.multiselect("Categories", vessel_cat_options, default=['Select All'])
    
    if 'Select All' in vessel_cat:
        vessel_cat = vessel_cat_options[1:]
        
    filtered_result1 = filtered_result1[filtered_result1['CATEGORIES'].isin(vessel_cat)]
    
    # Sub Categories
    vessel_subcat_options = filtered_result1['SUB_CATEGORIES'].unique()
    vessel_subcat_options = ['Select All'] + vessel_subcat_options.tolist()
    vessel_subcat = col5.multiselect("Sub Categories", vessel_subcat_options, default=['Select All'])

    if 'Select All' in vessel_subcat:
        vessel_subcat = vessel_subcat_options[1:]
        
    # Filter by selected subcategories
    filtered_result = filtered_result1[filtered_result1['SUB_CATEGORIES'].isin(vessel_subcat)].reset_index(drop=True)
    
    # Selected Vessels Dropdown
    selected_vessels = filtered_result['VESSEL NAME'].unique()
    selected_vessels_option = ['Select All'] + selected_vessels.tolist()
    selected_vessels_dropdown = col6.multiselect('Selected Vessels', selected_vessels_option, default=['Select All'])  # Dropdown for selected vessels

    # Filter DataFrame based on selected vessels
    if 'Select All' in selected_vessels_dropdown:
        selected_vessels_dropdown = selected_vessels_option[1:]
    
    filtered_result = filtered_result[filtered_result['VESSEL NAME'].isin(selected_vessels_dropdown)].reset_index(drop=True)
    
    mrk_down = f"\n<u>Vessels Selected Count:</u> {filtered_result['COST_CENTER'].nunique()}"
    st.markdown(f"<h5 style='text-align: left; color: white;'>{mrk_down}</h5>", unsafe_allow_html=True)
    # st.write(f"<u>Vessel Selected Count:</u> {filtered_result['COST_CENTER'].nunique()}")
    
    st.divider()

    filtered_result['DATE'] = pd.to_datetime(filtered_result['PERIOD'], format='%Y%m').dt.date
    filtered_result['YEAR'] = pd.to_datetime(filtered_result['DATE']).dt.year.astype('str')
    
    cat_df = det_cat_data(filtered_result).reset_index()
    subcat_df = det_subcat_data(filtered_result).reset_index()
    # cat_df['Model-1 Budget'] = calculate_geometric_mean(filtered_result)['Geometric Mean']
    # subcat_df['Model-1 Budget'] = calculate_geometric_mean(filtered_result, level='SUB_CATEGORIES')['Geometric Mean']
    
    manning = ['CREW WAGES', 'CREW EXPENSES', 'VICTUALLING EXPENSES']
    tech = ['STORES', 'SPARES', 'REPAIRS & MAINTENANCE', 'MISCELLANEOUS', 'LUBE OIL CONSUMPTION']
    fees = ['MANAGEMENT FEES']
    admin = ['VESSEL BANK CHARGE', 'ADMINISTRATIVE EXPENSES']
    
    budgeted_expenses = manning + tech + fees + admin
    non_budget = ['INSURANCE', 'P&I/H&M EXPENSES', 'CAPITAL EXPENDITURE', 'NON-BUDGETED EXPENSES', 'VOYAGE/CHARTERERS EXPENSES', 'EXTRA ORDINARY ITEMS', 'VESSEL UPGRADING COSTS', 'SHIP SOFTWARE']
    event_cats = ['PRE-DELIVERY EXPENSES', 'DRYDOCKING EXPENSES']
    all_cat = budgeted_expenses + non_budget + event_cats
    
    # Assign order based on the dictionary
    cat_df['order'] = cat_df['CATEGORIES'].apply(lambda x: all_cat.index(x) if x in all_cat else len(all_cat))
    cat_df = cat_df.sort_values(by='order')
    cat_df.drop('order', axis=1, inplace=True)
    # print(cat_df.columns)
    
    subcat_df['order'] = subcat_df['CATEGORIES'].apply(lambda x: all_cat.index(x) if x in all_cat else len(all_cat))
    subcat_df = subcat_df.sort_values(by='order')
    subcat_df.drop('order', axis=1, inplace=True)
    # print(subcat_df.columns)
    
    def group_budget(x):
        if x in manning:
            return "Manning"
        elif x in tech:
            return "Technical"
        elif x in fees:
            return "Management"
        elif x in admin:
            return "Administrative Expenses"
        elif x in non_budget:
            return "ADDITIONAL CATEGORIES"
        elif x in event_cats:
            return "EVENT CATEGORIES"
        
    def split_cats(x):
        x=x.split(",")
        return x[0].strip('(')
    
    # import balham as st
    if page == "1. Report Page":
        order = ['Manning', 'Technical', 'Management', 'Administative Expenses']
        tab1, tab2, tab3 = st.tabs(['BUDGET CATEGORIES', 'ADDITIONAL CATEGORIES', 'EVENT CATEGORIES'])
        
        with tab1:
            budget_cat_df = cat_df[cat_df.CATEGORIES.isin(budgeted_expenses)]
            budget_cat_df['Header'] = budget_cat_df['CATEGORIES'].apply(group_budget)
            budget_cat_df = budget_cat_df.set_index(['Header', 'CATEGORIES']).astype('int').reset_index().set_index('Header')
            budget_cat_df['order'] = budget_cat_df['CATEGORIES'].apply(lambda x: all_cat.index(x) if x in all_cat else len(all_cat))
            budget_cat_df = budget_cat_df.sort_values(by='order')
            
            df2 = budget_cat_df.copy()
            wo_lo = df2[df2.CATEGORIES != 'LUBE OIL CONSUMPTION'].reset_index().set_index((['Header', 'CATEGORIES']))
            wo_lo = wo_lo.sum(numeric_only=True)
            wo_lo = wo_lo.reset_index(name='Total OPEX').transpose().tail(1)
            wo_lo['CATEGORIES'] = 'Without Lube Oil'
            
            wo_lo = wo_lo.rename(columns={0:'median_50perc_population', 
                                  1:'optimal_63perc_population',
                                  2:'top_75perc_population',
                                  3 : 'order'}).reset_index(names='Header')
            
            opex_per_day = pd.DataFrame()
            opex_per_day['median_50perc_population'] = wo_lo['median_50perc_population']//30
            opex_per_day['optimal_63perc_population'] = wo_lo['optimal_63perc_population']//30
            opex_per_day['top_75perc_population'] = wo_lo['top_75perc_population']//30
            opex_per_day['order'] = 99
            opex_per_day['CATEGORIES'] = 'Without Lube Oil'
            opex_per_day['Header'] = 'OPEX/DAY'
            
            wo_lo = pd.concat([wo_lo, opex_per_day])
            # st.dataframe(wo_lo)
            
            
            w_lo = df2.copy().reset_index().set_index((['Header', 'CATEGORIES']))
            w_lo = w_lo.sum(numeric_only=True)
            w_lo = w_lo.reset_index(name='Total OPEX').transpose().tail(1)
            w_lo['CATEGORIES'] = 'With Lube Oil'
            
            w_lo = w_lo.rename(columns={0:'median_50perc_population', 
                                  1:'optimal_63perc_population',
                                  2:'top_75perc_population',
                                  3 : 'order'}).reset_index(names='Header')
            
            opex_per_day = pd.DataFrame()
            opex_per_day['median_50perc_population'] = w_lo['median_50perc_population']//30
            opex_per_day['optimal_63perc_population'] = w_lo['optimal_63perc_population']//30
            opex_per_day['top_75perc_population'] = w_lo['top_75perc_population']//30
            opex_per_day['order'] = 100
            opex_per_day['CATEGORIES'] = 'With Lube Oil'
            opex_per_day['Header'] = 'OPEX/DAY'
            
            w_lo = pd.concat([w_lo, opex_per_day])
            
            total_df = pd.concat([w_lo, wo_lo]).sort_values(by=['order']).reset_index(drop=True)
            budget_cat_df = pd.concat([budget_cat_df.reset_index(), total_df]).sort_values(by='order', ascending=True)
            
            st.markdown("<h3 style='text-align: center; color: #0476D0;'><u>CATEGORY LEVEL BUDGET - PER MONTH </u></h3>", unsafe_allow_html=True)
            
            dload1 = st.empty()
            with dload1:
                download_csv(budget_cat_df)
                
            budget_catdf_placeholder = st.empty()
            with budget_catdf_placeholder:
                budget_cat_data = get_json_data(budget_cat_df)     
                
                # Specify the filename
                filename = 'check.json'
                # Open a file in write mode and save the JSON data
                with open(filename, 'w') as file:
                    json.dump(budget_cat_data, file, indent=4)
                    
                gridOptions = return_grid_options(budget_cat_data)
                r = AgGrid(None, height=250, gridOptions=gridOptions, allow_unsafe_jscode=True, enable_enterprise_modules=True, udate_mode=GridUpdateMode.SELECTION_CHANGED, key="an_unique_key", fit_columns_on_grid_load=True, theme='balham')
            # budget_catdf_placeholder.dataframe(budget_cat_df, use_container_width=True)
            
            st.markdown("<h3 style='text-align: center; color: #0476D0;'><u>SUB-CATEGORY LEVEL BUDGET - PER MONTH </u></h3>", unsafe_allow_html=True)
            budget_subcatdf = subcat_df[subcat_df.CATEGORIES.isin(budgeted_expenses)]
            budget_subcatdf['Header'] = budget_subcatdf['CATEGORIES'].apply(group_budget)
            budget_subcatdf = budget_subcatdf.set_index(['Header', 'CATEGORIES', 'ACCOUNT_CODE', 'SUB_CATEGORIES']).astype('int').reset_index()
            budget_subcatdf['order'] = budget_subcatdf['CATEGORIES'].apply(lambda x: all_cat.index(x) if x in all_cat else len(all_cat))
            budget_subcatdf = budget_subcatdf.sort_values(by='order')
            # budget_subcatdf.drop('order', axis=1, inplace=True)
            budget_subcatdf = budget_subcatdf.set_index(['Header', 'CATEGORIES', 'ACCOUNT_CODE', 'SUB_CATEGORIES'])
            # budget_subcatdf_placeholder.dataframe(budget_subcatdf, use_container_width=True)
            dload2 = st.empty()
            with dload2:
                download_csv(budget_subcatdf)
                
            budget_subcatdf_placeholder = st.empty()
            with budget_subcatdf_placeholder:
                budget_subcat_data = get_json_data(budget_subcatdf, flag='subcat')
                    
                gridOptions = return_grid_options(budget_subcat_data, flag='subcat')
                r = AgGrid(None, height=250, gridOptions=gridOptions, allow_unsafe_jscode=True, enable_enterprise_modules=True, udate_mode=GridUpdateMode.SELECTION_CHANGED, fit_columns_on_grid_load=True, theme='balham')
            
        with tab2:
            nonbudget_cat_df = cat_df[cat_df.CATEGORIES.isin(non_budget)]
            nonbudget_cat_df['Header'] = nonbudget_cat_df['CATEGORIES'].apply(group_budget)
            nonbudget_cat_df = nonbudget_cat_df.set_index(['Header', 'CATEGORIES']).astype('int').reset_index().set_index('Header')
            nonbudget_cat_df['order'] = nonbudget_cat_df['CATEGORIES'].apply(lambda x: all_cat.index(x) if x in all_cat else len(all_cat))
            nonbudget_cat_df = nonbudget_cat_df.sort_values(by='order', ascending=True)
            # nonbudget_cat_df.drop('order', axis=1, inplace=True)
            
            st.markdown("<h3 style='text-align: center; color: #0476D0;'><u>CATEGORY LEVEL BUDGET - PER MONTH </u></h3>", unsafe_allow_html=True)
            dload3 = st.empty()
            with dload3:
                download_csv(nonbudget_cat_df)
            non_budget_cat_data = get_json_data(nonbudget_cat_df)
            # st.dataframe(nonbudget_cat_df)
            
            nonbudget_catdf_placeholder = st.empty()
            with nonbudget_catdf_placeholder:
                gridOptions = return_grid_options(non_budget_cat_data)
                r = AgGrid(None, height=250, gridOptions=gridOptions, allow_unsafe_jscode=True, enable_enterprise_modules=True, udate_mode=GridUpdateMode.SELECTION_CHANGED, fit_columns_on_grid_load=True, theme='balham')
                
            # budget_catdf_placeholder.dataframe(budget_cat_df, use_container_width=True)
            
            st.markdown("<h3 style='text-align: center; color: #0476D0;'><u>SUB-CATEGORY LEVEL BUDGET - PER MONTH </u></h3>", unsafe_allow_html=True)
            nonbudget_subcatdf = subcat_df[subcat_df.CATEGORIES.isin(non_budget)]
            nonbudget_subcatdf['Header'] = nonbudget_subcatdf['CATEGORIES'].apply(group_budget)
            nonbudget_subcatdf = nonbudget_subcatdf.set_index(['Header', 'CATEGORIES', 'ACCOUNT_CODE', 'SUB_CATEGORIES']).astype('int').reset_index()
            nonbudget_subcatdf['order'] = nonbudget_subcatdf['CATEGORIES'].apply(lambda x: all_cat.index(x) if x in all_cat else len(all_cat))
            nonbudget_subcatdf = nonbudget_subcatdf.sort_values(by='order')
            # nonbudget_cat_df.drop('order', axis=1, inplace=True)
            nonbudget_subcatdf = nonbudget_subcatdf.set_index(['Header', 'CATEGORIES', 'ACCOUNT_CODE', 'SUB_CATEGORIES'])
            
            # budget_subcatdf_placeholder.dataframe(budget_subcatdf, use_container_width=True)
            dload4 = st.empty()
            with dload4:
                download_csv(nonbudget_subcatdf)
            nonbudget_subcat_data = get_json_data(nonbudget_subcatdf, flag='subcat')
                
            # st.dataframe(nonbudget_subcatdf)
            
            nonbudget_subcatdf_placeholder = st.empty()
            with nonbudget_subcatdf_placeholder:
                gridOptions = return_grid_options(nonbudget_subcat_data, flag='subcat')
                r = AgGrid(None, height=250, gridOptions=gridOptions, allow_unsafe_jscode=True, enable_enterprise_modules=True, udate_mode=GridUpdateMode.SELECTION_CHANGED, fit_columns_on_grid_load=True, theme='balham')
                
        with tab3:
            event_cat_df = filtered_result[filtered_result.CATEGORIES.isin(event_cats)].reset_index(drop=True)
            
            df_dd = event_cat_df[event_cat_df['CATEGORIES'] == 'DRYDOCKING EXPENSES'].reset_index(drop=True)
            df_pd = event_cat_df[event_cat_df['CATEGORIES'] == 'PRE-DELIVERY EXPENSES'].reset_index(drop=True)
                
            cat_seg_, subcat_seg_, cat_seg, subcat_seg = get_pd_data(df_pd)
            dd_subcat_filtered_df, subcat_df_seg = get_dd_subcat(df_dd)
            
            numeric_cols = subcat_df_seg.select_dtypes(include='number').columns
            subcat_df_seg[numeric_cols] = subcat_df_seg[numeric_cols].astype(int)
            
            event_subcat_df1 = pd.concat([subcat_seg, subcat_df_seg]).reset_index()
            
            st.markdown("<h3 style='text-align: center; color: #0476D0;'><u>SUB-CATEGORY LEVEL BUDGET - PER EVENT </u></h3>", unsafe_allow_html=True)
            
            event_subcat_df = event_subcat_df1.copy()
            event_subcat_df['Header'] = event_subcat_df.reset_index()['CATEGORIES'].apply(group_budget)
            event_subcatdf = event_subcat_df.set_index(['Header', 'CATEGORIES', 'ACCOUNT_CODE', 'SUB_CATEGORIES']).astype('int').reset_index()
            event_subcatdf['order'] = event_subcatdf['CATEGORIES'].apply(lambda x: all_cat.index(x) if x in all_cat else len(all_cat))
            event_subcatdf = event_subcatdf.sort_values(by='order')
            # event_subcatdf = event_subcatdf.set_index(['Header', 'CATEGORIES']).astype('int').reset_index().set_index('Header')
            event_subcatdf = event_subcatdf.set_index(['Header', 'CATEGORIES', 'ACCOUNT_CODE', 'SUB_CATEGORIES']).reset_index()
            event_subcatdf['Header'] = event_subcatdf['CATEGORIES']
            event_subcatdf['CATEGORIES'] = event_subcatdf['ACCOUNT_CODE'] + "; " + event_subcatdf['SUB_CATEGORIES']
            # event_subcatdf.to_excel('event_excel.xlsx', index=False)
            
            # budget_subcatdf_placeholder.dataframe(budget_subcatdf, use_container_width=True)
            dload6 = st.empty()
            with dload6:
                download_csv(event_subcatdf)
            event_subcat_data = get_json_data(event_subcatdf, flag='cat')
            # st.dataframe(nonbudget_subcatdf)
            
            event_subcatdf_placeholder = st.empty()
            with event_subcatdf_placeholder:
                gridOptions = return_grid_options(event_subcat_data, flag='cat')
                r = AgGrid(None, height=750, gridOptions=gridOptions, allow_unsafe_jscode=True, enable_enterprise_modules=True, udate_mode=GridUpdateMode.SELECTION_CHANGED, fit_columns_on_grid_load=True, theme='balham')
        
        ########## Budget ############
        
        cat_df = get_cat_optimal_mean(filtered_result[filtered_result.CATEGORIES.isin(budgeted_expenses)])
        cat_df['Stats Model - Optimal Budget'] = cat_df['Stats Model - Optimal Budget'].astype(int)
        
        cat_df = cat_df.set_index('CATEGORIES').join(budget_cat_df.set_index('CATEGORIES'), lsuffix='_cat_df', rsuffix='_budget_cat_df', how='outer').reset_index()
        temp = cat_df[~cat_df.CATEGORIES.isin(['With Lube Oil', 'Without Lube Oil'])].reset_index(drop=True)
        
        w_lo_tot = temp['Stats Model - Optimal Budget'].sum()
        wo_lo_tot = temp[temp.CATEGORIES != 'LUBE OIL CONSUMPTION']['Stats Model - Optimal Budget'].sum()
        
        conditions = (cat_df['CATEGORIES'] == 'With Lube Oil') & (cat_df['Header'] == 'Total OPEX')
        cat_df.loc[conditions, 'Stats Model - Optimal Budget'] = w_lo_tot
        
        conditions = (cat_df['CATEGORIES'] == 'Without Lube Oil') & (cat_df['Header'] == 'Total OPEX')
        cat_df.loc[conditions, 'Stats Model - Optimal Budget'] = wo_lo_tot
        
        conditions = (cat_df['CATEGORIES'] == 'With Lube Oil') & (cat_df['Header'] == 'OPEX/DAY')
        cat_df.loc[conditions, 'Stats Model - Optimal Budget'] = w_lo_tot/30
        
        conditions = (cat_df['CATEGORIES'] == 'Without Lube Oil') & (cat_df['Header'] == 'OPEX/DAY')
        cat_df.loc[conditions, 'Stats Model - Optimal Budget'] = wo_lo_tot/30
        
        budget_cat_df = cat_df.set_index(['Header', 'CATEGORIES']).astype('int').reset_index().set_index('Header')
        budget_cat_df = budget_cat_df.sort_values(by='order', ascending=True)
        
        with dload1:
            download_csv(budget_cat_df)
        with budget_catdf_placeholder:
            budget_cat_data = get_json_data(budget_cat_df)  
            gridOptions = return_grid_options(budget_cat_data)
            r = AgGrid(None, height=250, gridOptions=gridOptions, allow_unsafe_jscode=True, enable_enterprise_modules=True, udate_mode=GridUpdateMode.SELECTION_CHANGED, fit_columns_on_grid_load=True, theme='balham')
        
        subcat_df = get_subcat_optimal_mean(filtered_result[filtered_result.CATEGORIES.isin(budgeted_expenses)])
        
        subcat_df['CATEGORIES'] = subcat_df.reset_index()['SUB CATEGORIES'].apply(split_cats)
        # st.dataframe(subcat_df)
        
        subcat_df['order'] = subcat_df['CATEGORIES'].apply(lambda x: all_cat.index(x) if x in all_cat else len(all_cat))
        subcat_df = subcat_df.sort_values(by='order')
        subcat_df['Stats Model - Optimal Budget'] = subcat_df['Stats Model - Optimal Budget'].astype(int)
        budget_subcatdf = budget_subcatdf.reset_index()
        
        scats = []
        for index, row in budget_subcatdf[['CATEGORIES', 'ACCOUNT_CODE', 'SUB_CATEGORIES']].iterrows():
            scats.append("('{}', '{}', '{}')".format(row['CATEGORIES'], row['ACCOUNT_CODE'], row['SUB_CATEGORIES']))
        
        budget_subcatdf['SUB CATEGORIES'] = pd.Series(scats)
        budget_subcatdf = budget_subcatdf.merge(subcat_df, on='SUB CATEGORIES')
        budget_subcatdf = budget_subcatdf.rename(columns={'SUB CATEGORIES' : 'CATEGORIES'})
        # budget_subcatdf = budget_subcatdf.set_index(['CATEGORIES', 'ACCOUNT_CODE', 'SUB_CATEGORIES'])
        
        with dload2:
            download_csv(budget_subcatdf)
            
        with budget_subcatdf_placeholder:       
            budget_subcat_data = get_json_data(budget_subcatdf)         
            gridOptions = return_grid_options(budget_subcat_data)
            r = AgGrid(None, height=250, gridOptions=gridOptions, allow_unsafe_jscode=True, enable_enterprise_modules=True, udate_mode=GridUpdateMode.SELECTION_CHANGED, fit_columns_on_grid_load=True, theme='balham')
        
        
        ########## Non-Budget ############
        cat_df = get_cat_optimal_mean(filtered_result[filtered_result.CATEGORIES.isin(non_budget)])
        cat_df['Stats Model - Optimal Budget'] = cat_df['Stats Model - Optimal Budget'].astype(int)
        cat_df = cat_df.merge(nonbudget_cat_df, on='CATEGORIES')
        cat_df['Header'] = cat_df.CATEGORIES.apply(group_budget)
        cat_df['order'] = cat_df['CATEGORIES'].apply(lambda x: all_cat.index(x) if x in all_cat else len(all_cat))
        cat_df = cat_df.sort_values(by='order')
        nonbudget_cat_df = cat_df.set_index(['Header', 'CATEGORIES']).astype('int').reset_index().set_index('Header')
        
        with dload3:
            download_csv(nonbudget_cat_df)
            
        with nonbudget_catdf_placeholder:       
            nonbudget_cat_data = get_json_data(nonbudget_cat_df)
            gridOptions = return_grid_options(nonbudget_cat_data)
            r = AgGrid(None, height=250, gridOptions=gridOptions, allow_unsafe_jscode=True, enable_enterprise_modules=True, udate_mode=GridUpdateMode.SELECTION_CHANGED, fit_columns_on_grid_load=True, theme='balham')
            
        
        subcat_df = get_subcat_optimal_mean(filtered_result[filtered_result.CATEGORIES.isin(non_budget)])
        subcat_df['Stats Model - Optimal Budget'] = subcat_df['Stats Model - Optimal Budget'].astype(int)
        nonbudget_subcatdf = nonbudget_subcatdf.reset_index()
        
        scats = []
        for index, row in nonbudget_subcatdf[['CATEGORIES', 'ACCOUNT_CODE', 'SUB_CATEGORIES']].iterrows():
            scats.append("('{}', '{}', '{}')".format(row['CATEGORIES'], row['ACCOUNT_CODE'], row['SUB_CATEGORIES']))
        
        nonbudget_subcatdf['SUB CATEGORIES'] = pd.Series(scats)
        nonbudget_subcatdf = nonbudget_subcatdf.merge(subcat_df, on='SUB CATEGORIES')
        nonbudget_subcatdf = nonbudget_subcatdf.rename(columns={'SUB CATEGORIES' : 'CATEGORIES'})
        # nonbudget_subcatdf = nonbudget_subcatdf.set_index(['CATEGORIES', 'ACCOUNT_CODE', 'SUB_CATEGORIES'])
        
        with dload4:
            download_csv(nonbudget_subcatdf)
            
        with nonbudget_subcatdf_placeholder:       
            nonbudget_subcat_data = get_json_data(nonbudget_subcatdf)         
            gridOptions = return_grid_options(nonbudget_subcat_data)
            r = AgGrid(None, height=250, gridOptions=gridOptions, allow_unsafe_jscode=True, enable_enterprise_modules=True, udate_mode=GridUpdateMode.SELECTION_CHANGED, fit_columns_on_grid_load=True, theme='balham')
        
        ############ Event Categories ########
        event_DF = pd.concat([df_dd, df_pd])
        
        subcat_df = get_event_subcat_optimal_mean(event_DF.rename(columns={'Expense' : 'EXPENSE'}))
        subcat_df['Stats Model - Optimal Budget'] = subcat_df['Stats Model - Optimal Budget'].astype(int)
        subcat_df['ACCOUNT_CODE'] = subcat_df['SUB CATEGORIES'].apply(lambda x: x.strip().split(";")[1])
        # subcat_df = subcat_df.rename(columns={'SUB CATEGORIES' : 'CATEGORIES'})
        
        subcat_df = pd.merge(event_subcatdf, subcat_df, on='ACCOUNT_CODE', how='inner').drop(columns=['order'])
        
        # Assuming you know the column names or have a way to identify them
        numeric_cols = [col for col in subcat_df.columns if subcat_df[col].dtype in ['int32', 'float32']]
        numeric_sum = subcat_df[numeric_cols].sum()
        numeric_sum['Header']='TOTAL EVENT EXPENSES'
        # numeric_sum = numeric_sum.rename(columns={0 : 'Total Event Expense'})
        
        # st.json(numeric_sum.to_dict())
        
        with dload6:
            download_csv(subcat_df)
            
        with event_subcatdf_placeholder: 
            event_subcat_data = get_json_data(subcat_df, flag='cat')
            event_subcat_data.append(numeric_sum.to_dict())
            gridOptions = return_grid_options(event_subcat_data, flag='cat')
            r = AgGrid(None, height=750, gridOptions=gridOptions, allow_unsafe_jscode=True, enable_enterprise_modules=True, udate_mode=GridUpdateMode.SELECTION_CHANGED, fit_columns_on_grid_load=True, theme='balham')
        
    elif page == "2. Trend Analysis":
        with st.container():
            st.markdown("<h2 style='text-align: center; color: #0476D0;'><u>Trend Analysis </u></h2>", unsafe_allow_html=True)
            
            st.markdown("<h3 style='text-align: left; color: #0476D0;'><u>1. Monthly Trend - PER MONTH </u></h3>", unsafe_allow_html=True)
            st.plotly_chart(plotly_monthly_quartiles(filtered_result), use_container_width=True)
            
            st.markdown("<h3 style='text-align: left; color: #0476D0;'><u>2. Yearly - PER MONTH </u></h3>", unsafe_allow_html=True)
            st.plotly_chart(plotly_yearly_quartiles(filtered_result), use_container_width=True)
            
            st.markdown("<h3 style='text-align: left; color: #0476D0;'><u>Yearly Trend - PER MONTH </u></h3>", unsafe_allow_html=True)
            st.pyplot(new_modified.get_optimal_mean(filtered_result))
            # st.plotly_chart(new_modified.get_optimal_mean(filtered_result), use_container_width=True)
            st.pyplot((whole_year.get_optimal_mean(filtered_result)))