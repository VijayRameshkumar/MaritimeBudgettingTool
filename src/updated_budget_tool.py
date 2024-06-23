import pandas as pd
from multithreading_optimization1 import get_cat_optimal_mean, get_subcat_optimal_mean

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import plotly.graph_objects as go
from src import new_modified
from src import whole_year
# from src.snowflake_connect import get_expense_data

### Read Data
vessel_particulars = pd.read_excel('VESSEL_PARTICULARS.xlsx')
last_3_years = pd.read_csv('2021_2022_2023_expense.csv', skiprows=1).rename(columns={'AMOUNT_USD': 'Expense'})
# last_3_years = get_expense_data()
# Merge last_3_years with vessel_particulars to include DATE column
merged_df = pd.merge(last_3_years, vessel_particulars[['COST_CENTER', 'VESSEL NAME', 'BUILD YEAR']], on='COST_CENTER', how='left')
# Add VESSEL AGE to merged_df
merged_df['VESSEL AGE'] = pd.to_datetime(merged_df['DATE']).dt.year - merged_df['BUILD YEAR']
last_3_years['VESSEL AGE'] = pd.to_datetime(merged_df['DATE']).dt.year - merged_df['BUILD YEAR']

# Set page layout
st.set_page_config(layout="wide")
# Sidebar navigation
page = st.sidebar.radio("Navigation", ["1. Report Page", "2. Trend Analysis"])

with st.container():
    # col1, col2, col3, col4, col5 = st.columns([1, 2, 1, 2, 2])  # Added another column for the selected vessels dropdown
    col000, col00 = st.columns([1, 2])

    logo_path = "./Synergy.jpg"
    col000.image(logo_path)
    col00.markdown("<h1 style='text-align: left; margin-top: 1em; color: black; '><u>Budget Analysis Tool v1.0</u></h1>", unsafe_allow_html=True)
    # logo_path = "./Syn-DD.png"
    # col0.image(logo_path, width=250)
    st.markdown("---")

# Filter DataFrame based on slicer values
def filter_dataframe(vessel_type, vessel_subtype, vessel_age_start, vessel_age_end):
    filtered_df = vessel_particulars[
        (vessel_particulars['VESSEL TYPE'] == vessel_type) &
        (vessel_particulars['VESSEL SUBTYPE'].isin(vessel_subtype))]
    
    selected_vessels = merged_df[merged_df['COST_CENTER'].isin(filtered_df['COST_CENTER'].unique())].reset_index(drop=True)
    selected_vessels = selected_vessels[selected_vessels['VESSEL AGE'].between(vessel_age_start, vessel_age_end)]
    selected_vessels = selected_vessels.groupby(['PERIOD', 'COST_CENTER', 'VESSEL NAME', 'CATEGORIES', 'ACCOUNT_CODE', 'SUB_CATEGORIES'])['Expense'].sum().reset_index()
    return selected_vessels

def get_cat_quartiles(filtered_result1):
        
    # Calculate percentiles for each month
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
        low_25_perc_population=('q1', lambda x: np.quantile(x, 0.25)),
        optimal_50perc_population=('q2', lambda x: np.quantile(x, 0.50)),
        optimal_63perc_population=('median', lambda x: np.quantile(x, 0.63)),
        top_75perc_population=('q3', lambda x: np.quantile(x, 0.75))
    )
    
    return percentiles

def get_subcat_quartiles(filtered_result1):
    
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
        low_25_perc_population=('q1', lambda x: np.quantile(x, 0.25)),
        optimal_50perc_population=('q2', lambda x: np.quantile(x, 0.50)),
        optimal_63perc_population=('median', lambda x: np.quantile(x, 0.63)),
        top_75perc_population=('q3', lambda x: np.quantile(x, 0.75))
    )
    
    return percentiles

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
                'x': 0.0,
                'y': 0.95,
                'font': {
                    'color': '#252425',  # Set the title color to white
                    'size': 20
                    }
                },
            xaxis=dict(title='Month', color='black', showgrid=False, zeroline=False),  # Light grey grid color for x-axis
            yaxis=dict(title='Expense', color='black', showgrid=False, zeroline=False),
            xaxis_tickangle=-45,
            showlegend=True,
            height=500,
            width=1120,
            paper_bgcolor='#e6e6e6',  # Dark grey background color
            plot_bgcolor='#e6e6e6',     # Dark grey plot background color
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
                'x': 0.0,
                'y': 0.95,
                'font': {
                    'color': '#252425',  # Set the title color to white
                    'size': 20
                    }
                },
            xaxis=dict(title='Month', color='black', showgrid=False, zeroline=False),  # Light grey grid color for x-axis
            yaxis=dict(title='Expense', color='black', showgrid=False, zeroline=False),
            xaxis_tickangle=-45,
            showlegend=True,
            paper_bgcolor='#e6e6e6',  # Dark grey background color
            plot_bgcolor='#e6e6e6',     # Dark grey plot background color
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
    st.markdown(f"<h5 style='text-align: left; color: black;'>{mrk_down}</h5>", unsafe_allow_html=True)
    # st.write(f"<u>Vessel Selected Count:</u> {filtered_result['COST_CENTER'].nunique()}")
    
    st.divider()

    filtered_result['DATE'] = pd.to_datetime(filtered_result['PERIOD'], format='%Y%m').dt.date
    filtered_result['YEAR'] = pd.to_datetime(filtered_result['DATE']).dt.year.astype('str')
   
    # import streamlit as st
    if page == "1. Report Page":
        st.markdown("<h3 style='text-align: center; color: blue;'><u>CATEGORY LEVEL BUDGET - PER MONTH </u></h3>", unsafe_allow_html=True)
        df_placeholder1 = st.empty()
        df = det_cat_data(filtered_result)
        df1 = df.copy()
        total_row = df.sum(numeric_only=True)
        total_row = total_row.round(2)
        total_row.name = 'Total'
        df = df.append(total_row.transpose())
        df_placeholder1.dataframe(df.style.highlight_max(axis=0, color='lightgrey'), use_container_width=True)
        # st.data_editor(df.style.highlight_max(axis=0, color='lightgrey'), use_container_width=True)
        
        st.markdown("<h3 style='text-align: center; color: blue;'><u>SUB-CATEGORY LEVEL BUDGET - PER MONTH </u></h3>", unsafe_allow_html=True)
        df_placeholder2 = st.empty()
        df = det_subcat_data(filtered_result)
        df2 = df.copy()
        total_row = df.sum(numeric_only=True)
        total_row = total_row.round(2)
        total_row.name = 'Total'
        df = df.append(total_row.transpose())
        df_placeholder2.dataframe(df.style.highlight_max(axis=0, color='lightgrey'), use_container_width=True)
        # st.data_editor(df.style.highlight_max(axis=0, color='lightgrey'), use_container_width=True)
        
        ### update with opt_cat_stat_model_value
        opt_cat = get_cat_optimal_mean(filtered_result)
        merged_df = pd.merge(df1, opt_cat, on='CATEGORIES')
        merged_df.set_index('CATEGORIES', inplace=True)
        total_row = merged_df.sum(numeric_only=True)
        total_row = total_row.round(2)
        total_row.name = 'Total'
        merged_df = merged_df.append(total_row.transpose())
        df_placeholder1.dataframe(merged_df.style.highlight_max(axis=0, color='lightgrey'), use_container_width=True)
        
        
        opt_subcat = get_subcat_optimal_mean(filtered_result)        
        df2.reset_index(inplace=True)
        # df2['index'] = df.apply(lambda row: tuple(str(x) for x in row), axis=1)
        df2['SUB CATEGORIES'] = df2.apply(lambda row: "('{}', '{}', '{}')".format(str(row['CATEGORIES']), str(row['ACCOUNT_CODE']), str(row['SUB_CATEGORIES'])), axis=1)
        
        df2.set_index('SUB CATEGORIES', inplace=True)
        opt_subcat.set_index('SUB CATEGORIES', inplace=True)
        
        merged_df = df2.join(opt_subcat, how='inner').drop(columns=['CATEGORIES', 'ACCOUNT_CODE', 'SUB_CATEGORIES'])
        
        total_row = merged_df.sum(numeric_only=True)
        total_row = total_row.round(2)
        total_row.name = 'Total'
        merged_df = merged_df.append(total_row.transpose())
        df_placeholder2.dataframe(merged_df.style.highlight_max(axis=0, color='lightgrey'), use_container_width=True)
        
        # merged_df = pd.merge(df2, opt_subcat, left_index=True, right_index=True)
        
    elif page == "2. Trend Analysis":
        with st.container():
            st.markdown("<h2 style='text-align: center; color: blue;'><u>Trend Analysis </u></h2>", unsafe_allow_html=True)
            
            st.markdown("<h3 style='text-align: left; color: blue;'><u>1. Monthly Trend - PER MONTH </u></h3>", unsafe_allow_html=True)
            st.plotly_chart(plotly_monthly_quartiles(filtered_result), use_container_width=True)
            
            st.markdown("<h3 style='text-align: left; color: blue;'><u>2. Yearly - PER MONTH </u></h3>", unsafe_allow_html=True)
            st.plotly_chart(plotly_yearly_quartiles(filtered_result), use_container_width=True)
            
            st.markdown("<h3 style='text-align: left; color: blue;'><u>Yearly Trend - PER MONTH </u></h3>", unsafe_allow_html=True)
            st.pyplot(new_modified.get_optimal_mean(filtered_result))
            # st.plotly_chart(new_modified.get_optimal_mean(filtered_result), use_container_width=True)
            st.pyplot((whole_year.get_optimal_mean(filtered_result)))