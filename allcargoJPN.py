import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from statsmodels.tsa.seasonal import seasonal_decompose
import plotly.graph_objects as go

# Load the data and cache for faster loading
@st.cache_data
def load_data():
    data = pd.read_csv('ACLL_Historical_Data.csv')
    return data

# Load and process data
data = load_data()

# Convert Date column to datetime
data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y')

# Function to convert the 'Vol.' column to numeric
def convert_volume(vol_str):
    if 'M' in vol_str:
        return float(vol_str.replace('M', '').strip()) * 1_000_000
    elif 'K' in vol_str:
        return float(vol_str.replace('K', '').strip()) * 1_000
    else:
        return float(vol_str.strip())  # In case there is no suffix

# Apply the conversion function
data['Vol.'] = data['Vol.'].apply(convert_volume)

# Convert 'Change %' column to float
data['Change %'] = data['Change %'].replace('%', '', regex=True).astype(float)

# Set Date as index
data.set_index('Date', inplace=True)

# Static Data Display
st.title('Stock Price Analysis of Allcargo Logistics')
st.write("### Dataset Summary")

# Display basic information about the dataset
st.write(f"**Number of Rows:** {data.shape[0]}")
st.write(f"**Number of Columns:** {data.shape[1]}")
st.write("### First 5 Rows")
st.write(data.head())
st.write("### Last 5 Rows")
st.write(data.tail())
st.write("### Statistical Summary")
st.write(data.describe())

# Dropdown for selecting analysis type
st.write("### Select Analysis Type")
analysis_type = st.selectbox(
    'Select analysis type:',
    ('Price Trend Over Time', 'Price vs Open', 'Price Distribution', 'Moving Average', 'Correlation Heatmap', 'Boxplot')
)

# Interactive chart based on selected analysis type
if analysis_type == 'Price Trend Over Time':
    st.subheader('Price Trend Over Time')

    # Add sliders for zooming
    start_date = st.date_input('Start date', min_value=data.index.min(), max_value=data.index.max(), value=data.index.min())
    end_date = st.date_input('End date', min_value=data.index.min(), max_value=data.index.max(), value=data.index.max())

    filtered_data = data[(data.index >= pd.to_datetime(start_date)) & (data.index <= pd.to_datetime(end_date))]

    # Create interactive plot using Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=filtered_data.index, y=filtered_data['Price'], mode='lines', name='Price'))
    fig.update_layout(
        title='Price Trend Over Time',
        xaxis_title='Date',
        yaxis_title='Price',
        hovermode='closest',
        template='plotly_dark',
    )

    st.plotly_chart(fig)

elif analysis_type == 'Price vs Open':
    st.subheader('Price vs Open Price')

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(data['Open'], data['Price'], color='red', alpha=0.5)
    ax.set_title('Price vs. Open')
    ax.set_xlabel('Open Price')
    ax.set_ylabel('Price')

    st.pyplot(fig)

elif analysis_type == 'Price Distribution':
    st.subheader('Price Distribution')

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(data['Price'], bins=50, color='purple', edgecolor='black')
    ax.set_title('Price Distribution')
    ax.set_xlabel('Price')
    ax.set_ylabel('Frequency')

    st.pyplot(fig)

elif analysis_type == 'Moving Average':
    st.subheader('Price and 50-day Moving Average')

    # Calculate and fill NaN values for the moving average
    data['SMA_50'] = data['Price'].rolling(window=50).mean()
    data['SMA_50'] = data['SMA_50'].fillna(data['SMA_50'].mean())  # Fill NaN values with the mean

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data.index, data['Price'], label='Price', color='blue', alpha=0.5)
    ax.plot(data.index, data['SMA_50'], label='50-day Moving Average', color='orange', linewidth=2)
    ax.set_title('Price and 50-day Moving Average')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.grid()
    ax.legend()

    st.pyplot(fig)

elif analysis_type == 'Correlation Heatmap':
    st.subheader('Correlation Heatmap')

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap="coolwarm", cbar=True, ax=ax)
    ax.set_title('Feature Correlation Heatmap')

    st.pyplot(fig)

elif analysis_type == 'Boxplot':
    st.subheader('Box Plot for Price, Open, High, Low')

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot([data['Price'], data['Open'], data['High'], data['Low']], tick_labels=['Price', 'Open', 'High', 'Low'])
    ax.set_title('Box Plot for Price, Open, High, Low')
    ax.set_ylabel('Value')

    st.pyplot(fig)
