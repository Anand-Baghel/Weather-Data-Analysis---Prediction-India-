import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Weather Data Analysis & Prediction (India)", layout="wide")
st.title("Weather Data Analysis & Prediction (India)")

CITY_FILES = {
    'Delhi': 'delhi_daily_temperature_1951_2024.csv',
    'Kolkata': 'kolkata_daily_temperature_1951_2024.csv',
    'Mumbai': 'mumbai_daily_temperature_1951_2024.csv',  # Add your new city here
    # Add more cities as needed
}

@st.cache_data
def load_data(city):
    df = pd.read_csv(CITY_FILES[city])
    # Coerce invalid dates to NaT
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    # Drop rows with invalid dates
    df = df.dropna(subset=['Date'])
    return df

def create_lag_features(df, target_col, lags=7):
    for lag in range(1, lags+1):
        df[f'{target_col}_lag{lag}'] = df[target_col].shift(lag)
    return df

def train_model(df):
    df = create_lag_features(df, 'Temp Max', lags=7)
    df = df.dropna().reset_index(drop=True)
    feature_cols = [f'Temp Max_lag{i}' for i in range(1, 8)]
    X = df[feature_cols]
    y = df['Temp Max']
    model = LinearRegression()
    model.fit(X, y)
    return model, df

def predict_next_7_days(model, df):
    last_row = df.iloc[-1]
    last_lags = [last_row[f'Temp Max_lag{i}'] for i in range(1, 8)]
    preds = []
    for i in range(7):
        pred = model.predict([last_lags])[0]
        preds.append(pred)
        last_lags = [pred] + last_lags[:-1]
    return np.round(preds, 2)

city = st.sidebar.selectbox('Select City', list(CITY_FILES.keys()))
df = load_data(city)
st.write(f"### Showing data for {city}")
st.dataframe(df.tail(10))

# EDA: Plot daily max temperature
tab1, tab2, tab3 = st.tabs(["Trend", "Monthly Avg", "Seasonality Heatmap"])

with tab1:
    st.subheader("Daily Maximum Temperature Trend")
    fig, ax = plt.subplots(figsize=(15, 4))
    ax.plot(df['Date'], df['Temp Max'], label=f'{city} Max Temp', alpha=0.7)
    ax.set_xlabel('Year')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title(f'{city} Daily Maximum Temperature (1951-2024)')
    st.pyplot(fig)

with tab2:
    st.subheader("Monthly Average Temperatures")
    df['Month'] = df['Date'].dt.month
    monthly = df.groupby('Month')[['Temp Max', 'Temp Min']].mean()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(monthly.index, monthly['Temp Max'], label='Max Temp')
    ax.plot(monthly.index, monthly['Temp Min'], label='Min Temp')
    ax.set_xlabel('Month')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title(f'{city} Monthly Average Temperatures')
    ax.legend()
    st.pyplot(fig)

with tab3:
    st.subheader("Seasonality Heatmap (Max Temp)")
    df['Year'] = df['Date'].dt.year
    pivot = df.pivot_table(index='Year', columns='Month', values='Temp Max')
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.heatmap(pivot, cmap='coolwarm', cbar_kws={'label': 'Max Temp (°C)'}, ax=ax)
    ax.set_title(f'{city} Max Temperature Seasonality (Heatmap)')
    st.pyplot(fig)

# Prediction
df_model = create_lag_features(df.copy(), 'Temp Max', lags=7).dropna().reset_index(drop=True)
model, df_model = train_model(df)
preds = predict_next_7_days(model, df_model)
st.header(f"Next 7 Days Predicted Max Temperatures for {city}")
pred_dates = pd.date_range(df['Date'].max() + pd.Timedelta(days=1), periods=7)
pred_df = pd.DataFrame({'Date': pred_dates, 'Predicted Max Temp (°C)': preds})
st.table(pred_df)
