import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Helper to create lag features
def create_lag_features(df, target_col, lags=7):
    for lag in range(1, lags+1):
        df[f'{target_col}_lag{lag}'] = df[target_col].shift(lag)
    return df

# Load and preprocess data
def load_city_data(filename):
    df = pd.read_csv(filename)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df = df.sort_values('Date')
    df = create_lag_features(df, 'Temp Max', lags=7)
    df = df.dropna().reset_index(drop=True)
    return df

def train_and_predict(df, city_name):
    # Features: last 7 days' max temp
    feature_cols = [f'Temp Max_lag{i}' for i in range(1, 8)]
    X = df[feature_cols]
    y = df['Temp Max']
    # Train/test split (last year for test)
    split_idx = int(len(df) * 0.9)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    # Model
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f'{city_name} RMSE: {rmse:.2f} °C')
    # Plot
    plt.figure(figsize=(12, 5))
    plt.plot(df['Date'].iloc[split_idx:], y_test, label='Actual')
    plt.plot(df['Date'].iloc[split_idx:], y_pred, label='Predicted')
    plt.title(f'{city_name} - Actual vs Predicted Max Temperature')
    plt.xlabel('Date')
    plt.ylabel('Max Temperature (°C)')
    plt.legend()
    plt.tight_layout()
    plt.show()
    # Predict next 7 days
    last_row = df.iloc[-1]
    last_lags = [last_row[f'Temp Max_lag{i}'] for i in range(1, 8)]
    preds = []
    for i in range(7):
        pred = model.predict([last_lags])[0]
        preds.append(pred)
        # Update lags for next day
        last_lags = [pred] + last_lags[:-1]
    print(f'Next 7 days predicted max temperatures for {city_name}:')
    print(np.round(preds, 2))
    return model

if __name__ == '__main__':
    for city, file in [('Delhi', 'delhi_daily_temperature_1951_2024.csv'),
                       ('Kolkata', 'kolkata_daily_temperature_1951_2024.csv')]:
        print(f'\n--- {city} ---')
        df = load_city_data(file)
        train_and_predict(df, city)
