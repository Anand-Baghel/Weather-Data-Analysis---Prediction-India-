import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
delhi = pd.read_csv('delhi_daily_temperature_1951_2024.csv')
kolkata = pd.read_csv('kolkata_daily_temperature_1951_2024.csv')

# Parse dates robustly
delhi['Date'] = pd.to_datetime(delhi['Date'], dayfirst=True, errors='coerce')
kolkata['Date'] = pd.to_datetime(kolkata['Date'], dayfirst=True, errors='coerce')

# Check for rows with invalid dates
print("Delhi missing dates:", delhi['Date'].isna().sum())
print("Kolkata missing dates:", kolkata['Date'].isna().sum())

# Optionally drop rows with invalid dates
delhi = delhi.dropna(subset=['Date'])
kolkata = kolkata.dropna(subset=['Date'])

# Basic info
print('Delhi Data:')
print(delhi.info())
print(delhi.describe())
print('\nKolkata Data:')
print(kolkata.info())
print(kolkata.describe())

# Plotting temperature trends
plt.figure(figsize=(15, 5))
plt.plot(delhi['Date'], delhi['Temp Max'], label='Delhi Max Temp', alpha=0.7)
plt.plot(kolkata['Date'], kolkata['Temp Max'], label='Kolkata Max Temp', alpha=0.7)
plt.title('Daily Maximum Temperature (1951-2024)')
plt.xlabel('Year')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.tight_layout()
plt.show()

# Monthly average temperature trends
delhi['Month'] = delhi['Date'].dt.month
kolkata['Month'] = kolkata['Date'].dt.month

delhi_monthly = delhi.groupby('Month')[['Temp Max', 'Temp Min']].mean()
kolkata_monthly = kolkata.groupby('Month')[['Temp Max', 'Temp Min']].mean()

plt.figure(figsize=(12, 6))
plt.plot(delhi_monthly.index, delhi_monthly['Temp Max'], label='Delhi Max')
plt.plot(delhi_monthly.index, delhi_monthly['Temp Min'], label='Delhi Min')
plt.plot(kolkata_monthly.index, kolkata_monthly['Temp Max'], label='Kolkata Max')
plt.plot(kolkata_monthly.index, kolkata_monthly['Temp Min'], label='Kolkata Min')
plt.title('Monthly Average Temperatures (1951-2024)')
plt.xlabel('Month')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.tight_layout()
plt.show()

# Seasonality heatmap for Delhi
delhi['Year'] = delhi['Date'].dt.year
pivot = delhi.pivot_table(index='Year', columns='Month', values='Temp Max')
plt.figure(figsize=(12, 8))
sns.heatmap(pivot, cmap='coolwarm', cbar_kws={'label': 'Max Temp (°C)'})
plt.title('Delhi Max Temperature Seasonality (Heatmap)')
plt.xlabel('Month')
plt.ylabel('Year')
plt.tight_layout()
plt.show()
