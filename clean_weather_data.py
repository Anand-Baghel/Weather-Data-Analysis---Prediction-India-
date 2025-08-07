import pandas as pd

def clean_weather_csv(filename):
    df = pd.read_csv(filename)
    # Try to parse dates, mark errors as NaT
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    # Remove rows where date could not be parsed
    df_clean = df.dropna(subset=['Date'])
    # Remove rows with '-----' in any column
    df_clean = df_clean[~df_clean.astype(str).apply(lambda x: x.str.contains('-----')).any(axis=1)]
    # Save back to file (overwrite)
    df_clean.to_csv(filename, index=False)
    print(f"Cleaned {filename}: {len(df) - len(df_clean)} rows removed.")

clean_weather_csv('delhi_daily_temperature_1951_2024.csv')
clean_weather_csv('kolkata_daily_temperature_1951_2024.csv')
