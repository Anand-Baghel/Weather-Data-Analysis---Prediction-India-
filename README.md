# Weather Data Analysis and Prediction (India)

This project analyzes historical weather data and predicts future temperature trends for major Indian cities (Delhi and Kolkata) using time series regression and an interactive Streamlit app.

## Features
- Analyze and visualize daily temperature trends and seasonality
- Compare monthly averages and seasonality heatmaps
- Predict the next 7 days of maximum temperature using a regression model
- Interactive city selection (Delhi or Kolkata)

## Setup
1. **Clone or download this repository**
2. **Install dependencies** (recommended: use a virtual environment):
   ```bash
   pip install -r requirements.txt
   ```
3. **Ensure the following data files are present in the project directory:**
   - `delhi_daily_temperature_1951_2024.csv`
   - `kolkata_daily_temperature_1951_2024.csv`

## Usage
### Run the Streamlit app
```bash
streamlit run app.py
```
- Open the provided local URL in your browser
- Use the sidebar to select a city
- Explore EDA plots and view temperature predictions

### Run EDA or Model scripts (optional)
- For EDA: `python eda_weather.py`
- For model training/prediction: `python temperature_prediction_model.py`

## Data Source
- [OpenCity Urban Data Portal](https://data.opencity.in/dataset/daily-temperature-70-years-data-for-major-indian-cities)

## Requirements
- Python 3.8+
- See `requirements.txt` for package list

## License
This project is for educational and research purposes. Data is sourced from public domain resources.
