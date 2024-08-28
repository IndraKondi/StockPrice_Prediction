import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta
import requests

# Load the model
model = load_model('Stock Predictions Model.keras')

st.header('Stock Market Predictor')

# Fetch S&P 500 companies list dynamically
@st.cache_data
def get_sp500_companies():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    html = requests.get(url).text
    df = pd.read_html(html)[0]
    return df

sp500_companies = get_sp500_companies()
companies = dict(zip(sp500_companies['Security'], sp500_companies['Symbol']))

# Function to fetch stock data and calculate average return
@st.cache_data
def get_stock_performance(symbol):
    start_date = "2020-01-01"
    end_date = datetime.today().strftime('%Y-%m-%d')
    stock_data = yf.download(symbol, start=start_date, end=end_date)
    stock_data['Return'] = stock_data['Close'].pct_change()
    avg_return = stock_data['Return'].mean() * 252  # Annualize return
    return avg_return

# Caching the performance data calculation
@st.cache_data
def get_performance_data(companies):
    performance_data = []
    for company, symbol in companies.items():
        avg_return = get_stock_performance(symbol)
        performance_data.append((company, avg_return))
    performance_df = pd.DataFrame(performance_data, columns=['Company', 'Avg Annual Return'])
    return performance_df

performance_df = get_performance_data(companies)

# Company selection
stock_name = st.selectbox('Select Company', list(companies.keys()))
stock = companies[stock_name]

# Fetch data from Yahoo Finance
@st.cache_data
def fetch_data(stock):
    data = yf.download(stock, start="2012-01-01", end=datetime.today().strftime('%Y-%m-%d'))
    data = data[::-1]  # Reverse the data order
    return data

data = fetch_data(stock)

# Convert datetime index to just date and reset index
data.reset_index(inplace=True)
data['Date'] = data['Date'].dt.date
data.set_index('Date', inplace=True)

st.subheader('Stock Data')
st.write(data)

# Prepare data for prediction
data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])

scaler = MinMaxScaler(feature_range=(0,1))
pas_100_days = data_train.tail(100)
data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)

# Plot Moving Averages with interactive features
st.subheader('Moving Averages and Stock Prices')

fig1, ax1 = plt.subplots(figsize=(10,6))
ma_50_days = data.Close.rolling(50).mean()
ma_100_days = data.Close.rolling(100).mean()
ma_200_days = data.Close.rolling(200).mean()

ax1.plot(data.index, data.Close, label='Close Price', color='g', alpha=0.5)
ax1.plot(data.index, ma_50_days, label='MA50', color='r')
ax1.plot(data.index, ma_100_days, label='MA100', color='b')
ax1.plot(data.index, ma_200_days, label='MA200', color='orange')
ax1.set_xlabel('year')
ax1.set_ylabel('Price (USD)')
ax1.set_title('Stock Price with Moving Averages')
ax1.legend()

# Format y-axis labels as currency
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.2f}'))
st.pyplot(fig1)

# Prepare data for prediction
x = []
y = []

for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i,0])

x, y = np.array(x), np.array(y)

# Make predictions
predict = model.predict(x)

scale = scaler.scale_[0]
predict = predict * scale
y = y * scale

# Calculate accuracy metrics
mse = mean_squared_error(y, predict)
rmse = np.sqrt(mse)

# Calculate accuracy as a percentage
mean_actual = np.mean(y)
accuracy_percentage = 100 - (rmse / mean_actual * 100)

# Display accuracy metrics
st.subheader('Model Accuracy')
st.write(f"Mean Squared Error (MSE): ${mse:.2f}")
st.write(f"Root Mean Squared Error (RMSE): ${rmse:.2f}")
st.write(f"Model Accuracy: {accuracy_percentage:.2f}%")

# Plot predictions with enhanced user interaction
st.subheader('Prediction vs Actual Prices')

fig2, ax2 = plt.subplots(figsize=(10,6))
ax2.plot(data.index[-len(predict):], predict, label='Predicted Price', color='r')
ax2.plot(data.index[-len(y):], y, label='Original Price', color='g')
ax2.set_xlabel('Date')
ax2.set_ylabel('Price (USD)')
ax2.set_title('Original vs Predicted Prices')
ax2.legend()

# Format y-axis labels as currency
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.2f}'))
st.pyplot(fig2)

# Predicting for the next 10 days
last_100_days = data.Close.tail(100)
last_100_days_scaled = scaler.transform(last_100_days.values.reshape(-1, 1))

future_x = []
current_step = last_100_days_scaled

for _ in range(10):
    future_x.append(current_step)
    next_step = model.predict(current_step.reshape(1, 100, 1))
    current_step = np.append(current_step[1:], next_step, axis=0)

future_x = np.array(future_x).reshape(10, 100, 1)
future_predictions = model.predict(future_x)
future_predictions = future_predictions * scale

# Calculate future dates starting from today
today = datetime.today().date()
future_dates = [today + timedelta(days=i) for i in range(1, 11)]

# Create a DataFrame for future predictions
future_df = pd.DataFrame({
    'Date': future_dates,
    'Predicted Price (USD)': future_predictions.flatten()
})

# Format the 'Predicted Price' column as USD
future_df['Predicted Price (USD)'] = future_df['Predicted Price (USD)'].apply(lambda x: f'${x:,.6f}')

# Format the 'Date' column
future_df['Date'] = future_df['Date'].astype(str)

# Plot future predictions
st.subheader('Future Price Predictions (Next 10 Days)')

fig3, ax3 = plt.subplots(figsize=(10,6))
ax3.plot(future_dates, future_predictions, label='Predicted Future Price', color='b')
ax3.set_xlabel('Date')
ax3.set_ylabel('Price (USD)')
ax3.setTitle('Future Stock Price Predictions')
ax3.legend()
ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.6f}'))
st.pyplot(fig3)

# Display future predictions in a table
st.subheader('Future Price Predictions Table')
st.write(future_df)

# Get top N companies based on user input
st.sidebar.header('Top 100 Companies to Invest')
top_n = st.sidebar.selectbox('Select Top N Companies', [10, 20, 30, 50, 100])

top_companies = performance_df.sort_values(by='Avg Annual Return', ascending=False).head(top_n)

# Display top companies in the sidebar as a table
st.sidebar.subheader(f'Top {top_n} Companies to Invest')
st.sidebar.table(top_companies[['Company', 'Avg Annual Return']].style.format({
    'Avg Annual Return': '${:,.3f}'
}))
