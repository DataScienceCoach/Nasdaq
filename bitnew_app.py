import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import plotly.graph_objs as go
from datetime import timedelta
from streamlit_option_menu import option_menu

# Set page configuration
st.set_page_config(page_title="Trade Assistant", layout="wide", page_icon="🧑‍⚕️")

# Load the models
nas_model = load_model('nasdaq_prediction_model.h5')
btc_model = load_model('btcprice_prediction_model.h5')

# Sidebar for navigation
with st.sidebar:
    selected = option_menu('Double Asset Prediction System',
                           ['Nasdaq100 Next Day Prediction',
                            'Bitcoin Next Day Prediction'],
                           menu_icon='hospital-fill',
                           icons=['activity', 'activity'],
                           default_index=0)

# Function to fetch historical data
def fetch_data(ticker, period='70d'):
    data = yf.download(ticker, period=period, interval='1d')
    return data

# Function to display Nasdaq100 Prediction Page
def nasdaq_prediction_page():
    st.title('GM Analytics Nasdaq100 Next Day Prediction')

    if st.button('Fetch Latest Data and Predict'):
        data = fetch_data(ticker='^NDX')
        if not data.empty:
            data.dropna(inplace=True)
            st.write('Fetched data:')
            st.dataframe(data.tail(60))

            if len(data) < 60:
                st.error('Insufficient data. Ensure that you have at least 60 days of data.')
                return

            close_prices = data[['Close']]
            last_date = data.index[-1]

            try:
                last_60_days = close_prices[-60:].values
                scaler = MinMaxScaler()
                last_60_days_scaled = scaler.fit_transform(last_60_days)

                X_test = np.array([last_60_days_scaled])
                X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

                pred_price_scaled = nas_model.predict(X_test)
                pred_price = scaler.inverse_transform(pred_price_scaled)

                next_day = last_date + timedelta(days=1)
                st.success(f'Predicted Nasdaq100 Close Price for {next_day.date()}: ${pred_price[0, 0]:.2f}')

                # Load the data for plotting
                data = pd.read_csv('data2.csv')
                data['Date'] = pd.to_datetime(data['Date'])
                data.set_index('Date', inplace=True)

                scaled_data = scaler.fit_transform(data[['Close']])
                sequence_length = 60
                training_data_len = int(len(data) * 0.8)
                train_data = scaled_data[:training_data_len]
                test_data = scaled_data[training_data_len - sequence_length:]

                x_test, y_test = [], []
                for i in range(sequence_length, len(test_data)):
                    x_test.append(test_data[i - sequence_length:i, 0])
                    y_test.append(test_data[i, 0])

                x_test = np.array(x_test)
                y_test = np.array(y_test)
                x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

                predictions = nas_model.predict(x_test)
                predictions = scaler.inverse_transform(predictions)
                y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

                train = data[:training_data_len]
                valid = data[training_data_len:].copy()
                valid['Predictions'] = predictions

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=train.index, y=train['Close'], name='Train'))
                fig.add_trace(go.Scatter(x=valid.index, y=valid['Close'], name='Val'))
                fig.add_trace(go.Scatter(x=valid.index, y=valid['Predictions'], name='Predictions'))

                fig.update_layout(
                    title='LSTM Model - Stock: Nasdaq100 - Trained Model',
                    xaxis=dict(title='Date', showgrid=True, gridcolor='lightgray'),
                    yaxis=dict(title='Nasdaq100 Close Price USD ($)', showgrid=True, gridcolor='lightgray'),
                    legend=dict(x=0.01, y=0.99, bordercolor='black', borderwidth=1),
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    showlegend=True,
                )
                st.plotly_chart(fig)

            except Exception as e:
                st.error(f'An error occurred during prediction: {str(e)}')
        else:
            st.error('Failed to fetch data from Yahoo Finance.')

# Function to display Bitcoin Prediction Page
def bitcoin_prediction_page():
    st.title('GM Analytics Bitcoin Next Day Prediction')

    if st.button('Fetch Latest Data and Predict'):
        data = fetch_data(ticker='BTC-USD')
        if not data.empty:
            data.dropna(inplace=True)
            st.write('Fetched data:')
            st.dataframe(data.tail(60))

            if len(data) < 60:
                st.error('Insufficient data. Ensure that you have at least 60 days of data.')
                return

            close_prices = data[['Close']]
            last_date = data.index[-1]

            try:
                last_60_days = close_prices[-60:].values
                scaler = MinMaxScaler()
                last_60_days_scaled = scaler.fit_transform(last_60_days)

                X_test = np.array([last_60_days_scaled])
                X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

                pred_price_scaled = btc_model.predict(X_test)
                pred_price = scaler.inverse_transform(pred_price_scaled)

                next_day = last_date + timedelta(days=1)
                st.success(f'Predicted Bitcoin Price for {next_day.date()}: ${pred_price[0, 0]:.2f}')

                # Load the data for plotting
                data = pd.read_csv('data.csv')
                data['Date'] = pd.to_datetime(data['Date'])
                data.set_index('Date', inplace=True)

                scaled_data = scaler.fit_transform(data[['Close']])
                sequence_length = 60
                training_data_len = int(len(data) * 0.8)
                train_data = scaled_data[:training_data_len]
                test_data = scaled_data[training_data_len - sequence_length:]

                x_test, y_test = [], []
                for i in range(sequence_length, len(test_data)):
                    x_test.append(test_data[i - sequence_length:i, 0])
                    y_test.append(test_data[i, 0])

                x_test = np.array(x_test)
                y_test = np.array(y_test)
                x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

                predictions = btc_model.predict(x_test)
                predictions = scaler.inverse_transform(predictions)
                y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

                train = data[:training_data_len]
                valid = data[training_data_len:].copy()
                valid['Predictions'] = predictions

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=train.index, y=train['Close'], name='Train'))
                fig.add_trace(go.Scatter(x=valid.index, y=valid['Close'], name='Val'))
                fig.add_trace(go.Scatter(x=valid.index, y=valid['Predictions'], name='Predictions'))

                fig.update_layout(
                    title='LSTM Model - Crypto: Bitcoin - Trained Model',
                    xaxis=dict(title='Date', showgrid=True, gridcolor='lightgray'),
                    yaxis=dict(title='Bitcoin Close Price USD ($)', showgrid=True, gridcolor='lightgray'),
                    legend=dict(x=0.01, y=0.99, bordercolor='black', borderwidth=1),
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    showlegend=True,
                )
                st.plotly_chart(fig)

            except Exception as e:
                st.error(f'An error occurred during prediction: {str(e)}')
        else:
            st.error('Failed to fetch data from Yahoo Finance.')

# Main function to handle page selection
def main():
    if selected == 'Nasdaq100 Next Day Prediction':
        nasdaq_prediction_page()
    elif selected == 'Bitcoin Next Day Prediction':
        bitcoin_prediction_page()

if __name__ == '__main__':
    main()

