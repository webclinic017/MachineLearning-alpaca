import threading
import requests
import numpy as np
import pandas
import csv
from dotenv import load_dotenv
from pathlib import Path
import os
import time
from datetime import datetime
import warnings
import neptune
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import LSTM, Dense

load_dotenv()
NEPTUNE_API_TOKEN = os.getenv('NEPTUNE-API-TOKEN')
ALPHA_VANTAGE_TOKEN = os.getenv('ALPHA-VANTAGE-API-TOKEN')


def preprocess_testdat(data, scaler, window_size, test):
    raw = data['close'][len(data) - len(test) - window_size:].values
    raw = raw.reshape(-1, 1)
    raw = scaler.transform(raw)

    x_test = []

    for i in range(window_size, raw.shape[0]):
        x_test.append(raw[i - window_size:i, 0])

    x_test = np.array(x_test)

    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    return x_test


def extract_seqX_outcomeY(data, N, offset):
    """
    Split time-series into training sequence X and outcome value Y
    Args:
        data - dataset
        N - window size, e.g., 50 for 50 days of historical stock prices
       offset - position to start the split
    """
    X, y = [], []

    for i in range(offset, len(data)):
        X.append(data[i - N:i])
        y.append(data[i])

    return np.array(X), np.array(y)


def calculate_rmse(y_true, y_pred):

    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    return rmse


def calculate_mape(y_true, y_pred):

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    return mape


class Stock:

    def __init__(self, ticker, run, data, dataset_size: int = 400, test_ratio: float = None):
        self.dataset_size = dataset_size
        self.run = run
        self.ticker = ticker.strip()
        self.flippedData = data.copy().loc[::-1].reset_index(drop=True)[:dataset_size]
        self.test_ratio = test_ratio if test_ratio is not None else 1/dataset_size
        self.train_ratio = 1 - test_ratio
        self.train_size = int(self.train_ratio * dataset_size)
        self.test_size = int(test_ratio * dataset_size)
        self.train = self.flippedData[:self.train_size]
        self.test = self.flippedData[self.train_size:]

    def begin(self):

        self.train = self.flippedData[:self.train_size]
        self.test = self.flippedData[self.train_size:]

        self.handle_lstm()

    def begin_lstm(self, lstm_pars, plot=False):
        pandas.options.mode.chained_assignment = None  # default='warn'
        matplotlib.use('SVG')
        warnings.filterwarnings(action='ignore', category=UserWarning)

        self.train = self.flippedData[:self.train_size]
        self.test = self.flippedData[self.train_size:]

        cur_epochs = lstm_pars['cur_epochs']
        cur_batch_size = lstm_pars['cur_batch_size']
        window_size = lstm_pars['window_size']
        layer_units = lstm_pars['layer_units']
        scaler = StandardScaler()

        x_train, y_train = self.lstm_get_train_data(self.flippedData, scaler, cur_batch_size=cur_batch_size,
                                                    cur_epochs=cur_epochs, window_size=window_size)

        model = self.run_lstm(x_train, layer_units=layer_units, logNeptune=False)

        model.fit(x_train, y_train, epochs=tf.constant(cur_epochs, dtype="int64"), batch_size=tf.constant(cur_batch_size, dtype="int64"), verbose=0, validation_split=0.1,
                  shuffle=True)

        x_test = preprocess_testdat(data=self.flippedData, scaler=scaler, window_size=window_size, test=self.test)
        predicted_price_ = model.predict(x_test)
        predicted_price = scaler.inverse_transform(predicted_price_)
        self.test['Predictions_lstm'] = predicted_price

        mape_lstm = calculate_mape(np.array(self.test['close']), np.array(self.test['Predictions_lstm']))

        return mape_lstm

    def calculate_perf_metrics(self, var, stockprices, logNeptune=True, logmodelName='Simple MA', pathName=''):
        rmse = calculate_rmse(np.array(stockprices[self.train_size:]['close']),
                                   np.array(stockprices[self.train_size:][var]))

        mape = calculate_mape(np.array(stockprices[self.train_size:]['close']),
                                   np.array(stockprices[self.train_size:][var]))

        if logNeptune and not self.test_lstm:
            self.run[f"{self.ticker}/{pathName}/RMSE"].log(rmse)
            self.run[f"{self.ticker}/{pathName}/MAPE (%)"].log(mape)

        return rmse, mape

    def plot_stock_trend(self, var, cur_title, stockprices, logNeptune=True, logmodelName='Simple MA', pathName=''):
        ax = stockprices[['close', var, '200day']].plot(figsize=(20, 10))
        plt.grid(False)
        plt.title(cur_title)
        plt.axis('tight')
        plt.ylabel('Stock Price ($)')

        # plt.show()
        # Log images to Neptune new version
        if logNeptune and not self.test_lstm:
            self.run[f'{self.ticker}/{pathName}Plot of Stock Predictions with {logmodelName}'].upload(
                neptune.types.File.as_image(ax.get_figure()))

    def handle_lstm(self):
        cur_epochs = 15
        cur_batch_size = 20
        window_size = 50
        scaler = StandardScaler()

        x_train, y_train = self.lstm_get_train_data(self.flippedData, scaler, cur_batch_size=cur_batch_size,
                                                    cur_epochs=cur_epochs, window_size=window_size)

        model = self.run_lstm(x_train, NeptuneProject=self.run)
        model.fit(x_train, y_train, epochs=tf.constant(cur_epochs, dtype="int64"), batch_size=tf.constant(cur_batch_size, dtype="int64"), verbose=0, validation_split=0.1,
                  shuffle=True)

        x_test = preprocess_testdat(data=self.flippedData, scaler=scaler, window_size=window_size, test=self.test)
        predicted_price_ = model.predict(x_test)
        predicted_price = scaler.inverse_transform(predicted_price_)
        self.test['Predictions_lstm'] = predicted_price

        rmse_lstm = calculate_rmse(np.array(self.test['close']), np.array(self.test['Predictions_lstm']))
        mape_lstm = calculate_mape(np.array(self.test['close']), np.array(self.test['Predictions_lstm']))

        self.run[f"{self.ticker}/LSTM/RMSE"].log(rmse_lstm)
        self.run[f"{self.ticker}/LSTM/MAPE (%)"].log(mape_lstm)
        self.plot_stock_trend_lstm(self.train, self.test)

    def lstm_get_train_data(self, stockprices, scaler, layer_units=50, optimizer='adam', cur_epochs=15, cur_batch_size=20, window_size=50):

        if not self.test_lstm:
            cur_LSTM_pars = {'units': layer_units,
                             'optimizer': optimizer,
                             'batch_size': cur_batch_size,
                             'epochs': cur_epochs
                             }
            self.run['LSTMPars'] = cur_LSTM_pars

        scaled_data = scaler.fit_transform(stockprices[['close']])
        scaled_data_train = scaled_data[:self.train.shape[0]]

        x_train, y_train = extract_seqX_outcomeY(scaled_data_train, window_size, window_size)
        return x_train, y_train

    def run_lstm(self, x_train, layer_units=50, logNeptune=True, NeptuneProject=None):
        inp = Input(shape=(x_train.shape[1], 1))

        x = LSTM(units=layer_units, return_sequences=True)(inp)
        # x = LSTM(units=layer_units, return_sequences=True)(x)
        x = LSTM(units=layer_units)(x)

        out = Dense(1, activation='linear')(x)
        model = Model(inp, out)

        model.compile(loss='mean_squared_error', optimizer='adam')

        if logNeptune:
            model.summary(print_fn=lambda z: NeptuneProject[f"{self.ticker}/LSTM/model_summary"].log(z))

        return model

    def plot_stock_trend_lstm(self, train, test, logNeptune=True):
        fig = plt.figure(figsize=(20, 10))
        train_x = np.asarray(train['timestamp'], dtype='datetime64[s]')
        test_x = np.asarray(test['timestamp'], dtype='datetime64[s]')
        plt.plot(train_x, train['close'], label='Train Closing Price')
        plt.plot(test_x, test['close'], label='Test Closing Price')
        plt.plot(test_x, test['Predictions_lstm'], label='Predicted Closing Price')
        plt.title('LSTM Model')
        plt.xlabel('Date')
        plt.ylabel('Stock Price ($)')
        plt.legend(loc='upper left')

        if logNeptune:
            self.run[f"{self.ticker}/LSTM/LSTM Prediction Model"].upload(neptune.types.File.as_image(fig))
