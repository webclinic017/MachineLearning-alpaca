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
import neptune.new as neptune
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import LSTM, Dense
import tensorflow_model_optimization as tfmot

load_dotenv()
NEPTUNE_API_TOKEN = os.getenv('NEPTUNE-API-TOKEN')
ALPHA_VANTAGE_TOKEN = os.getenv('ALPHA-VANTAGE-API-TOKEN')


class Stock:

    def __init__(self, ticker, run, test_ratio=0.2, dataset_size=200, test_lstm=False):
        self.test_lstm = test_lstm
        self.dataset_size = dataset_size
        self.run = run
        self.ticker = ticker.strip()
        self.data = self.get_new_data()
        self.flippedData = self.data.copy().loc[::-1].reset_index(drop=True)
        self.test_ratio = test_ratio
        self.train_ratio = 1 - test_ratio
        self.train_size = int(self.train_ratio * len(self.data))
        self.test_size = int(test_ratio * len(self.data))
        self.train = self.data[:self.train_size]
        self.test = self.data[self.train_size:]

        if not test_lstm:
            self.begin()

    def begin(self):
        # stockprices = self.extract_seqX_outcomeY(self.train['close'], 50, 60)
        # X = stockprices[0]
        # Y = stockprices[1].reshape((len(stockprices[1]), 1))
        # stockprices = (X, Y)

        self.simple_moving_average(50, stockprices=self.flippedData)
        self.exponential_moving_average(50, stockprices=self.flippedData)

        self.train = self.flippedData[:self.train_size]
        self.test = self.flippedData[self.train_size:]

        self.handle_lstm()

    def begin_lstm(self, lstm_pars, plot=False):
        pandas.options.mode.chained_assignment = None  # default='warn'
        matplotlib.use('SVG')
        warnings.filterwarnings(action='ignore', category=UserWarning)

        self.simple_moving_average(50, stockprices=self.flippedData, plot=plot)
        self.exponential_moving_average(50, stockprices=self.flippedData, plot=plot)

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
        # compress the model to be smaller
        # converter = tf.lite.TFLiteConverter.from_keras_model(model_for_export)
        # converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # quantized_and_pruned_tflite_model = converter.convert()

        model.fit(x_train, y_train, epochs=tf.constant(cur_epochs, dtype="int64"), batch_size=tf.constant(cur_batch_size, dtype="int64"), verbose=0, validation_split=0.1,
                  shuffle=True)

        x_test = self.preprocess_testdat(data=self.flippedData, scaler=scaler, window_size=window_size, test=self.test)
        predicted_price_ = model.predict(x_test)
        predicted_price = scaler.inverse_transform(predicted_price_)
        self.test['Predictions_lstm'] = predicted_price

        mape_lstm = self.calculate_mape(np.array(self.test['close']), np.array(self.test['Predictions_lstm']))

        return mape_lstm

    def get_new_data(self):
        self.get_new_data_to_file()
        return self.read_data_from_file()

    def get_new_data_to_file(self):
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={self.ticker}&apikey={ALPHA_VANTAGE_TOKEN}&datatype=csv&outputsize=full"

        try:
            response = requests.get(url)
        except Exception as e:
            print(e)
            exit(1)

        lines = response.text.splitlines()
        if self.dataset_size < len(lines):
            lines = lines[:self.dataset_size]
        reader = csv.reader(lines)

        f = open(f"Data/{self.ticker}_data.csv", 'w')
        csv_writer = csv.writer(f)

        for row in reader:
            csv_writer.writerow(row)
        f.close()

    def read_data_from_file(self):
        if Path(f"Data/{self.ticker}_data.csv").is_file():
            return pandas.read_csv(f"Data/{self.ticker}_data.csv")
        else:
            print(f"No file: Data/{self.ticker}_data.csv, getting data")
            self.get_new_data_to_file()
            return pandas.read_csv(f"Data/{self.ticker}_data.csv")

    def extract_seqX_outcomeY(self, data, N, offset):
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

    def calculate_rmse(self, y_true, y_pred):

        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

        return rmse

    def calculate_mape(self, y_true, y_pred):

        y_true, y_pred = np.array(y_true), np.array(y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        return mape

    def calculate_perf_metrics(self, var, stockprices, logNeptune=True, logmodelName='Simple MA', pathName=''):
        rmse = self.calculate_rmse(np.array(stockprices[self.train_size:]['close']),
                                   np.array(stockprices[self.train_size:][var]))

        mape = self.calculate_mape(np.array(stockprices[self.train_size:]['close']),
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

    def simple_moving_average(self, window_size, stockprices, plot=True):

        window_var = str(window_size) + 'day'

        stockprices[window_var] = stockprices[['close']].rolling(window_size).mean()
        stockprices['200day'] = stockprices[['close']].rolling(200).mean()

        if plot:
            self.plot_stock_trend(var=window_var, cur_title='Simple Moving Averages', stockprices=stockprices,
                                  logmodelName='Simple MA', pathName='SMA')

        self.calculate_perf_metrics(var=window_var, stockprices=stockprices, logmodelName='Simple MA', pathName='SMA')

    def exponential_moving_average(self, window_size, stockprices, plot=True):
        window_var = str(window_size) + 'day_EMA'

        stockprices[window_var] = stockprices['close'].ewm(span=window_size, adjust=False).mean()
        stockprices['200day'] = stockprices['close'].rolling(200).mean()

        if plot:
            self.plot_stock_trend(var=window_var, cur_title='Exponential Moving Averages', stockprices=stockprices,
                                  logmodelName='Exp MA', pathName='EMA')
        self.calculate_perf_metrics(var=window_var, stockprices=stockprices, logmodelName='Exp MA', pathName='EMA')

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

        x_test = self.preprocess_testdat(data=self.flippedData, scaler=scaler, window_size=window_size, test=self.test)
        predicted_price_ = model.predict(x_test)
        predicted_price = scaler.inverse_transform(predicted_price_)
        self.test['Predictions_lstm'] = predicted_price

        rmse_lstm = self.calculate_rmse(np.array(self.test['close']), np.array(self.test['Predictions_lstm']))
        mape_lstm = self.calculate_mape(np.array(self.test['close']), np.array(self.test['Predictions_lstm']))

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

        x_train, y_train = self.extract_seqX_outcomeY(scaled_data_train, window_size, window_size)
        return x_train, y_train

    def run_lstm(self, x_train, layer_units=50, logNeptune=True, NeptuneProject=None):
        inp = Input(shape=(x_train.shape[1], 1))

        x = LSTM(units=layer_units, return_sequences=True)(inp)
        x = LSTM(units=layer_units)(x)

        out = Dense(1, activation='linear')(x)
        model = Model(inp, out)

        model.compile(loss='mean_squared_error', optimizer='adam')

        if logNeptune:
            model.summary(print_fn=lambda z: NeptuneProject[f"{self.ticker}/LSTM/model_summary"].log(z))

        return model

    # this is attempting to be a more optimized function for training models but wtf is tf
    def run_lstm_tensor(self, x_train, pruning_params, layer_units=50, logNeptune=True, NeptuneProject=None):
        x_train = tf.convert_to_tensor(x_train)

        inp = Input(shape=(x_train.shape[1], 1))
        x = LSTM(units=layer_units, return_sequences=True)(inp)
        x = LSTM(units=layer_units)(x)
        out = Dense(1, activation='linear')(x)

        model_for_pruning = Model(inp, out)

        prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

        model = prune_low_magnitude(model_for_pruning, **pruning_params)

        model.compile(loss='mean_squared_error', optimizer='adam')

        if logNeptune:
            model.summary(print_fn=lambda z: NeptuneProject[f"{self.ticker}/LSTM/model_summary"].log(z))

        return model

    def preprocess_testdat(self, data, scaler, window_size, test):
        raw = data['close'][len(data) - len(test) - window_size:].values
        raw = raw.reshape(-1, 1)
        raw = scaler.transform(raw)

        x_test = []

        for i in range(window_size, raw.shape[0]):
            x_test.append(raw[i - window_size:i, 0])

        x_test = np.array(x_test)

        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        return x_test

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


def start(ticker, run):
    Stock(ticker, run)


if __name__ == '__main__':
    pandas.options.mode.chained_assignment = None  # default='warn'
    matplotlib.use('SVG')
    warnings.filterwarnings(action='ignore', category=UserWarning)

    dateTimeObj = datetime.now()
    custom_id = 'EXP-' + dateTimeObj.strftime("%d-%b-%Y-(%H:%M:%S)")

    run = neptune.init(
        project="elitheknight/Stock-Prediction",
        api_token=NEPTUNE_API_TOKEN,
        custom_run_id=custom_id
    )
    blacklist = ['AMZN', 'GOOG', 'NVDA', 'TSLA', 'NKE']
    # max to run at once is 20 because of pandas figure limits
    listoftickers = ['NKE', 'TMO']
    threads = []
    get_new_data = True
    for ticker_ in listoftickers:
        if ticker_ in blacklist:
            continue
        threads.append(threading.Thread(target=start, args=(ticker_, run)))

    for thread in threads:
        thread.start()
        if get_new_data:
            time.sleep(20)

    for thread in threads:
        thread.join()

    run.stop()
