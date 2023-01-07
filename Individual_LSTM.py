import numpy as np
import neptune.new as neptune
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import LSTM, Dense


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


def calculate_change(last_value, predicted_value):
    """
    :param last_value: most recent stock price data point
    :param predicted_value: predicted stock price data point
    :return: a tuple of the price change and percent change (price, percent)
    """
    price = predicted_value - last_value
    percent = (predicted_value/last_value - 1) * 100

    return price, percent


def preprocess_testdat(data, scaler, window_size):
    """
    :param data: dataset
    :param scaler: StandardScaler
    :param window_size: # of previous days it uses to predict the next value
    :return: array of size (window_size, 1) to put into model.predict()
    """
    raw = data['close'][len(data) - window_size - 1:].values
    raw = raw.reshape(-1, 1)
    raw = scaler.transform(raw)

    x_test = []

    for i in range(window_size, raw.shape[0]):
        x_test.append(raw[i - window_size:i, 0])

    x_test = np.array(x_test)

    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    return x_test


class IndividualLSTM:

    def __init__(self, ticker, data, run):
        self.run = run
        self.ticker = ticker.strip()
        self.flippedData = data.copy().loc[::-1].reset_index(drop=True)
        # self.test_ratio = 0.2
        # self.train_ratio = 1 - self.test_ratio
        # self.train_size = int(self.train_ratio * len(data))
        # self.test_size = int(self.test_ratio * len(data))
        # self.train = self.flippedData[:self.train_size]
        # self.test = self.flippedData[self.train_size:]
        self.handle_lstm()

    def handle_lstm(self):
        cur_epochs = 15
        cur_batch_size = 20
        window_size = 50
        scaler = StandardScaler()

        x_train, y_train = self.lstm_get_train_data(self.flippedData, scaler, cur_batch_size=cur_batch_size, cur_epochs=cur_epochs, window_size=window_size)
        model = self.run_lstm(x_train, NeptuneProject=self.run)
        model.fit(x_train, y_train, epochs=tf.constant(cur_epochs, dtype="int64"), batch_size=tf.constant(cur_batch_size, dtype="int64"), verbose=0, validation_split=0.1, shuffle=True)
        x_test = preprocess_testdat(data=self.flippedData, scaler=scaler, window_size=window_size)

        predicted_price_array = model.predict(x_test)
        predicted_price_array = scaler.inverse_transform(predicted_price_array)

        # predicted price for the next day
        predicted_price = predicted_price_array[0][0]
        change_price, change_percent = calculate_change(self.flippedData.iloc[-1]['close'], predicted_price)

        self.run[f"Predictions/{self.ticker}/LSTM/Price"].log(predicted_price)
        self.run[f"Predictions/{self.ticker}/LSTM/Change (%)"].log(change_percent)
        self.run[f"Predictions/{self.ticker}/LSTM/Change ($)"].log(change_price)

    def lstm_get_train_data(self, stockprices, scaler, layer_units=50, optimizer='adam', cur_epochs=15, cur_batch_size=20, window_size=50):

        cur_LSTM_pars = {'units': layer_units,
                         'optimizer': optimizer,
                         'batch_size': cur_batch_size,
                         'epochs': cur_epochs
                         }
        self.run['LSTMPars'] = cur_LSTM_pars

        scaled_data = scaler.fit_transform(stockprices[['close']].values)
        scaled_data_train = scaled_data[:self.flippedData.shape[0]]

        x_train, y_train = extract_seqX_outcomeY(scaled_data_train, window_size, window_size)
        return x_train, y_train

    def run_lstm(self, x_train, layer_units=50, logNeptune=True, NeptuneProject=None):
        inp = Input(shape=(x_train.shape[1], 1))

        x = LSTM(units=layer_units, return_sequences=True)(inp)
        x = LSTM(units=layer_units)(x)

        out = Dense(1, activation='linear')(x)
        model = Model(inp, out)

        model.compile(loss='mean_squared_error', optimizer='adam')

        if logNeptune:
            model.summary(print_fn=lambda z: NeptuneProject[f"Predictions/{self.ticker}/LSTM/model_summary"].log(z))

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
