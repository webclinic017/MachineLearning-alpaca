import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import LSTM, Dense


def extract_seqX_outcomeY(data, N, offset):
    """
    Split time-series into training sequence X and outcome value Y
    :param data: dataset
    :param N: window size, e.g., 50 for 50 days of historical stock prices
    :param offset: position to start the split (same as N)
    :return: numpy arrays of x, y training data
    """
    X, y = [], []

    for i in range(offset, len(data)):
        X.append(data[i - N:i])
        y.append(data[i])

    return np.array(X), np.array(y)


def calculate_change(last_value, predicted_value):
    """
    Calculates price and percent change between last known price value and predicted value
    :param last_value: most recent stock price data point
    :param predicted_value: predicted stock price data point
    :return: a tuple of the price change and percent change (price, percent)
    """
    price = predicted_value - last_value
    percent = (predicted_value/last_value - 1) * 100

    return price, percent


def preprocess_testdata(data, scaler, window_size, data_var: str):
    """
    formats data to pass into the Model
    :param data: dataset
    :param scaler: StandardScaler
    :param window_size: # of previous days it uses to predict the next value
    :return: array of size (window_size, 1) to put into model.predict()
    """
    raw = data[data_var][len(data) - window_size - 1:].values
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
        self.close = {
            'predicted_price': None,
            'change_percentage': None,
            'change_price': None,
            'second_predicted_price': None,
            'second_change_percentage': None,
            'second_change_price': None,
        }

        self.open = {
            'predicted_price': None,
            'change_percentage': None,
            'change_price': None,
            'second_predicted_price': None,
            'second_change_percentage': None,
            'second_change_price': None,
        }

        self.run = run
        self.ticker = ticker.strip()
        self.flippedData = data.copy().loc[::-1].reset_index(drop=True)
        self.handle_lstm()

    def handle_lstm(self):
        """
        runs whole LSTM prediction process and saves prediction to object variables
        """
        cur_epochs = 31
        cur_batch_size = 38
        window_size = 96
        layer_units = 38

        self.close = self.make_model(cur_epochs, cur_batch_size, window_size, layer_units, 'close')
        self.open = self.make_model(cur_epochs, cur_batch_size, window_size, layer_units, 'open')

        if self.run is not None:
            # log everything to neptune
            self.run[f"Predictions/{self.ticker}/LSTM/open"].log(self.open)
            self.run[f"Predictions/{self.ticker}/LSTM/close"].log(self.close)

    def make_model(self, cur_epochs: int, cur_batch_size: int, window_size: int, layer_units: int, data_var: str):
        scaler = StandardScaler()

        x_train, y_train = self.lstm_get_train_data(self.flippedData, scaler, data_var, cur_batch_size=cur_batch_size, cur_epochs=cur_epochs, window_size=window_size, layer_units=layer_units)
        model = self.run_lstm(x_train, layer_units=layer_units)
        model.fit(x_train, y_train, epochs=tf.constant(cur_epochs, dtype="int64"), batch_size=tf.constant(cur_batch_size, dtype="int64"), verbose=0, validation_split=0.1, shuffle=True)
        x_test = preprocess_testdata(data=self.flippedData, scaler=scaler, window_size=window_size, data_var=data_var)

        predicted_price_array = model.predict(x_test)
        predicted_price_array = scaler.inverse_transform(predicted_price_array)

        # set object variables, so I can get the results in Main
        # predicted price for the next day
        predicted_price = predicted_price_array[0][0]
        change_price, change_percentage = calculate_change(self.flippedData.iloc[-1][data_var], predicted_price)

        # predict 2nd day out
        # add prediction to dataframe to predict the next day
        self.flippedData.loc[len(self.flippedData.index)] = [0, 0, 0, 0, 0, predicted_price, 0, 0, 0, 1.0]
        x_test = preprocess_testdata(data=self.flippedData, scaler=scaler, window_size=window_size, data_var=data_var)
        predicted_price_array = model.predict(x_test)
        predicted_price_array = scaler.inverse_transform(predicted_price_array)

        # remove 1st prediction from dataframe to not affect other results
        self.flippedData = self.flippedData.drop(self.flippedData.index[-1])

        second_predicted_price = predicted_price_array[0][0]
        second_change_price, second_change_percentage = calculate_change(predicted_price, second_predicted_price)
        return {
            'predicted_price': predicted_price,
            'change_percentage': change_percentage,
            'change_price': change_price,
            'second_predicted_price': second_predicted_price,
            'second_change_percentage': second_change_percentage,
            'second_change_price': second_change_price
        }

    def lstm_get_train_data(self, stockprices, scaler, data_var: str, layer_units=50, optimizer='adam', cur_epochs=15, cur_batch_size=20, window_size=50):
        """
        logs LSTM parameters to neptune and prepares data for model.fit()
        :param data_var: csv column name
        :param stockprices: stock data
        :param scaler: StandardScaler object
        :param layer_units: layer units
        :param optimizer: model optimizer
        :param cur_epochs: # of epochs
        :param cur_batch_size: batch size
        :param window_size: window size
        :return: x and y training arrays to be passing into model.fit()
        """
        cur_LSTM_pars = {'units': layer_units,
                         'optimizer': optimizer,
                         'batch_size': cur_batch_size,
                         'epochs': cur_epochs
                         }
        if self.run is not None:
            self.run['LSTMPars'] = cur_LSTM_pars

        scaled_data = scaler.fit_transform(stockprices[[data_var]].values)
        scaled_data_train = scaled_data[:self.flippedData.shape[0]]

        x_train, y_train = extract_seqX_outcomeY(scaled_data_train, window_size, window_size)
        return x_train, y_train

    def run_lstm(self, x_train, layer_units=50):
        """
        Builds machine learning model
        :param x_train: training set, needed for its shape for input values
        :param layer_units: # layer_units
        :param logNeptune: unless False, will log model summary to Neptune
        :return: Model
        """
        inp = Input(shape=(x_train.shape[1], 1))

        x = LSTM(units=layer_units, return_sequences=True)(inp)
        x = LSTM(units=layer_units)(x)

        out = Dense(1, activation='linear')(x)
        model = Model(inp, out)

        model.compile(loss='mean_squared_error', optimizer='adam')

        if self.run is not None:
            model.summary(print_fn=lambda z: self.run[f"Predictions/{self.ticker}/LSTM/model_summary"].log(z))

        return model
