import numpy as np
from sklearn.preprocessing import StandardScaler
from keras import Input, Model
from keras.layers import LSTM, Dense
import keras.optimizers as kop


def extract_seqX_outcomeY(data, N):
    """
    Split time-series into training sequence X and outcome value Y
    :param data: dataset
    :param N: window size, e.g., 50 for 50 days of historical stock prices
    :param offset: position to start the split (same as N)
    :return: numpy arrays of x, y training data
    """
    X, y = [], []

    for i in range(N, len(data)):
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

    def __init__(self, ticker, data, run, cur_pars):
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

        self.cur_epochs = cur_pars['cur_epochs']
        self.window_size = cur_pars['window_size']
        self.learning_rate = cur_pars['learning_rate']
        self.beta_1 = cur_pars['beta_1']
        self.beta_2 = cur_pars['beta_2']
        self.epsilon = cur_pars['epsilon']

        self.run = run
        self.ticker = ticker.strip()
        self.flippedData = data.copy().loc[::-1].reset_index(drop=True)
        self.handle_lstm()

    def handle_lstm(self):
        """
        runs whole LSTM prediction process and saves prediction to object variables
        """
        # cur_epochs = 50
        # window_size = 50
        # learning_rate = 0.001
        # beta_1 = 0.9
        # beta_2 = 0.999
        # epsilon = 1e-07
        # weight_decay = None

        # self.run['LSTMpars/cur_epochs'].log(self.cur_epochs)
        # self.run['LSTMpars/window_size'].log(self.window_size)
        # self.run['LSTMpars/learning_rate'].log(self.learning_rate)
        # self.run['LSTMpars/beta_1'].log(self.beta_1)
        # self.run['LSTMpars/beta_2'].log(self.beta_2)
        # self.run['LSTMpars/epsilon'].log(self.epsilon)
        # self.run['LSTMpars/weight_decay'].log(weight_decay)

        # default adam settings
        # learning_rate = 0.001,
        # beta_1 = 0.9,
        # beta_2 = 0.999,
        # epsilon = 1e-07,
        # amsgrad = False,
        # weight_decay = None,
        # clipnorm = None,
        # clipvalue = None,
        # global_clipnorm = None,
        # use_ema = False,
        # ema_momentum = 0.99,
        # ema_overwrite_frequency = None,
        # jit_compile = True

        optimizer = kop.Adam(learning_rate=self.learning_rate, beta_1=self.beta_1, beta_2=self.beta_2, epsilon=self.epsilon)
        # optimizer.weight_decay = weight_decay

        self.close = self.make_model(self.cur_epochs, self.window_size, 'close', optimizer)
        self.open = self.make_model(self.cur_epochs, self.window_size, 'open', optimizer)

        if self.run is not None:
            # log everything to neptune
            self.run[f"Predictions/{self.ticker}/LSTM/open"].log(str(self.open))
            self.run[f"Predictions/{self.ticker}/LSTM/close"].log(str(self.close))

    def make_model(self, cur_epochs: int, window_size: int, data_var: str, optimizer: kop = 'adam'):
        scaler = StandardScaler()

        x_train, y_train = self.lstm_get_train_data(self.flippedData, scaler, data_var, window_size=window_size)

        inp = Input(shape=(x_train.shape[1], 1))
        x = LSTM(units=60, return_sequences=True)(inp)
        x = LSTM(units=30)(x)
        out = Dense(1, activation='linear')(x)

        model = Model(inp, out)
        model.compile(loss='mean_squared_error', optimizer=optimizer)

        # if self.run is not None:
        #     model.summary(print_fn=lambda z: self.run[f"Predictions/{self.ticker}/LSTM/model_summary"].log(z))

        model.fit(x_train, y_train, epochs=cur_epochs, batch_size=32, verbose=0, validation_split=0.1, shuffle=True)
        x_test = preprocess_testdata(data=self.flippedData, scaler=scaler, window_size=window_size, data_var=data_var)

        predicted_price_array = model.predict(x_test, verbose=0)
        predicted_price_array = scaler.inverse_transform(predicted_price_array)

        # set object variables, so I can get the results in Main
        # predicted price for the next day
        predicted_price = predicted_price_array[0][0]
        change_price, change_percentage = calculate_change(self.flippedData.iloc[-1][data_var], predicted_price)

        # predict 2nd day out
        # add prediction to dataframe to predict the next day
        if data_var == 'close':
            self.flippedData.loc[len(self.flippedData.index)] = [0, 0, 0, 0, predicted_price, 0, 0, 0, 1.0]
        else:
            self.flippedData.loc[len(self.flippedData.index)] = [0, predicted_price, 0, 0, 0, 0, 0, 0, 1.0]
        x_test = preprocess_testdata(data=self.flippedData, scaler=scaler, window_size=window_size, data_var=data_var)
        predicted_price_array = model.predict(x_test, verbose=0)
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

    def lstm_get_train_data(self, stockprices, scaler, data_var: str, window_size=50):
        """
        logs LSTM parameters to neptune and prepares data for model.fit()
        :param data_var: csv column name
        :param stockprices: stock data
        :param scaler: StandardScaler object
        :param window_size: window size
        :return: x and y training arrays to be passing into model.fit()
        """

        # cur_LSTM_pars = {'units': layer_units,
        #                  'optimizer': optimizer,
        #                  'batch_size': cur_batch_size,
        #                  'epochs': cur_epochs
        #                  }
        # if self.run is not None:
        #     self.run['LSTMPars'] = str(cur_LSTM_pars)

        scaled_data = scaler.fit_transform(stockprices[[data_var]].values)
        scaled_data_train = scaled_data[:self.flippedData.shape[0]]

        x_train, y_train = extract_seqX_outcomeY(scaled_data_train, window_size)
        return x_train, y_train
