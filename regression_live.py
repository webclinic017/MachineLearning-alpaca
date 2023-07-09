import os
import neptune
from dotenv import load_dotenv
import keras.models
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.layers import Dense, GRU, Dropout
import keras.optimizers as kop
from joblib import dump, load
import datetime
import Trading
from typing import Union


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
        y.append(data[i][1])  # data[i][1] is close price

    return np.array(X), np.array(y)


def preprocess_testdata(data, scaler: StandardScaler, window_size, data_var: str):
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


def get_data(test: bool = False, window: int = 50, days_back: int = 70):
    with open('tickers.txt', 'r') as f:
        tickers = f.readlines()

    tickers = [i.strip() for i in tickers]
    data = []

    for i in range(len(tickers)//50 + 1 if len(tickers) % 50 != 0 else len(tickers)//50):
        data.extend(Trading.get_historicals(tickers[i*50: min((i+1)*50, len(tickers))], span='year'))

    # gets (open, close, high, low, volume)

    new_data = []
    for i in data:
        new_data.append(list(i.values())[1:6])

    data = np.array(new_data, dtype='float64')

    data = np.array(np.split(data, len(tickers)), dtype='float64')

    if test:   # if testing, remove most recent day of data so I can calculate error, go back 2 if testing with Individual_LSTM
        data = np.delete(data, -1, 1)
        data = np.delete(data, -1, 1)

    adjusted_data = np.copy(data)
    for j in range(adjusted_data.shape[-1] - 1):
        for i in range(adjusted_data.shape[1] - 1):
            adjusted_data[:, i+1, j] = ((data[:, i+1, j] / data[:, i, j]) - 1) * 100

    # have to do this at least once
    adjusted_data = np.delete(adjusted_data, 0, 1)

    if adjusted_data.shape[1] > days_back:
        while adjusted_data.shape[1] > days_back:
            adjusted_data = np.delete(adjusted_data, 0, 1)

    y_data = adjusted_data[:, -1, 1]

    x_data = np.delete(adjusted_data, -1, 1)

    return x_data, y_data


def get_prediction_data(tickers: Union[str, list] = None, span: str = 'year', test: bool = False, window: int = 50):
    if tickers is None:
        with open('tickers.txt', 'r') as f:
            tickers = f.readlines()

        tickers = [i.strip() for i in tickers]
    elif isinstance(tickers, str):
        tickers = [tickers]

    data = []
    for i in range(len(tickers)//50 + 1 if len(tickers) % 50 != 0 else len(tickers)//50):
        data.extend(Trading.get_historicals(tickers[i*50: min((i+1)*50, len(tickers))], span=span))

    new_data = []
    for i in data:
        new_data.append(list(i.values())[1:6])

    data = np.array(new_data, dtype='float64')

    data = np.array(np.split(data, len(tickers)), dtype='float64')

    if test:   # if testing, remove most recent day of data so I can calculate MAPE, 2 if testing with Individual_LSTM
        data = np.delete(data, -1, 1)
        data = np.delete(data, -1, 1)

    adjusted_data = np.copy(data)
    for j in range(adjusted_data.shape[-1] - 1):
        for i in range(adjusted_data.shape[1] - 1):
            adjusted_data[:, i + 1, j] = ((data[:, i + 1, j] / data[:, i, j]) - 1) * 100

    # have to do this at least once
    adjusted_data = np.delete(adjusted_data, 0, 1)

    if adjusted_data.shape[1] > window:
        while adjusted_data.shape[1] > window:
            adjusted_data = np.delete(adjusted_data, 0, 1)

    return adjusted_data


class Regression:

    def __init__(self, run):
        # date when making the model should always be the current date
        self.date = datetime.date.today().strftime('%Y-%m-%d')
        self.run = run
        self.path = f"Models/GRU/{self.date}"
        self.window = 50
        self.days_back = 70

        model = self.make_model(test=False)
        predictions = self.predict_from_model(model, test=False)
        self.run[f"GRU/Predictions"].log(predictions)

        for k, v in predictions.items():
            run[f"Predictions/{k}/GRU"].log(v)

        sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        self.run[f"GRU/Sorted_Predictions"].log(sorted_preds)
        ind_pos_from_mean = 0

        middle = np.mean(list(predictions.values()))
        for j in range(len(sorted_preds)):
            if sorted_preds[j][1] <= middle:
                ind_pos_from_mean = j
                break

        pos_sorted_preds = sorted_preds[:ind_pos_from_mean]
        order_stocks = pos_sorted_preds[:int(len(pos_sorted_preds) * 3 / 4)]
        self.order_stocks = [i[0] for i in order_stocks]
        print(f"order_stocks: {self.order_stocks}")
        run["GRU/order_stocks"].log(self.order_stocks)

    def make_model(self, test: bool = False):

        learning_rate = 0.0005
        beta_1 = 0.9
        beta_2 = 0.85
        epsilon = 0.0000000128
        weight_decay = None

        cur_epochs = 100
        dropout = 0.1

        self.run["model_args/cur_epochs"].log(cur_epochs)
        self.run[f"model_args/learning_rate"].log(learning_rate)
        self.run[f"model_args/beta_1"].log(beta_1)
        self.run[f"model_args/beta_2"].log(beta_2)
        self.run[f"model_args/epsilon"].log(epsilon)
        self.run[f"model_args/dropout"].log(dropout)
        self.run[f"model_args/weight_decay"].log(weight_decay if weight_decay else 'None')
        self.run[f"model_args/window"].log(self.window)
        self.run[f"model_args/days_back"].log(self.days_back)

        opt = kop.Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon)
        opt.weight_decay = weight_decay

        regressionGRU = keras.Sequential()
        regressionGRU.add(GRU(units=100, input_shape=(self.window, 5), activation='relu', return_sequences=True))
        regressionGRU.add(Dropout(dropout))
        regressionGRU.add(GRU(units=40, input_shape=(self.window, 5), activation='relu', return_sequences=False))
        regressionGRU.add(Dropout(dropout))
        regressionGRU.add(Dense(units=1, activation='sigmoid'))

        regressionGRU.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        print('making model')

        if not os.path.exists(self.path):
            os.makedirs(self.path)

        print('starting training loop')
        data, y_data = get_data(test=test, window=self.window, days_back=self.days_back)
        first = True
        for i in data:
            x, y = extract_seqX_outcomeY(i, self.window)
            x = x.reshape((1, x.shape[0], x.shape[1], x.shape[2]))
            y = y.reshape((1, y.shape[0]))
            if first:
                x_train = x
                y_train = y
                first = False
            else:
                x_train = np.concatenate((x_train, x), axis=0)
                y_train = np.concatenate((y_train, y), axis=0)
        x_train = x_train.reshape((x_train.shape[0]*x_train.shape[1], x_train.shape[2], x_train.shape[3]))
        y_train = y_train.reshape((y_train.shape[0] * y_train.shape[1]))
        first = True
        for i in range(x_train.shape[-1]):
            scaler = StandardScaler()
            sub_arr = x_train[:, :, i]
            # sub_arr = sub_arr.reshape(sub_arr.shape[0]*sub_arr.shape[1])
            sub_scaled_data = scaler.fit_transform(sub_arr)
            sub_scaled_data = sub_scaled_data.reshape((sub_scaled_data.shape[0], sub_scaled_data.shape[1], 1))
            if first:
                scaled_data = sub_scaled_data
                first = False
            else:
                scaled_data = np.concatenate((scaled_data, sub_scaled_data), axis=2)
            dump(scaler, f"{self.path}/{i}scaler.bin")

        # y_scaler = StandardScaler()
        y_train = y_train.reshape((y_train.shape[0], 1))
        # binary_y_data = (y_data > 0).astype(int)
        scaled_y_data = (y_train-np.min(y_train))/(np.max(y_train)-np.min(y_train))
        regressionGRU.fit(scaled_data, scaled_y_data, epochs=cur_epochs, batch_size=32, verbose=1, shuffle=True)
        # output with binary_y_data is probability between 0-1 of increasing

        # save models
        print('saving model')
        keras.models.save_model(regressionGRU, self.path)

        return regressionGRU

    def predict_from_model(self, model, test: bool = False):
        with open('tickers.txt', 'r') as f:
            tickers = f.readlines()

        tickers = [i.strip() for i in tickers]

        data = get_prediction_data(tickers=tickers, test=test, window=self.window)
        first = True
        for i in range(data.shape[-1]):
            scaler = load(f"{self.path}/{i}scaler.bin")
            sub_arr = data[:, :, i]
            sub_scaled_data = scaler.transform(sub_arr)
            sub_scaled_data = sub_scaled_data.reshape((sub_scaled_data.shape[0], sub_scaled_data.shape[1], 1))
            if first:
                scaled_data = sub_scaled_data
                first = False
            else:
                scaled_data = np.concatenate((scaled_data, sub_scaled_data), axis=2)

        results = {}

        for i in range(len(scaled_data)):
            predicted_price = model.predict(scaled_data[i].reshape(1, scaled_data[i].shape[0], scaled_data[i].shape[1]), verbose=0)
            results[tickers[i]] = float(predicted_price[0][0])
        return results
