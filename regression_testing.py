import os

import keras.models
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.layers import Dense, GRU, Dropout
import keras.optimizers as kop
from joblib import dump, load
from datetime import date
import Trading
from typing import Union


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


def get_data():
    with open('tickers.txt', 'r') as f:
        tickers = f.readlines()

    tickers = [i.strip() for i in tickers]
    data = []

    for i in range(len(tickers)//50 + 1 if len(tickers) % 50 != 0 else len(tickers)//50):
        data.extend(Trading.get_historicals(tickers[i*50: min((i+1)*50, len(tickers))], span='month'))

    new_data = []
    for i in data:
        new_data.append(list(i.values())[1:6])

    data = np.array(new_data, dtype='float64')

    data = np.array(np.split(data, len(tickers)), dtype='float64')

    # index of data[x:y:1] is the close price
    y_data = data[:, -1, 1]

    x_data = np.delete(data, -1, 1)

    return x_data, y_data


def get_prediction_data(tickers: Union[str, list] = None, span: str = 'month'):
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

    data = np.delete(data, 0, 1)

    return data


def make_model(cur_epochs: int, layer_units: int):

    print('making model')
    regressionGRU = keras.Sequential()
    regressionGRU.add(GRU(units=layer_units, return_sequences=True, input_shape=(19, 5)))
    regressionGRU.add(Dropout(0.2))
    regressionGRU.add(GRU(units=layer_units, return_sequences=True, input_shape=(19, 5)))
    regressionGRU.add(Dropout(0.2))
    regressionGRU.add(GRU(units=layer_units, return_sequences=True, input_shape=(19, 5)))
    regressionGRU.add(Dropout(0.2))
    regressionGRU.add(GRU(units=layer_units, input_shape=(19, 5)))
    regressionGRU.add(Dropout(0.2))
    regressionGRU.add(Dense(units=1))
    regressionGRU.compile(loss='mean_absolute_error', optimizer=kop.SGD(), metrics=['mae'])

    path = f"Models/GRU/{date}"
    if not os.path.exists(path):
        os.makedirs(path)

    print('starting training loop')
    data, y_data = get_data()
    first = True
    for i in range(data.shape[-1]):
        scaler = StandardScaler()
        sub_arr = data[:, :, i]
        # sub_arr = sub_arr.reshape(sub_arr.shape[0]*sub_arr.shape[1])
        sub_scaled_data = scaler.fit_transform(sub_arr)
        sub_scaled_data = sub_scaled_data.reshape((sub_scaled_data.shape[0], sub_scaled_data.shape[1], 1))
        if first:
            scaled_data = sub_scaled_data
            first = False
        else:
            scaled_data = np.concatenate((scaled_data, sub_scaled_data), axis=2)

        dump(scaler, f"{path}/{i}scaler.bin")

    print(y_data)

    regressionGRU.fit(scaled_data, y_data, epochs=cur_epochs, verbose=0, validation_split=0.1, shuffle=True)

    # save models
    print('saving model')
    keras.models.save_model(regressionGRU, f"Models/GRU/{date}")


def predict_from_model(model, path):
    tickers = ['AAPL', 'ABBV']

    data = get_prediction_data(tickers=tickers)
    first = True
    for i in range(data.shape[-1]):
        scaler = load(f"{path}/{i}scaler.bin")
        sub_arr = data[:, :, i]
        sub_scaled_data = scaler.transform(sub_arr)
        sub_scaled_data = sub_scaled_data.reshape((sub_scaled_data.shape[0], sub_scaled_data.shape[1], 1))
        if first:
            scaled_data = sub_scaled_data
            first = False
        else:
            scaled_data = np.concatenate((scaled_data, sub_scaled_data), axis=2)

    assert(len(scaled_data) == len(tickers))
    results = {}
    for i in range(len(scaled_data)):
        predicted_price = model.predict(scaled_data[i].reshape(1, scaled_data[i].shape[0], scaled_data[i].shape[1]), verbose=0)
        results[tickers[i]] = float(predicted_price[0][0])

    return results


if __name__ == '__main__':
    # date when making the model should always be the current date
    date = date.today().strftime('%Y-%m-%d')
    trader = Trading.Trader()

    cur_epochs = 50
    layer_units = 50

    # make_model(cur_epochs, layer_units)

    # date in path when predicting data should be set to any model currently stored (best to use as close to present as possible)

    path = f"Models/GRU/{date}"
    model = keras.models.load_model(path)

    predictions = predict_from_model(model, path)
    print(predictions)

