import keras.models
import numpy as np
import pandas
from sklearn.preprocessing import StandardScaler
from keras import Input, Model
from keras.layers import LSTM, Dense
import keras.optimizers as kop
from joblib import dump, load


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


def make_both_models():
    """
    runs whole LSTM prediction process and saves prediction to object variables
    """
    cur_epochs = 10
    cur_batch_size = 25
    window_size = 50
    layer_units = 50

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

    optimizer = kop.Adam(learning_rate=0.006, beta_1=0.850, beta_2=0.999, epsilon=1e-07)
    optimizer.weight_decay = 0.004

    # if run is not None:
    #     # log everything to neptune
    #     run[f"Predictions/{ticker}/LSTM/open"].log(open)
    #     run[f"Predictions/{ticker}/LSTM/close"].log(close)

    make_model(cur_epochs, cur_batch_size, window_size, layer_units, 'close', optimizer)
    make_model(cur_epochs, cur_batch_size, window_size, layer_units, 'open', optimizer)


def make_model(cur_epochs: int, cur_batch_size: int, window_size: int, layer_units: int, data_var: str, optimizer: kop = 'adam'):
    with open('tickers.txt', 'r') as f:
        tickers = f.readlines()

    tickers = [i.strip() for i in tickers]
    date = pandas.read_csv(f"Data/AAPL_data.csv")['timestamp'][0]

    scaler = StandardScaler()
    data_size = 100
    print('making model')
    inp = Input(shape=(window_size, 1))
    x = LSTM(units=layer_units, return_sequences=True)(inp)
    x = LSTM(units=layer_units)(x)
    out = Dense(1, activation='linear')(x)
    model = Model(inp, out)
    model.compile(loss='mean_squared_error', optimizer='adam')

    print('starting training loop')
    fit = True
    for ticker in tickers:
        flippedData = pandas.read_csv(f"Data/{ticker}_data.csv").copy().loc[::-1].reset_index(drop=True)[data_size*-1 - 2:-2]
        if fit:
            scaled_data = scaler.fit_transform(flippedData[[data_var]].values)
            fit = False
        else:
            scaled_data = scaler.transform(flippedData[[data_var]].values)
        scaled_data_train = scaled_data[:flippedData.shape[0]]

        x_train, y_train = extract_seqX_outcomeY(scaled_data_train, window_size, window_size)
        model.fit(x_train, y_train, epochs=cur_epochs, batch_size=cur_batch_size, verbose=0, validation_split=0.1, shuffle=True)

    # save models
    print('saving model')
    keras.models.save_model(model, f"Models/{date}/{data_var}")
    dump(scaler, f"Models/{date}/{data_var}scaler.bin")

    # ticker = 'AAPL'
    # flippedData = pandas.read_csv(f"Data/{ticker}_data.csv").copy().loc[::-1].reset_index(drop=True)[data_size * -1 - 2:-2]
    #
    # x_test = preprocess_testdata(data=flippedData, scaler=scaler, window_size=window_size, data_var=data_var)
    #
    # predicted_price_array = model.predict(x_test, verbose=0)
    # predicted_price_array = scaler.inverse_transform(predicted_price_array)
    #
    # # set object variables, so I can get the results in Main
    # # predicted price for the next day
    # predicted_price = predicted_price_array[0][0]
    # change_price, change_percentage = calculate_change(flippedData.iloc[-1][data_var], predicted_price)
    # print(predicted_price)
    # print(change_percentage)
    #
    # # predict 2nd day out
    # # add prediction to dataframe to predict the next day
    # if data_var == 'close':
    #     flippedData.loc[len(flippedData.index)] = [0, 0, 0, 0, 0, predicted_price, 0, 0, 0, 1.0]
    # else:
    #     flippedData.loc[len(flippedData.index)] = [0, 0, predicted_price, 0, 0, 0, 0, 0, 0, 1.0]
    #
    # x_test = preprocess_testdata(data=flippedData, scaler=scaler, window_size=window_size, data_var=data_var)
    # predicted_price_array = model.predict(x_test, verbose=0)
    # predicted_price_array = scaler.inverse_transform(predicted_price_array)
    #
    # # remove 1st prediction from dataframe to not affect other results
    # flippedData = flippedData.drop(flippedData.index[-1])
    #
    # second_predicted_price = predicted_price_array[0][0]
    # second_change_price, second_change_percentage = calculate_change(predicted_price, second_predicted_price)
    # print(second_predicted_price)
    # print(second_change_percentage)


def predict_from_model(model, data_var, date):
    ticker = 'AAPL'
    data_size = 100
    window_size = 50

    scaler = load(f"Models/{date}/{data_var}scaler.bin")
    flippedData = pandas.read_csv(f"Data/{ticker}_data.csv").copy().loc[::-1].reset_index(drop=True)[data_size * -1 - 2:-2]
    print(flippedData)
    x_test = preprocess_testdata(data=flippedData, scaler=scaler, window_size=window_size, data_var=data_var)

    predicted_price_array = model.predict(x_test, verbose=0)
    predicted_price_array = scaler.inverse_transform(predicted_price_array)

    # set object variables, so I can get the results in Main
    # predicted price for the next day
    predicted_price = predicted_price_array[0][0]
    change_price, change_percentage = calculate_change(flippedData.iloc[-1][data_var], predicted_price)
    print(predicted_price)
    print(change_percentage)

    # predict 2nd day out
    # add prediction to dataframe to predict the next day
    if data_var == 'close':
        flippedData.loc[len(flippedData.index)] = [0, 0, 0, 0, 0, predicted_price, 0, 0, 0, 1.0]
    else:
        flippedData.loc[len(flippedData.index)] = [0, 0, predicted_price, 0, 0, 0, 0, 0, 0, 1.0]

    x_test = preprocess_testdata(data=flippedData, scaler=scaler, window_size=window_size, data_var=data_var)
    predicted_price_array = model.predict(x_test, verbose=0)
    predicted_price_array = scaler.inverse_transform(predicted_price_array)

    # remove 1st prediction from dataframe to not affect other results
    flippedData = flippedData.drop(flippedData.index[-1])

    second_predicted_price = predicted_price_array[0][0]
    second_change_price, second_change_percentage = calculate_change(predicted_price, second_predicted_price)
    print(second_predicted_price)
    print(second_change_percentage)
    # return {
    #     'predicted_price': predicted_price,
    #     'change_percentage': change_percentage,
    #     'change_price': change_price,
    #     'second_predicted_price': second_predicted_price,
    #     'second_change_percentage': second_change_percentage,
    #     'second_change_price': second_change_price
    # }


if __name__ == '__main__':
    make_both_models()
    # date = pandas.read_csv(f"Data/AAPL_data.csv")['timestamp'][0]
    #
    # open = keras.models.load_model(f"Models/{date}/open")
    # close = keras.models.load_model(f"Models/{date}/close")
    #
    # predict_from_model(open, 'open', date)
    # predict_from_model(close, 'close', date)

