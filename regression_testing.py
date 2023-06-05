import os
import neptune
from neptune.types import File
from dotenv import load_dotenv
import keras.models
import keras.metrics as metrics
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.layers import Dense, GRU, Dropout
import keras.losses as losses
import keras.optimizers as kop
from joblib import dump, load
from datetime import date, datetime
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


def calculate_mape(y_true, y_pred):

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    return mape


def calculate_difference(y_true, y_pred):
    return np.abs(y_true-y_pred)


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


def get_data(test: bool = False):
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

    if test:   # if testing, remove most recent day of data so I can calculate error
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


def get_prediction_data(tickers: Union[str, list] = None, span: str = 'year', test: bool = False):
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

    if test:   # if testing, remove most recent day of data so I can calculate MAPE
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


def make_model(cur_epochs: int, layer_units: int, test: bool = False):

    learning_rate = 0.00051
    beta_1 = 0.9
    # beta_2 = 0.45
    epsilon = 1e-7
    weight_decay = None
    #
    # keras.optimizers.Adam()
    #
    run["model_args/cur_epochs"].log(cur_epochs)
    # run["model_args/layer_units"].log(layer_units)
    run[f"model_args/learning_rate"].log(learning_rate)
    run[f"model_args/beta_1"].log(beta_1)
    run[f"model_args/beta_2"].log(beta_2)
    run[f"model_args/epsilon"].log(epsilon)
    run[f"model_args/weight_decay"].log(weight_decay if weight_decay else 'None')
    #
    opt = kop.Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon)
    opt.weight_decay = weight_decay
    # loss = losses.KLDivergence()    #mean absolute percentage error is next best
    # metric = metrics.SparseTopKCategoricalAccuracy()
    # dropout = 0.2
    #
    # run[f"model_args/dropout"].log(dropout)
    # run[f"model_args/loss"].log(loss.name)
    # run[f"model_args/metrics"].log(metric.name if metric else 'None')

    regressionGRU = keras.Sequential()
    # regressionGRU.add(GRU(units=160, input_shape=(window, 5), return_sequences=True, activation='relu'))
    # regressionGRU.add(Dropout(0.1))
    # regressionGRU.add(GRU(units=80, input_shape=(window, 5), return_sequences=True, activation='relu'))
    # regressionGRU.add(Dropout(0.1))
    # regressionGRU.add(GRU(units=120, input_shape=(window, 5), return_sequences=True, activation='relu'))
    # regressionGRU.add(Dropout(0.1))
    regressionGRU.add(GRU(units=100, input_shape=(window, 5), activation='relu', return_sequences=True))
    # regressionGRU.add(Dropout(0.1))
    # regressionGRU.add(GRU(units=30, input_shape=(window, 5), activation='relu', return_sequences=True))
    regressionGRU.add(Dropout(0.1))
    regressionGRU.add(GRU(units=40, input_shape=(window, 5), activation='relu', return_sequences=False))
    # regressionGRU.add(Dropout(0.1))

    regressionGRU.add(Dense(units=1, activation='sigmoid'))

    regressionGRU.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    print('making model')
    # regressionGRU = keras.Sequential()
    # regressionGRU.add(GRU(units=layer_units, return_sequences=True, input_shape=(17, 5)))
    # regressionGRU.add(Dropout(dropout))
    # regressionGRU.add(GRU(units=layer_units, return_sequences=True, input_shape=(17, 5)))
    # regressionGRU.add(Dropout(dropout))
    # regressionGRU.add(GRU(units=layer_units, return_sequences=True, input_shape=(17, 5)))
    # regressionGRU.add(Dropout(dropout))
    # regressionGRU.add(GRU(units=layer_units, return_sequences=True, input_shape=(17, 5)))
    # regressionGRU.add(Dropout(dropout))
    # regressionGRU.add(GRU(units=layer_units, return_sequences=True, input_shape=(17, 5)))
    # regressionGRU.add(Dropout(dropout))
    # regressionGRU.add(GRU(units=layer_units, return_sequences=True, input_shape=(17, 5)))
    # regressionGRU.add(Dropout(dropout))
    # regressionGRU.add(GRU(units=layer_units, return_sequences=True, input_shape=(17, 5)))
    # regressionGRU.add(Dropout(dropout))
    # regressionGRU.add(GRU(units=layer_units, return_sequences=True, input_shape=(17, 5)))
    # regressionGRU.add(Dropout(dropout))
    # regressionGRU.add(GRU(units=layer_units, return_sequences=True, input_shape=(17, 5)))
    # regressionGRU.add(Dropout(dropout))
    # regressionGRU.add(GRU(units=layer_units, return_sequences=True, input_shape=(17, 5)))
    # regressionGRU.add(Dropout(dropout))
    # regressionGRU.add(GRU(units=layer_units, input_shape=(17, 5)))
    # regressionGRU.add(Dropout(dropout))
    # regressionGRU.add(Dense(units=1))
    # regressionGRU.compile(loss=loss, optimizer=opt, metrics=metric)

    path = f"Models/GRU/{date}"
    if not os.path.exists(path):
        os.makedirs(path)

    print('starting training loop')
    data, y_data = get_data(test=test)
    first = True
    for i in data:
        x, y = extract_seqX_outcomeY(i, window)
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
        dump(scaler, f"{path}/{i}scaler.bin")

    # y_scaler = StandardScaler()
    y_train = y_train.reshape((y_train.shape[0], 1))
    # binary_y_data = (y_data > 0).astype(int)
    scaled_y_data = (y_train-np.min(y_train))/(np.max(y_train)-np.min(y_train))
    regressionGRU.fit(scaled_data, scaled_y_data, epochs=cur_epochs, batch_size=32, verbose=0, shuffle=True)
    # output with binary_y_data is probability between 0-1 of increasing

    # save models
    print('saving model')
    keras.models.save_model(regressionGRU, f"Models/GRU/{date}")

    return regressionGRU


def predict_from_model(model, path, test: bool = False):
    # tickers = ['AAPL', 'ABBV', 'TSLA']
    with open('tickers.txt', 'r') as f:
        tickers = f.readlines()

    tickers = [i.strip() for i in tickers]

    data = get_prediction_data(tickers=tickers, test=test)
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
    # y_scaler = load(f"{path}/yscaler.bin")
    results = {}
    for i in range(len(scaled_data)):
        predicted_price = model.predict(scaled_data[i].reshape(1, scaled_data[i].shape[0], scaled_data[i].shape[1]), verbose=0)
        # predicted_price = y_scaler.inverse_transform(predicted_price)
        results[tickers[i]] = float(predicted_price[0][0])

    return results


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    load_dotenv()
    NEPTUNE_API_TOKEN = os.getenv('NEPTUNE-API-TOKEN')
    # date when making the model should always be the current date
    date = date.today().strftime('%Y-%m-%d')
    trader = Trading.Trader()

    window = 50
    days_back = 70

    for i in range(18):
        dateTimeObj = datetime.now()
        custom_id = 'EXP-' + dateTimeObj.strftime("%d-%b-%Y-(%H:%M:%S)")
        run = neptune.init_run(
            project="elitheknight/Stock-Testing",
            custom_run_id=custom_id,
            api_token=NEPTUNE_API_TOKEN,
            capture_stdout=False,
            capture_stderr=False,
            capture_hardware_metrics=False
        )

        cur_epochs = 100
        layer_units = 60
        beta_2 = 0.1 + i*0.05
        path = f"Models/GRU/{date}"

        model = make_model(cur_epochs, layer_units, test=True)
        # model = keras.models.load_model(path)
        predictions = predict_from_model(model, path, test=True)

        with open('tickers.txt', 'r') as f:
            tickers = f.readlines()

        tickers = [i.strip() for i in tickers]
        # errors = {}
        true_vals = dict(zip(tickers, list(np.array(Trading.get_last_close_percent_change(tickers)) > 0)))
        print(true_vals)
        middle = np.mean(list(predictions.values()))
        median = np.median(list(predictions.values()))
        predictions_binary = {k: v > 0.5 for k, v in predictions.items()}
        correct_signs_half = 0
        correct_signs_mean = 0
        correct_signs_median = 0
        for k, v in predictions.items():
            if true_vals[k] == (v > 0.5):
                correct_signs_half += 1
            if true_vals[k] == (v > middle):
                correct_signs_mean += 1
            if true_vals[k] == (v > median):
                correct_signs_median += 1
            run[f"Predictions/{k}"].log(v)
            # errors[k] = calculate_difference(true_vals[k], v)
            # run[f"Prediction_Errors/{k}"].log(errors[k])

        correct_signs_half = correct_signs_half / len(predictions) * 100
        correct_signs_median = correct_signs_median / len(predictions) * 100
        correct_signs_mean = correct_signs_mean / len(predictions) * 100
        # average_error = np.mean(list(errors.values()))
        # max_error = np.max(list(errors.values()))
        run[f"correct_signs (%)"].log(f"correct (%) - 0.5: {correct_signs_half} mean: {correct_signs_mean} median: {correct_signs_median}")
        run[f"preds"].log(predictions_binary)
        run[f"true"].log(true_vals)
        # run[f"average_error"].log(average_error)
        # run[f"max_error"].log(max_error)
        print(predictions)
        print(predictions_binary)
        print(f"correct (%) - 0.5: {correct_signs_half} mean: {correct_signs_mean} median: {correct_signs_median}")
        print(f"mean: {middle}, median: {median}")
        print(f"min_pred: {min(list(predictions.values()))}, max_pred: {max(list(predictions.values()))}")
        run["min max"].log(f"min_pred: {min(list(predictions.values()))}, max_pred: {max(list(predictions.values()))}")
        # print(errors)
        # print(f"max error: {max_error}")
        # print(average_error)

        run.stop()

        # date in path when predicting data should be set to any model currently stored (best to use as close to present as possible)

        # load model from files:
        # model = keras.models.load_model(path)

        # predictions = predict_from_model(model, path)
        # print(predictions)

