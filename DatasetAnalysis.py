import requests
import numpy as np
import pandas
import csv
from dotenv import load_dotenv
import os
import neptune.new as neptune
import matplotlib.pyplot as plt


load_dotenv()
NEPTUNE_API_TOKEN = os.getenv('NEPTUNE-API-TOKEN')
ALPHA_VANTAGE_TOKEN = os.getenv('ALPHA-VANTAGE-API-TOKEN')

def get_new_data():
    stock_symbol = 'IBM'

    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={stock_symbol}&apikey={ALPHA_VANTAGE_TOKEN}&datatype=csv"

    try:
        response = requests.get(url)
    except Exception as e:
        print(e)
        exit(1)

    lines = response.text.splitlines()
    reader = csv.reader(lines)

    f = open('data.csv', 'w')
    csv_writer = csv.writer(f)
    for row in reader:
        csv_writer.writerow(row)

    f.close()


def read_data():
    return pandas.read_csv('data.csv')


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
        X.append(data[i-N:i])
        y.append(data[i])

    return  np.array(X), np.array(y)


def calculate_rmse(y_true, y_pred):

    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    return rmse


def calculate_mape(y_true, y_pred):

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    return mape


def calculate_perf_metrics(var, logNeptune=True, logmodelName='Simple MA'):
    # rmse = calculate_rmse(np.array(stockprices[train_size:]['close']), np.array(stockprices[train_size:][var]))
    #
    # mape = calculate_mape(np.array(stockprices[train_size:]['close']), np.array(stockprices[train_size:][var]))

    rmse = calculate_rmse(np.array(stockprices[1][var]), np.array(stockprices[0][var]))

    mape = calculate_mape(np.array(stockprices[1][var]), np.array(stockprices[0][var]))

    if logNeptune:
        run['RMSE'].log(rmse)
        run['MAPE (%)'].log(mape)

    return rmse, mape


def plot_stock_trend(var, cur_title, stockprices, logNeptune=True, logmodelName='Simple MA'):
    ax = stockprices[['close']].plot(figsize=(20, 10))
    plt.grid(False)
    plt.title(cur_title)
    plt.axis('tight')
    plt.ylabel('Stock Price ($)')

    # plt.show()
    # Log images to Neptune new version
    if logNeptune:
        run[f'Plot of Stock Predictions with {logmodelName}'].upload(neptune.types.File.as_image(ax.get_figure()))


def simple_moving_average(window_size, stockprices):

    window_var = str(window_size) + 'day'

    stockprices[window_var] = stockprices[['close']].rolling(window_size).mean()
    stockprices['200day'] = stockprices[['close']].rolling(200).mean()

    plot_stock_trend(var=window_var, cur_title='Simple Moving Averages')

if __name__ == '__main__':
    # get closing price for each day
    # get_new_data()
    data = read_data()
    # reverses pandas data
    # data = data.iloc[::-1]

    # sets up test sets
    test_ratio = 0.2
    train_ratio = 1 - test_ratio
    train_size = int(train_ratio * len(data))
    test_size = int(test_ratio * len(data))
    train = data[:train_size]
    test = data[train_size:]

    stockprices = extract_seqX_outcomeY(train['close'], 50, 60)
    X = stockprices[0]
    Y = stockprices[1].reshape((20, 1))
    stockprices = (X,Y)

    run = neptune.init(
        project="elitheknight/Stock-Prediction",
        api_token=NEPTUNE_API_TOKEN,
    )

    # changes data to have correct indexes for time
    data = data.loc[::-1].reset_index(drop=True)
    plot_stock_trend('close', 'First Try', stockprices=data)

    for i in range(len(stockprices[1])):
        calculate_perf_metrics(i)

