import pandas
from dotenv import load_dotenv
import os
import time
from datetime import datetime
import warnings
import neptune.new as neptune
import matplotlib
from multiprocessing import Pool
from Individual_LSTM import IndividualLSTM


def begin(ticker, id, NAT, AVT):
    dataset_size = 400
    data = get_data_to_file(ticker, AVT, dataset_size)
    run = neptune.init(
        project="elitheknight/Stock-Prediction",
        api_token=NAT,
        custom_run_id=id,
        capture_stdout=False,
        capture_stderr=False,
        capture_hardware_metrics=False
    )
    stock = IndividualLSTM(ticker, data, run)
    print(f"{ticker} complete")
    return ticker, stock.predicted_price, stock.change_price, stock.change_percentage


def get_data(ticker, AVT, dataset_size):
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={ticker}&apikey={AVT}&datatype=csv&outputsize=full"
    try:
        return pandas.read_csv(url)[:dataset_size]
    except Exception as e:
        print(f"error in getting data for ticker: {ticker}, url: {url}, error: {e}")
        exit(1)


def get_data_to_file(ticker, AVT, dataset_size):
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={ticker}&apikey={AVT}&datatype=csv&outputsize=full"

    try:
        df = pandas.read_csv(url)[:dataset_size]
    except Exception as e:
        print(f"error in getting data for ticker: {ticker}, url: {url}, error: {e}")
        exit(1)
        return

    df.to_csv(f"Data/{ticker}_data.csv")

    return df


def predict_stock_prices():
    blacklist = ['AMZN', 'GOOG', 'NVDA', 'TSLA', 'NKE']
    # max to run at once is 20 because of pandas figure limits
    with open('tickers.txt', 'r') as f:
        lines = f.readlines()
    # lines = ['AAPL']
    listOfTickers = [(line.strip(), custom_id, NEPTUNE_API_TOKEN, ALPHA_VANTAGE_TOKEN) for line in lines if line.strip() not in blacklist]

    tickers_to_run = listOfTickers.copy()
    num_processes = 4
    num_tickers = len(listOfTickers)
    groups = num_tickers // num_processes + 1
    stock_prediction_data = []

    for i in range(groups):
        # multiprocessing
        pool = Pool(processes=num_processes)
        print(f"starting wave {i+1}/{groups}")
        result = pool.starmap_async(begin, tickers_to_run[:num_processes])
        tickers_to_run = tickers_to_run[num_processes:]
        # I have to wait at least 60 seconds to limit my API calls
        time.sleep(60)
        # start_time = time.time()
        results = result.get()
        stock_prediction_data.extend(results)
        # print(f"delay: {time.time() - start_time}")
        pool.close()

    return stock_prediction_data


def handle_predictions(data):
    """
    :param data: list of tuples in form (ticker, predicted_price, change_price, change_percentage)
    :return:
    """

    sorted_by_percent = sorted(data, key=lambda x: x[3], reverse=True)
    run_[f"All_prediction_data/Data"].log(sorted_by_percent)
    print(sorted_by_percent)


if __name__ == '__main__':
    load_dotenv()
    NEPTUNE_API_TOKEN = os.getenv('NEPTUNE-API-TOKEN')
    ALPHA_VANTAGE_TOKEN = os.getenv('ALPHA-VANTAGE-API-TOKEN')

    pandas.options.mode.chained_assignment = None  # default='warn'
    matplotlib.use('SVG')
    warnings.filterwarnings(action='ignore', category=UserWarning)

    dateTimeObj = datetime.now()
    custom_id = 'EXP-' + dateTimeObj.strftime("%d-%b-%Y-(%H:%M:%S)")
    run_ = neptune.init(
        project="elitheknight/Stock-Prediction",
        api_token=NEPTUNE_API_TOKEN,
        custom_run_id=custom_id,
        capture_stdout=False,
        capture_stderr=False,
        capture_hardware_metrics=False
    )

    stock_prediction_data = predict_stock_prices()

    handle_predictions(stock_prediction_data)
