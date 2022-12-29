import threading
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
    data = get_data(ticker, AVT)
    run = neptune.init(
        project="elitheknight/Stock-Prediction",
        api_token=NAT,
        custom_run_id=id,
        capture_stdout=False,
        capture_stderr=False,
        capture_hardware_metrics=False
    )
    IndividualLSTM(ticker, data, run)
    print(f"{ticker} complete")


def get_data(ticker, AVT):
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={ticker}&apikey={AVT}&datatype=csv&outputsize=full"
    try:
        return pandas.read_csv(url)
    except Exception as e:
        print(f"error in getting data for ticker: {ticker}, url: {url}, error: {e}")
        exit(1)


if __name__ == '__main__':
    load_dotenv()
    NEPTUNE_API_TOKEN = os.getenv('NEPTUNE-API-TOKEN')
    ALPHA_VANTAGE_TOKEN = os.getenv('ALPHA-VANTAGE-API-TOKEN')
    dataset_size = 200

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

    blacklist = ['AMZN', 'GOOG', 'NVDA', 'TSLA', 'NKE']
    # max to run at once is 20 because of pandas figure limits
    with open('tickers.txt', 'r') as f:
        lines = f.readlines()
    listOfTickers = [(line.strip(), custom_id, NEPTUNE_API_TOKEN, ALPHA_VANTAGE_TOKEN) for line in lines if line.strip() not in blacklist]

    # multiprocessing
    tickers_to_run = listOfTickers.copy()
    max_per_min = 4
    num_processes = 4
    num_tickers = len(listOfTickers)
    groups = num_tickers // max_per_min
    pool = Pool(processes=num_processes)
    for i in range(groups):
        print(f"starting wave {i+1}/{groups}")
        result = pool.starmap_async(begin, tickers_to_run[:max_per_min])
        results = result.get()
        tickers_to_run = tickers_to_run[max_per_min:]
        time.sleep(60)

    pool.close()

    # # threading
    # threads = []
    #
    # for ticker_ in listOfTickers:
    #     if ticker_ in blacklist:
    #         continue
    #     threads.append(threading.Thread(target=begin, args=(ticker_,)))
    #
    # for thread in threads:
    #     thread.start()
    #     time.sleep(20)
    #
    # for thread in threads:
    #     thread.join()
