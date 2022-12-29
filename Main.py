import threading
import pandas
from dotenv import load_dotenv
import os
import time
from datetime import datetime
import warnings
import neptune.new as neptune
import matplotlib
from Individual_LSTM import IndividualLSTM


def start(ticker):
    data = get_data(ticker)
    IndividualLSTM(ticker, data, run)


def get_data(ticker):
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={ticker}&apikey={ALPHA_VANTAGE_TOKEN}&datatype=csv&outputsize=full"
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
    run = neptune.init(
        project="elitheknight/Stock-Prediction",
        api_token=NEPTUNE_API_TOKEN,
        custom_run_id=custom_id
    )

    blacklist = ['AMZN', 'GOOG', 'NVDA', 'TSLA', 'NKE']
    # max to run at once is 20 because of pandas figure limits
    listoftickers = ['NKE', 'TMO']
    threads = []
    get_new_data = True
    for ticker_ in listoftickers:
        if ticker_ in blacklist:
            continue
        threads.append(threading.Thread(target=start, args=(ticker_, run)))

    for thread in threads:
        thread.start()
        time.sleep(20)

    for thread in threads:
        thread.join()
