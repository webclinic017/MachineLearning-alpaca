import time

import pandas
with open('tickers.txt', 'r') as f:
    tickers = f.readlines()


def get_data_to_file(ticker: str, dataset_size: int):
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={ticker}&apikey=ZLGDWBU4HK5QHIXT&datatype=csv&outputsize=full"

    try:
        df = pandas.read_csv(url)[:dataset_size]
    except Exception as e:
        print(f"error in getting data for ticker: {ticker}, url: {url}, error: {e}")
        exit(1)

    df.to_csv(f"Data/{ticker}_data.csv")

    return df


for i in tickers:
    get_data_to_file(i.strip(), 400)
    time.sleep(15)
