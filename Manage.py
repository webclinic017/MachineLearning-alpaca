import pandas
import time
import neptune.new as neptune
from multiprocessing import Pool
from Individual_LSTM import IndividualLSTM
from Trading import Trader
import Trading
from datetime import datetime
from threading import Timer


def begin(ticker: str, id: str, NAT, AVT):
    """
    function multiprocessing processes run. Makes an IndividualLSTM object and runs stock predictions
    :param ticker: str, ticker of stock to predict
    :param id: str, neptune run custom id
    :param NAT: Neptune API Token
    :param AVT: AlphaVantage Token
    :return: stock prediction data for the next day: tuple(ticker, predicted_price, change_price, change_percent)
    """
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
    # print(f"{ticker} complete")
    return ticker, stock.predicted_price, stock.change_price, stock.change_percentage


def get_data(ticker: str, AVT, dataset_size: int):
    """
    gets stock csv data, exits the process if it fails
    :param ticker: str, stock ticker
    :param AVT: AlphaVantage API token
    :param dataset_size: int, # of days of data to keep
    :return: DataFrame of csv data
    """
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={ticker}&apikey={AVT}&datatype=csv&outputsize=full"
    try:
        return pandas.read_csv(url)[:dataset_size]
    except Exception as e:
        print(f"error in getting data for ticker: {ticker}, url: {url}, error: {e}")
        exit(1)


def get_data_to_file(ticker: str, AVT, dataset_size: int):
    """
    gets stock csv data to file and returns it, exits the process if it fails
    :param ticker: str, stock ticker
    :param AVT: AlphaVantage API token
    :param dataset_size: int, # of days of data to keep
    :return: DataFrame of csv data
    """
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={ticker}&apikey={AVT}&datatype=csv&outputsize=full"

    try:
        df = pandas.read_csv(url)[:dataset_size]
    except Exception as e:
        print(f"error in getting data for ticker: {ticker}, url: {url}, error: {e}")
        exit(1)
        return

    df.to_csv(f"Data/{ticker}_data.csv")

    return df


class Manager:

    def __init__(self, NAT, AVT):
        # create neptune run with custom_id, so it can be referenced in multiprocessing processes
        dateTimeObj = datetime.now()
        self.custom_id = 'EXP-' + dateTimeObj.strftime("%d-%b-%Y-(%H:%M:%S)")
        self.run = neptune.init(
            project="elitheknight/Stock-Prediction",
            api_token=NAT,
            custom_run_id=self.custom_id,
            capture_stdout=False,
            capture_stderr=False,
            capture_hardware_metrics=False
        )

        self.NEPTUNE_API_TOKEN = NAT
        self.ALPHA_VANTAGE_TOKEN = AVT
        self.trader = None
        self.orders_for_day = None
        self.stock_prediction_data = None

    def schedule(self):
        """
        Starts trading each day at the same time
        :return:
        """
        while True:
            dt = datetime.now()
            hr = dt.hour
            if hr == 8:
                # one day minus one minute of seconds
                timer = Timer(86340, self.start)
                timer.run()
                time.sleep(3700)
            else:
                time.sleep(60)

    def start(self):

        print(f"starting trade bot at: {datetime.now()}")
        self.start_trader()
        if self.trader is None:
            time.sleep(72000)    # sleep for 20 hrs if closed so it doesn't go back to the while loop (more efficient)
            return

        self.stock_prediction_data = self.predict_stock_prices()

        self.create_orders()

    def create_orders(self):
        stocks_to_invest = []

        # add all positive stocks to list
        for entry in self.stock_prediction_data:
            if entry[3] <= 0:
                break
            stocks_to_invest.append(entry)

        stocks_to_invest = stocks_to_invest[:int(len(stocks_to_invest) * 2 / 3)]  # take top 2/3 of positive stocks

        tickers = [entry[0] for entry in stocks_to_invest]
        previous_close_prices = Trading.get_last_close_price(tickers)
        current_prices = Trading.get_stocks_current_price(tickers)
        assert (len(current_prices) == len(previous_close_prices) == len(tickers))

        indices_to_remove = []
        for i in range(len(tickers)):
            if max(previous_close_prices[i], current_prices[i]) / min(previous_close_prices[i], current_prices[i]) > \
                    stocks_to_invest[i][3]:
                indices_to_remove.insert(0, i)      # too much activity since last close, prediction might not still be accurate

        for i in indices_to_remove:
            del stocks_to_invest[i]     # remove unstable stocks

        orders = []  # (ticker, $ amount to buy)

        total_percent = sum([percent[3] for percent in stocks_to_invest])
        assert (total_percent > 0)

        money_to_invest = Trading.get_my_stocks()['equity']     # money in robinhood
        floating_additions = 0  # max of my equity is 10%, any more gets divided among remaining investments
        for i in range(len(stocks_to_invest)):
            amount = money_to_invest * (stocks_to_invest[i][3] / total_percent) + floating_additions
            if amount / money_to_invest > 0.1:
                new_amount = money_to_invest * 0.1
                floating_additions += (amount - new_amount) / (len(stocks_to_invest) - i)
                amount = new_amount

            orders.append((stocks_to_invest[i][0], amount))

        self.orders_for_day = orders

    def predict_stock_prices(self):
        """
        Loads tickers.txt list of stock tickers and creates Individual_LSTM objects
        to predict next day stock prices and divides the tasks among a processing pool
        :return: list[tuples] with stock data (ticker: str, predicted_price: float, change_price: float, change_percent: float)
        """
        blacklist = ['AMZN', 'GOOG', 'NVDA', 'TSLA', 'NKE']
        # max to run at once is 20 because of pandas figure limits
        with open('tickers.txt', 'r') as f:
            lines = f.readlines()
        listOfTickers = [(line.strip(), self.custom_id, self.NEPTUNE_API_TOKEN, self.ALPHA_VANTAGE_TOKEN) for line in lines if
                         line.strip() not in blacklist]

        tickers_to_run = listOfTickers.copy()
        num_processes = 4
        num_tickers = len(listOfTickers)
        groups = num_tickers // num_processes + 1
        stock_predictions = []

        for i in range(groups):
            # multiprocessing
            pool = Pool(processes=num_processes)
            print(f"\tstarting wave {i + 1}/{groups}")
            result = pool.starmap_async(begin, tickers_to_run[:num_processes])
            tickers_to_run = tickers_to_run[num_processes:]
            # I have to wait at least 60 seconds to limit my API calls (max 5/min and 500/day)
            time.sleep(60)
            # start_time = time.time()
            results = result.get()
            stock_predictions.extend(results)
            # print(f"delay: {time.time() - start_time}")
            pool.close()

        # sorts the predictions by predicted percent change from highest to lowest
        stock_predictions = sorted(stock_predictions, key=lambda x: x[3], reverse=True)
        # logs the data to neptune
        self.run[f"all_prediction_data/Data"].log(stock_predictions)
        return stock_predictions

    def start_trader(self):
        trader = Trader()
        if trader.hours is None:
            return None
        self.trader = trader
