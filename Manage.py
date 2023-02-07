import pandas
import time
import neptune.new as neptune
from multiprocessing import Pool
from Individual_LSTM import IndividualLSTM
from Trading import Trader
import Trading
from datetime import datetime, date
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
    return ticker, stock.open, stock.close


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

    df.to_csv(f"Data/{ticker}_data.csv")

    return df


def start_trader():
    trader = Trader()
    if trader.hours is None:
        return None
    return trader


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
        self.orders_for_day = None
        self.stock_prediction_data = None
        self.sell_for_day = None
        self.schedule()

    def schedule(self):
        """
        Starts trading each day at the same time
        :return:
        """
        while True:
            dt = datetime.now()
            hr = dt.hour
            # start each day at 8 am, if not between 8-9, sleep an appropriate amount of time
            if hr == 8:
                trader = start_trader()
                if trader is None:
                    print(f"sleeping for{23*3600} seconds")
                    time.sleep(23*3600)
                    continue
                self.run_day(trader)
                print(f"sleeping for{3601} seconds")
                time.sleep(3601)
            elif hr > 8:
                print(f"sleeping for{(31-hr)*3600} seconds")
                time.sleep((31-hr)*3600)
            elif hr < 7:
                print(f"sleeping for{3600} seconds")
                time.sleep(3600)
            else:
                time.sleep(60)

    def run_day(self, trader):
        print(f"starting trade bot at: {datetime.now()}")
        self.run["status"].log(f"starting trade bot at: {datetime.now()}")

        if trader.tomorrow is None:  # then today is friday or holiday the next day so sell all stocks to clean up
            print(f"stock market is not open tomorrow, selling all stocks at close to clean up")
            self.run["status"].log(f"stock market is not open tomorrow, selling all stocks at close to clean up")

            sell_open = [i for i in self.sell_for_day if i[2] is True] if self.sell_for_day is not None else []

            while True:
                current_time = datetime.timestamp(datetime.utcnow())
                if current_time > trader.hours[0]:
                    print(f"making trades at: {datetime.now()}")
                    self.run["status"].log(f"making trades at: {datetime.now()}")
                    self.execute_orders(sell_open, buying=False)
                    break
                else:
                    time.sleep(60)

            # sleep till close
            time_before_close = int(trader.hours[1] - trader.hours[0])
            print(f"sold stocks, sleeping for: {time_before_close - 1200} seconds")
            self.run["status"].log(f"sold stocks, sleeping for: {time_before_close - 1200} seconds")
            time.sleep(int(time_before_close) - 1200)

            print(f"selling all stocks at: {datetime.now()}")
            self.run["status"].log(f"selling all stocks at: {datetime.now()}")
            self.record_order_details(Trading.sell_all_stocks(), buying=False)

            self.orders_for_day = None
            self.sell_for_day = None
            return

        self.stock_prediction_data = self.predict_stock_prices()

        print(f"making orders")
        self.run["status"].log("making orders")
        self.orders_for_day = self.create_orders()

        buy_open = [i for i in self.orders_for_day if i[2] is True]
        buy_close = [i for i in self.orders_for_day if i[2] is False]

        sell_open = [i for i in self.sell_for_day if i[2] is True] if self.sell_for_day is not None else []
        sell_close = [i for i in self.sell_for_day if i[2] is False] if self.sell_for_day is not None else []

        bought = []
        # wait until market is open and then execute all open orders
        while True:
            current_time = datetime.timestamp(datetime.utcnow())
            if current_time > trader.hours[0]:
                print(f"making trades at: {datetime.now()}")
                self.run["status"].log(f"making trades at: {datetime.now()}")
                self.execute_orders(sell_open, buying=False)
                bought.extend(self.execute_orders(buy_open))
                break
            else:
                time.sleep(60)

        # sleep till close
        time_before_close = int(trader.hours[1] - trader.hours[0])
        print(f"bought stocks, sleeping for: {time_before_close - 1200} seconds")
        self.run["status"].log(f"bought stocks, sleeping for: {time_before_close - 1200} seconds")
        time.sleep(int(time_before_close) - 1200)

        # execute closing orders
        print(f"making trades at: {datetime.now()}")
        self.run["status"].log(f"making trades at: {datetime.now()}")
        self.execute_orders(sell_close, buying=False)
        bought.extend(self.execute_orders(buy_close))

        # shift orders to be sold, make orders None
        self.run["sell_tomorrow (ticker, quantity, sell_open)"].log(bought)
        self.sell_for_day = bought
        self.orders_for_day = None

    def create_orders(self):
        stocks_to_invest = []

        # find percent change for best buy-sell combo
        for i in range(len(self.stock_prediction_data)):
            buy = min(self.stock_prediction_data[i][1]['predicted_price'], self.stock_prediction_data[i][2]['predicted_price'])
            sell = max(self.stock_prediction_data[i][1]['second_predicted_price'], self.stock_prediction_data[i][2]['second_predicted_price'])
            percent_change = ((sell/buy - 1) * 100,)
            self.stock_prediction_data[i] = self.stock_prediction_data[i] + percent_change

        self.stock_prediction_data = sorted(self.stock_prediction_data, key=lambda x: x[3], reverse=True)

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
            if max(float(previous_close_prices[i]), float(current_prices[i])) / min(float(previous_close_prices[i]), float(current_prices[i])) > \
                    stocks_to_invest[i][3]:
                indices_to_remove.insert(0, i)      # too much activity since last close, prediction might not still be accurate

        for i in indices_to_remove:
            del stocks_to_invest[i]     # remove unstable stocks

        orders = []  # (ticker, $ amount to buy, buy_open: bool, sell_open: bool) if buy_open/sell_open is false, buy or sell the close

        total_percent = sum([percent[3] for percent in stocks_to_invest])
        assert (total_percent > 0)

        user_info = Trading.get_user_info()
        money_to_invest = float(user_info['cash']) if 'cash' in user_info else float(user_info['equity'])     # money in robinhood
        # limit to 1/3 of spendable money so I can invest each day with settlement periods
        # TODO: Delete when I switch back to instant account
        if money_to_invest * 3 > float(user_info['equity']):
            money_to_invest = float(user_info['equity']) / 3

        floating_additions = 0  # max of my equity is 10%, any more gets divided among remaining investments
        for i in range(len(stocks_to_invest)):
            amount = round(money_to_invest * (stocks_to_invest[i][3] / total_percent) + floating_additions, 5)
            if amount / money_to_invest > 0.1:
                new_amount = money_to_invest * 0.1
                floating_additions += (amount - new_amount) / (len(stocks_to_invest) - i)
                amount = new_amount

            buy_time = stocks_to_invest[i][1]['predicted_price'] < stocks_to_invest[i][2]['predicted_price']
            sell_time = stocks_to_invest[i][1]['second_predicted_price'] > stocks_to_invest[i][2]['second_predicted_price']

            orders.append((stocks_to_invest[i][0], amount, buy_time, sell_time))

        self.run[f"orders_to_buy"].log(orders)
        return orders

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

        # logs the data to neptune
        self.run[f"all_prediction_data/Data"].log(stock_predictions)
        return stock_predictions

    def record_order_details(self, records, buying: bool):
        if buying:
            self.run[f"buying_records/{str(date.today())}"].log(records)
            filename = 'buying_records.txt'
        else:
            self.run[f"selling_records/{str(date.today())}"].log(records)
            filename = 'selling_records.txt'
        with open(filename, 'a') as f:
            for record in records:
                f.write(f"{record}/n")

    def execute_orders(self, orders, buying=True):
        # if Trading.get_my_stocks():
        #     # if I still have any stocks, sell them so I can make all my orders
        #     selling_info = Trading.sell_all_stocks()
        #     self.record_order_details(selling_info, buying=False)
        #     time.sleep(120)

        details = []
        results = []

        if buying:
            for order in orders:
                detail = Trading.buy_stock(ticker=order[0], price=order[1])
                details.append(detail)
                if 'quantity' in detail:
                    results.append((order[0], detail['quantity'], order[3]))
        else:
            for order in orders:
                details.append(Trading.sell_stock_by_quantity(ticker=order[0], quantity=order[1]))

        self.record_order_details(details, buying=buying)
        return results
