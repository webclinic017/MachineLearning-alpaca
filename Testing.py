import neptune.new as neptune
import pandas
from dotenv import load_dotenv
import os
import warnings
import matplotlib
from datetime import datetime
from multiprocessing import Pool
import time
from Individual_LSTM import IndividualLSTM
import Trading
import numpy as np


def begin(ticker: str, id: str, NAT):
    """
    function multiprocessing processes run. Makes an IndividualLSTM object and runs stock predictions
    :param ticker: str, ticker of stock to predict
    :param id: str, neptune run custom id
    :param NAT: Neptune API Token
    :param AVT: AlphaVantage Token
    :return: stock prediction data for the next day: tuple(ticker, predicted_price, change_price, change_percent)
    """
    new_run = neptune.init(
        project="elitheknight/Stock-Testing",
        api_token=NAT,
        custom_run_id=id,
        capture_stdout=False,
        capture_stderr=False,
        capture_hardware_metrics=False
    )

    data = pandas.read_csv(f"Data/{ticker}_data.csv", index_col=0)[2:]

    stock = IndividualLSTM(ticker, data, new_run)

    # new_run[f"Predictions/{ticker}/LSTM"].log((ticker, stock.open, stock.close))

    return ticker, stock.open, stock.close


def create_orders(stock_prediction_data):
    stocks_to_invest = []   # (ticker, percent_change, buy_open, sell_open)

    # find percent change for best buy-sell combo
    for ticker, i in stock_prediction_data.items():
        buy = min(i[1]['predicted_price'], i[2]['predicted_price'])
        sell = max(i[1]['second_predicted_price'], i[2]['second_predicted_price'])
        percent_change = (sell/buy - 1) * 100
        if percent_change < 0:
            continue
        sell_open = i[1]['second_predicted_price'] > i[2]['second_predicted_price']
        buy_open = i[1]['predicted_price'] < i[2]['predicted_price']
        stocks_to_invest.append((ticker, percent_change, buy_open, sell_open))

    stocks_to_invest = sorted(stocks_to_invest, key=lambda x: x[1], reverse=True)
    stocks_to_invest = stocks_to_invest[:int(len(stocks_to_invest) * 2 / 3)]  # take top 2/3 of positive stocks

    if not stocks_to_invest:
        run["status"].log("ERROR - stocks_to_invest is empty")
        return {}
    run["stocks_to_invest (ticker, percent_change, buy_open, sell_open)"].log(stocks_to_invest)

    orders = []  # (ticker, $ amount to buy, buy_open: bool, sell_open: bool) if buy_open/sell_open is false, buy or sell the close
    open_stocks_to_invest = [i for i in stocks_to_invest if i[2]]
    close_stocks_to_invest = [i for i in stocks_to_invest if not i[2]]

    total_money = 100

    money_for_open = 50
    money_for_close = 50

    percent_open = sum(i[1] for i in open_stocks_to_invest)
    percent_close = sum(i[1] for i in close_stocks_to_invest)

    floating_additions = 0  # max of my equity is 10%, any more gets divided among remaining investments
    remaining_money = money_for_open
    for i in range(len(open_stocks_to_invest)):
        amount = round(money_for_open * (open_stocks_to_invest[i][1] / percent_open) + floating_additions, 5)
        if amount / total_money > 0.1:
            new_amount = total_money * 0.1
            floating_additions += (amount - new_amount) / (len(open_stocks_to_invest) - i)
            amount = new_amount
        if amount < 1.00:
            amount = 1.00

        if amount > remaining_money:
            break

        remaining_money -= amount

        orders.append((stocks_to_invest[i][0], amount, stocks_to_invest[i][2], stocks_to_invest[i][3]))

    money_for_close += remaining_money

    floating_additions = 0  # max of my equity is 10%, any more gets divided among remaining investments
    remaining_money = money_for_close
    for i in range(len(close_stocks_to_invest)):
        amount = round(money_for_close * (close_stocks_to_invest[i][1] / percent_close) + floating_additions, 5)
        if amount / total_money > 0.1:
            new_amount = total_money * 0.1
            floating_additions += (amount - new_amount) / (len(close_stocks_to_invest) - i)
            amount = new_amount
        if amount < 1.00:
            amount = 1.00

        if amount > remaining_money:
            break

        remaining_money -= amount

        orders.append((stocks_to_invest[i][0], amount, stocks_to_invest[i][2], stocks_to_invest[i][3]))

    print(f"Orders: {orders}")

    orders_as_dict = {i[0]: i for i in orders}
    run[f"orders_to_buy"].log(orders_as_dict)
    return orders_as_dict


def predict_stock_prices(test: bool = False):
    """
    Loads tickers.txt list of stock tickers and creates Individual_LSTM objects
    to predict next day stock prices and divides the tasks among a processing pool
    :return: list[tuples] with stock data (ticker: str, predicted_price: float, change_price: float, change_percent: float)
    """
    blacklist = ['AMZN', 'GOOG', 'NVDA', 'TSLA', 'NKE']
    # max to run at once is 20 because of pandas figure limits
    with open('tickers.txt', 'r') as f:
        lines = f.readlines()
    listOfTickers = [(line.strip(), custom_id, NEPTUNE_API_TOKEN) for line in lines if
                     line.strip() not in blacklist]

    tickers_to_run = listOfTickers.copy()
    if test:
        stock_predictions = []
        # for i in range(len(listOfTickers) // 4):
        for i in range(2):
            pool = Pool(processes=4)
            result = pool.starmap_async(begin, tickers_to_run[4*i:4*(i+1)])
            tickers_to_run = tickers_to_run[4:]
            results = result.get()
            stock_predictions.extend(results)
            pool.close()
            time.sleep(60)

        stock_dict_predictions = {i[0]: i for i in stock_predictions}

        print(stock_dict_predictions)
        return stock_dict_predictions

    num_processes = 6
    num_tickers = len(listOfTickers)
    groups = num_tickers // num_processes + 1
    stock_predictions = []

    for i in range(groups):
        # multiprocessing
        pool = Pool(processes=num_processes)
        print(f"\tstarting wave {i + 1}/{groups}")
        result = pool.starmap_async(begin, tickers_to_run[:num_processes])
        tickers_to_run = tickers_to_run[num_processes:]
        # start_time = time.time()
        results = result.get()
        stock_predictions.extend(results)
        # print(f"delay: {time.time() - start_time}")
        pool.close()

    # logs the data to neptune
    run[f"all_prediction_data/Data"].log(stock_predictions)

    # change predictions to dictionary {ticker: (ticker, open, close)}

    stock_dict_predictions = {i[0]: i for i in stock_predictions}

    return stock_dict_predictions


def calculate_prediction_error(predictions):
    # errors = {}
    first_level_mapes = 0.0
    second_level_mapes = 0.0
    average_mapes = 0.0
    for ticker, values in predictions.items():
        dataframe = pandas.read_csv(f"Data/{ticker}_data.csv")
        true_close_1, true_close_2, true_open_1, true_open_2 = dataframe['close'][1], dataframe['close'][0], dataframe['open'][1], dataframe['open'][0]
        close_mape_1 = np.mean(np.abs((true_close_1 - values[1]['predicted_price']) / true_close_1)) * 100
        close_mape_2 = np.mean(np.abs((true_close_2 - values[1]['second_predicted_price']) / true_close_2)) * 100
        open_mape_1 = np.mean(np.abs((true_open_1 - values[2]['predicted_price']) / true_open_1)) * 100
        open_mape_2 = np.mean(np.abs((true_open_2 - values[2]['second_predicted_price']) / true_open_2)) * 100
        run[f"all_prediction_error/{ticker}"].log(f"close_mape_1: {close_mape_1}, close_mape_2: {close_mape_2}, open_mape_1: {open_mape_1}, open_mape_2: {open_mape_2}")
        first_level = (close_mape_1 + close_mape_2) / 2
        second_level = (open_mape_2 + open_mape_1) / 2
        average = (first_level + second_level) / 2
        first_level_mapes += first_level
        second_level_mapes += second_level
        average_mapes += average
        run[f"prediction_error"].log(f"{ticker}, first_level: {first_level}, second_level: {second_level}, average_mape: {average}")

    average_mapes = average_mapes / len(predictions)
    second_level_mapes = second_level_mapes / len(predictions)
    first_level_mapes = first_level_mapes / len(predictions)
    run[f"averages"].log(f"first_level_average: {first_level_mapes}, second_level_average: {second_level_mapes}, total_average: {average_mapes}")
    print(f"first_level_average: {first_level_mapes}, second_level_average: {second_level_mapes}, total_average: {average_mapes}")


def calculate_order_results(orders):
    start_price = 0
    price_change_amount = 0

    for ticker, values in orders.items():
        dataframe = pandas.read_csv(f"Data/{ticker}_data.csv")
        amount = values[1]
        start_price += amount
        if values[2]:
            buy = dataframe['open'][1]
        else:
            buy = dataframe['close'][1]

        if values[3]:
            sell = dataframe['open'][0]
        else:
            sell = dataframe['close'][0]

        percent_change = (sell/buy - 1) * 100
        price_change = amount * percent_change / 100
        price_change_amount += price_change

        run[f"order_results"].log(f"{ticker}, percent_change: {percent_change}, price_change: {price_change}")

    end_price = start_price + price_change_amount
    run["trade_results"].log(f"invested {start_price}, changed by {price_change_amount}, changed by price {(end_price/start_price - 1) * 100} ended with price {start_price+price_change_amount}")


if __name__ == '__main__':
    load_dotenv()
    NEPTUNE_API_TOKEN = os.getenv('NEPTUNE-API-TOKEN')
    ALPHA_VANTAGE_TOKEN = os.getenv('ALPHA-VANTAGE-API-TOKEN')

    # disables unnecessary warnings
    pandas.options.mode.chained_assignment = None  # default='warn'
    matplotlib.use('SVG')
    warnings.filterwarnings(action='ignore', category=UserWarning)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # tensorflow cpu not found error
    dateTimeObj = datetime.now()
    custom_id = 'EXP-' + dateTimeObj.strftime("%d-%b-%Y-(%H:%M:%S)")
    run = neptune.init(
        project="elitheknight/Stock-Testing",
        custom_run_id=custom_id,
        api_token=NEPTUNE_API_TOKEN,
        capture_stdout=False,
        capture_stderr=False,
        capture_hardware_metrics=False
    )

    stock_predictions = predict_stock_prices()
    print(f"Predictions: {stock_predictions}")
    try:
        calculate_prediction_error(stock_predictions)
    except Exception as e:
        print(f'error in calc prediction error: {e}')
    orders_to_buy = create_orders(stock_predictions)
    try:
        calculate_order_results(orders_to_buy)
    except Exception as e:
        print(f'error in calc order results error: {e}')

