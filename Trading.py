import os
import time
from typing import Union
from uuid import uuid4

import robin_stocks.robinhood
from robin_stocks import robinhood as r
import pyotp
from dotenv import load_dotenv
from datetime import date, datetime, timedelta
import numpy as np
from robin_stocks.robinhood.urls import orders_url
from robin_stocks.helper import request_post


def get_user_info():
    return r.build_user_profile()


def get_my_stocks():
    """
    gets dict of all owned stocks
    :return: dict of all owned stocks
    """
    return r.build_holdings()


def get_stocks_current_price(stocks: Union[str, list]):
    """
    gets most recent price of stock(s)
    :param stocks: stock symbol or list of stock symbols
    :return: current/latest price of stock(s)
    """
    return r.stocks.get_latest_price(stocks)


def buy_stock(ticker: str, price: float):
    """
    Buys stock by price
    :param ticker: str, stock symbol to buy
    :param price: float, amount in $ to buy
    :return: dict of order id and information
    """
    # response = r.order_buy_fractional_by_price(symbol=ticker, amountInDollars=price)
    # if response is not None and 'id' not in response:    # try again if didn't work
    #     time.sleep(5)
    #     response = r.order_buy_fractional_by_price(symbol=ticker, amountInDollars=price)
    # if response is not None and 'id' not in response:    # try again if didn't work
    #     time.sleep(5)
    #     response = r.order_buy_fractional_by_price(symbol=ticker, amountInDollars=price)
    #
    # return response

    fractional_shares = price / float(robin_stocks.robinhood.get_latest_price(ticker)[0])
    payload = {
        'account': robin_stocks.robinhood.load_account_profile(info='url'),
        'instrument': robin_stocks.robinhood.get_instruments_by_symbols(ticker, info='url')[0],
        'order_form_version': "2",
        'preset_percent_limit': "0.05",
        'symbol': ticker,
        'price': price,
        'quantity': fractional_shares,
        'ref_id': str(uuid4()),
        'type': "limit",
        'time_in_force': 'gfd',
        'trigger': "immediate",
        'side': "buy",
        'extended_hours': False
    }

    url = orders_url()
    print(url)
    print(payload)
    data = request_post(url, payload, json=True, jsonify_data=True)

    return data


def buy_stock_by_quantity(ticker: str, quantity: float):
    response = r.order_buy_fractional_by_quantity(symbol=ticker, quantity=quantity)
    if response is not None and 'id' not in response:  # try again if didn't work
        time.sleep(5)
        response = r.order_buy_fractional_by_quantity(symbol=ticker, quantity=quantity)
    if response is not None and 'id' not in response:  # try again if didn't work
        time.sleep(5)
        response = r.order_buy_fractional_by_quantity(symbol=ticker, quantity=quantity)

    return response


def sell_stock_by_quantity(ticker: str, quantity: float = None):
    """
    Sells stock by price
    :param ticker: str, stock symbol to sell
    :param quantity: float, quantity in shares to sell, if None than it sells all owned of that stock
    :return: dict of order id and information
    """
    if quantity is None:
        return r.order_sell_fractional_by_quantity(ticker, r.build_holdings[ticker]['quantity'])
    else:
        return r.order_sell_fractional_by_quantity(ticker, quantity)


def sell_all_stocks():
    """
    Completely sells all held stocks
    :return: the details of all the sales to be recorded
    """
    details = []
    for stock, info in r.build_holdings().items():
        details.append(r.order_sell_fractional_by_quantity(stock, info['quantity']))
        time.sleep(10)
    return details


def get_last_close_price(tickers: Union[str, list]):
    assert len(tickers) > 0
    if isinstance(tickers, str):
        tickers = [tickers]
    if len(tickers) > 50:
        closes = []
        for i in range(len(tickers)//50 + 1 if len(tickers) % 50 != 0 else len(tickers)//50):
            closes.extend(r.stocks.get_stock_historicals(tickers[i*50: min((i+1)*50, len(tickers))], interval="day", span="week", info='close_price'))
    else:
        closes = r.stocks.get_stock_historicals(tickers, interval="day", span="week", info='close_price')
    assert(isinstance(closes, list))
    days = len(closes)//len(tickers)
    closes = closes[days-1::days]
    return closes


def get_last_close_percent_change(tickers: Union[str, list]):
    assert len(tickers) > 0
    if isinstance(tickers, str):
        tickers = [tickers]
    if len(tickers) > 50:
        closes = []
        for i in range(len(tickers)//50 + 1 if len(tickers) % 50 != 0 else len(tickers)//50):
            closes.extend(r.stocks.get_stock_historicals(tickers[i*50: min((i+1)*50, len(tickers))], interval="day", span="week", info='close_price'))
    else:
        closes = r.stocks.get_stock_historicals(tickers, interval="day", span="week", info='close_price')
    assert(isinstance(closes, list))
    y_size = len(r.stocks.get_stock_historicals('AAPL', interval="day", span="week", info='close_price'))
    closes = np.array(closes, dtype='float64').reshape((len(tickers), y_size))
    closes = ((closes[:, -1] / closes[:, -2]) - 1) * 100
    return closes


def check_market_open(dateToday=None):
    if not dateToday:
        dateToday = str(date.today())
    market_data = r.markets.get_market_hours('XNYS', dateToday)

    if not market_data['is_open']:
        print(f"The New York Stock Exchange is NOT open today: {dateToday}")
        return None

    opens = datetime.timestamp(datetime.strptime(market_data['opens_at'], "%Y-%m-%dT%H:%M:%SZ"))
    closes = datetime.timestamp(datetime.strptime(market_data['closes_at'], "%Y-%m-%dT%H:%M:%SZ"))
    return opens, closes


def check_market_tomorrow():
    tomorrow = str(date.today() + timedelta(days=1))
    market_data = r.markets.get_market_hours('XNYS', tomorrow)

    if not market_data['is_open']:
        return None

    opens = datetime.timestamp(datetime.strptime(market_data['opens_at'], "%Y-%m-%dT%H:%M:%SZ"))
    closes = datetime.timestamp(datetime.strptime(market_data['closes_at'], "%Y-%m-%dT%H:%M:%SZ"))
    return opens, closes


def logout():
    r.logout()


def check_order(order_id):
    try:
        order = r.orders.get_stock_order_info(order_id)
        if order['state'] == 'filled':
            return 1
        elif order['state'] == 'unconfirmed':
            return 2
        elif order['state'] == 'cancelled':
            return 3
    except:
        return 0
    return 0


def get_fundamentals(tickers: Union[str, list]):
    assert len(tickers) > 0
    fundamentals = r.get_fundamentals(tickers)
    return fundamentals


def get_historicals(tickers: Union[str, list], span: str = 'week', interval: str = 'day'):
    assert len(tickers) > 0
    hist = r.get_stock_historicals(tickers, span=span, interval=interval)
    return hist


class Trader:

    def __init__(self):
        load_dotenv()
        username = os.getenv('ROBINHOOD-USERNAME')
        password = os.getenv('ROBINHOOD-PASSWORD')
        robin_totp = os.getenv('ROBINHOOD-TOTP')
        # verify
        totp = pyotp.TOTP(robin_totp).now()

        # login
        self.login = r.login(username, password, mfa_code=totp)
        self.account = r.load_account_profile()
        self.hours = check_market_open()
        self.tomorrow = check_market_tomorrow()

    def __exit__(self, exc_type, exc_val, exc_tb):
        logout()
