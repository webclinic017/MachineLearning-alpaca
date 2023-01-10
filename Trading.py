import os
from typing import Union
from robin_stocks import robinhood as r
import pyotp
from dotenv import load_dotenv


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
    response = r.order_buy_fractional_by_price(symbol=ticker, amountInDollars=price)
    return response


def sell_stock(ticker: str, price: float = None):
    """
    Sells stock by price
    :param ticker: str, stock symbol to sell
    :param price: float, amount in $ to sell, if None than it sells all owned stock
    :return: dict of order id and information
    """
    if price is None:
        total = get_my_stocks()[ticker]['price']
    else:
        total = price
    response = r.order_sell_fractional_by_price(ticker, total)
    return response


class Trader:

    def __init__(self):
        load_dotenv()
        username = os.getenv('ROBINHOOD-USERNAME')
        password = os.getenv('ROBINHOOD-PASSWORD')
        robin_totp = os.getenv('ROBINHOOD-TOTP')
        # verify
        totp = pyotp.TOTP(robin_totp).now()
        # print("Current OTP:", totp)
        # login
        self.login = r.login(username, password, mfa_code=totp)


