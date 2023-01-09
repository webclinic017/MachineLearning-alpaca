import os

from robin_stocks import robinhood as r
import pyotp
from dotenv import load_dotenv

load_dotenv()
username = os.getenv('ROBINHOOD-USERNAME')
password = os.getenv('ROBINHOOD-PASSWORD')
robin_totp = os.getenv('ROBINHOOD-TOTP')

totp = pyotp.TOTP(robin_totp).now()
print("Current OTP:", totp)
login = r.login(username, password, mfa_code=totp)


def get_my_stocks():
    """
    :return: dict of all owned stocks
    """
    return r.build_holdings()


def get_market():
    return r.get_all_positions()


def get_stock(ticker):
    market = r.get_all_positions()
    if ticker in market:
        return market[ticker]
    return None


def buy_stock(ticker, price):
    response = r.order_buy_fractional_by_price(symbol=ticker, amountInDollars=price)
    return response


def sell_stock(ticker, price=None):
    if price is None:
        total = get_my_stocks()[ticker]['price']
    else:
        total = price
    response = r.order_sell_fractional_by_price(ticker, total)
    return response
