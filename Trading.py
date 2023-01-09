from robin_stocks import robinhood as r
import pyotp


class Trader:

    def __init__(self, username, password):
        self.login = r.login(username, password)
        totp = pyotp.TOTP("My2factorAppHere").now()
        print("Current OTP:", totp)


