import StockBot
import pandas


class Individual:

    def __init__(self, lstm_pars, ticker):
        self.ticker = ticker
        self.lstm_pars = lstm_pars
        self.stock_bot = StockBot.Stock(ticker, None, test_lstm=True, data=pandas.read_csv(f"Data/{ticker}_data.csv"))

    def calculate_fitness(self):
        mape = self.stock_bot.begin_lstm(self.lstm_pars)
        print(f"DONE, params: {self.lstm_pars}, stock: {self.ticker}, mape: {mape}")
        return mape

    def calc_test(self):
        return self.stock_bot.begin_lstm(self.lstm_pars)
    