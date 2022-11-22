import StockBot

TARGET = 0


class Individual:

    def __init__(self, lstm_pars, ticker):
        self.lstm_pars = lstm_pars
        self.stock_bot = StockBot.Stock(ticker, None, test_lstm=True)

    def calculate_fitness(self, items, item_lock):
        global TARGET
        mape = self.stock_bot.begin_lstm(self.lstm_pars)
        item_lock.acquire()
        items[self.lstm_pars].add(mape)
        item_lock.release()

    def calc_test(self):
        return self.stock_bot.begin_lstm(self.lstm_pars)