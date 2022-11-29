import StockBot

TARGET = 0


class Individual:

    def __init__(self, lstm_pars, items, item_lock, ticker):
        self.item_lock = item_lock
        self.items = items
        self.lstm_pars = lstm_pars
        self.stock_bot = StockBot.Stock(ticker, None, test_lstm=True)

    def calculate_fitness(self):
        global TARGET
        mape = self.stock_bot.begin_lstm(self.lstm_pars)
        self.item_lock.acquire()
        self.items.add(mape)
        self.item_lock.release()

    def calc_test(self):
        return self.stock_bot.begin_lstm(self.lstm_pars)