import StockBot

TARGET = 0


class Individual:

    def __init__(self, lstm_pars):
        self.lstm_pars = lstm_pars
        self.stock_bot = StockBot.Stock('FXIAX', None, test_lstm=True)

    def calculate_fitness(self, items, item_lock):
        global TARGET
        mape = self.stock_bot.begin_lstm(self.lstm_pars)
        item_lock.acquire()
        items[mape] = self.lstm_pars
        item_lock.release()