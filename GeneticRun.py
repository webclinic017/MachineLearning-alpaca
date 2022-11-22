import Individual
from random import randint, shuffle
from threading import Thread, Lock
from statistics import mean

population_size = 5
gen = 1
gens = 10

best = None
lowest_mape = 1000

with open('tickers.txt', 'r') as f:
    tickers = f.readlines()

# lstm_pars = {
        #     'cur_epochs': 5-30,
        #     'cur_batch_size': 20-60,
        #     'window_size': 5-75,
        #     'layer_units': 5-75
        # }


def run_generation():
    global best, gen, lowest_mape, tickers
    item_lock = Lock()
    individuals = []
    threads = []
    items = {}
    shuffle(tickers)

    if gen == 1:
        for i in range(population_size):
            lstm_pars = {
                'cur_epochs': randint(5, 30),
                'cur_batch_size': randint(20, 60),
                'window_size': randint(5, 75),
                'layer_units': randint(5, 75)
            }
            items[lstm_pars] = {}
            for j in range(10):
                individuals.append(Individual.Individual(lstm_pars, ticker=tickers[j]))
    else:
        increment = 10//gen
        for i in range(population_size):
            lstm_pars = {
                'cur_epochs': randint(best['cur_epochs']-increment, best['cur_epochs']+increment),
                'cur_batch_size': randint(best['cur_batch_size']-increment, best['cur_batch_size']+increment),
                'window_size': randint(best['window_size']-increment, best['window_size']+increment),
                'layer_units': randint(best['layer_units']-increment, best['layer_units']+increment)
            }
            items[lstm_pars] = {}
            for j in range(10):
                individuals.append(Individual.Individual(lstm_pars, ticker=tickers[j]))

    for individual in individuals:
        threads.append(Thread(target=individual.calculate_fitness, args=(items, item_lock)))

    print('starting threads')
    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    low_mape = None
    for pars, val in items.items():
        mape = mean(val)
        if low_mape is None or mape < low_mape:
            low_mape = mape
            low_pars = pars

    if low_mape < lowest_mape:
        lowest_mape = low_mape
        best = low_pars

    print(f"generation: {gen}/{gens}\nbest: {best}\nmape: {lowest_mape}")

    gen += 1


if __name__ == '__main__':

    while gen <= gens:
        run_generation()
