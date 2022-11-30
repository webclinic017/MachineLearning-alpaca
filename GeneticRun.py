import Individual
from random import randint, shuffle
# from threading import Thread, Lock
from statistics import mean
from multiprocessing import Process, Pool, Lock

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
    shuffle(tickers)
    all_pars = []

    print('creating pars')
    if gen == 1:
        for i in range(population_size):
            lstm_pars = {
                'cur_epochs': randint(5, 30),
                'cur_batch_size': randint(20, 60),
                'window_size': randint(5, 75),
                'layer_units': randint(5, 75)
            }
            all_pars.append(lstm_pars)
    else:
        increment = 10 // gen
        for i in range(population_size):
            lstm_pars = {
                'cur_epochs': randint(best['cur_epochs'] - increment, best['cur_epochs'] + increment),
                'cur_batch_size': randint(best['cur_batch_size'] - increment, best['cur_batch_size'] + increment),
                'window_size': randint(best['window_size'] - increment, best['window_size'] + increment),
                'layer_units': randint(best['layer_units'] - increment, best['layer_units'] + increment)
            }
            all_pars.append(lstm_pars)

    pars = []
    for j in range(10):
        for h in range(population_size):
            pars.append((all_pars[h], tickers[j].strip()))

    print(f"pars: {pars}")
    pool = Pool(processes=10)

    print('starting processes')
    result = pool.starmap_async(run_individual, pars)

    results = result.get()
    print(results)
    stocks = tickers[:10]
    low_mape = None
    for i in range(population_size):
        mapes = [results[h*population_size+i] for h in range(10)]
        average_mape = mean(mapes)
        paramaters = pars[i][0]
        print(f"with pars: {paramaters},\nAverage mape is: {average_mape}")
        if low_mape is None or average_mape < low_mape:
            low_mape = average_mape
            low_pars = paramaters

    if low_mape < lowest_mape:
        lowest_mape = low_mape
        best = low_pars

    print(f"generation: {gen}/{gens}\nbest paramaters: {best}\nmape: {lowest_mape}\nfor stocks {stocks}")

    pool.close()
    gen += 1


def run_individual(lstm_pars, ticker):
    i = Individual.Individual(lstm_pars, ticker)
    return i.calculate_fitness()


if __name__ == '__main__':

    while gen <= gens:
        run_generation()

