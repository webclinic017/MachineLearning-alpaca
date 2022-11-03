import Individual
from random import randint
from threading import Thread, Lock

population_size = 15
gen = 0
gens = 10

best = None

# lstm_pars = {
        #     'cur_epochs': 5-30,
        #     'cur_batch_size': 20-60,
        #     'window_size': 5-75,
        #     'layer_units': 5-75
        # }


def run_generation():
    global best
    item_lock = Lock()
    individuals = []
    threads = []
    items = {}

    if gen == 0:
        for i in range(population_size):
            lstm_pars = {
                'cur_epochs': randint(5, 30),
                'cur_batch_size': randint(20, 60),
                'window_size': randint(5, 75),
                'layer_units': randint(5, 75)
            }
            individuals.append(Individual.Individual(lstm_pars))
    else:
        for i in range(population_size):
            lstm_pars = {
                'cur_epochs': randint(best['cur_epochs']-5, best['cur_epochs']+5),
                'cur_batch_size': randint(best['cur_batch_size']-5, best['cur_batch_size']+5),
                'window_size': randint(best['window_size']-5, best['window_size']+5),
                'layer_units': randint(best['layer_units']-5, best['layer_units']+5)
            }
            individuals.append(Individual.Individual(lstm_pars))

    for individual in individuals:
        threads.append(Thread(target=individual.calculate_fitness(items, item_lock)))

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    mapes = set(items.keys())
    min = min(mapes)

    best = items[min]
