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
    global best, gen
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
        increment = 10//gen
        for i in range(population_size):
            lstm_pars = {
                'cur_epochs': randint(best['cur_epochs']-increment, best['cur_epochs']+increment),
                'cur_batch_size': randint(best['cur_batch_size']-increment, best['cur_batch_size']+increment),
                'window_size': randint(best['window_size']-increment, best['window_size']+increment),
                'layer_units': randint(best['layer_units']-increment, best['layer_units']+increment)
            }
            individuals.append(Individual.Individual(lstm_pars))

    for individual in individuals:
        threads.append(Thread(target=individual.calculate_fitness, args=(items, item_lock)))

    print('starting threads')
    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    mapes = set(items.keys())
    minimum = min(mapes)

    best = items[minimum]

    print(f"generation: {gen}/{gens}\nbest: {best}\nmape: {minimum}")

    gen += 1


if __name__ == '__main__':

    while gen <= gens:
        run_generation()
