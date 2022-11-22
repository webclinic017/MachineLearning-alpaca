import Individual
import time
lstm_pars = {
                'cur_epochs': 20,
                'cur_batch_size': 50,
                'window_size': 50,
                'layer_units': 50
            }

start = time.time()
ind = Individual.Individual(lstm_pars)
mid = time.time()
print(str(mid-start) + " made individual")
mape = ind.calc_test()
end = time.time()
print(str(end-start) + " " + str(mape))

