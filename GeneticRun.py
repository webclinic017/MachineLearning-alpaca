import time
import Individual
from random import randint, shuffle
from statistics import mean
from multiprocessing import Pool
import os
import pandas
import neptune.new as neptune
from dotenv import load_dotenv
from datetime import datetime


def get_data(ticker: str, AVT, dataset_size: int = 400):
    """
    gets stock csv data, exits the process if it fails
    :param ticker: str, stock ticker
    :param AVT: AlphaVantage API token
    :param dataset_size: int, # of days of data to keep
    :return: DataFrame of csv data
    """
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={ticker}&apikey={AVT}&datatype=csv&outputsize=full"
    try:
        return pandas.read_csv(url)[:dataset_size]
    except Exception as e:
        print(f"error in getting data for ticker: {ticker}, url: {url}, error: {e}")
        exit(1)


def get_data_to_file(ticker: str, AVT, dataset_size: int = 400):
    """
    gets stock csv data to file and returns it, exits the process if it fails
    :param ticker: str, stock ticker
    :param AVT: AlphaVantage API token
    :param dataset_size: int, # of days of data to keep
    :return: DataFrame of csv data
    """
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={ticker}&apikey={AVT}&datatype=csv&outputsize=full"

    try:
        df = pandas.read_csv(url)[:dataset_size]
    except Exception as e:
        print(f"error in getting data for ticker: {ticker}, url: {url}, error: {e}")
        exit(1)

    df.to_csv(f"Data/{ticker}_data.csv")

    return df


def run_generation(log_neptune=True):
    start = time.time()
    global best, gen, lowest_mape, tickers
    shuffle(tickers)
    all_pars = []

    print('creating pars')
    if gen == 1:
        for i in range(population_size):
            lstm_pars = {
                'cur_epochs': randint(5, 30),
                'cur_batch_size': randint(20, 60),
                'window_size': randint(20, 100),
                'layer_units': randint(5, 75)
            }
            all_pars.append(lstm_pars)
    else:
        increment = 10 // gen
        all_pars.append(best)
        for i in range(population_size - 1):
            lstm_pars = {
                'cur_epochs': randint(best['cur_epochs'] - increment, best['cur_epochs'] + increment),
                'cur_batch_size': randint(best['cur_batch_size'] - increment, best['cur_batch_size'] + increment),
                'window_size': randint(best['window_size'] - increment, best['window_size'] + increment),
                'layer_units': randint(best['layer_units'] - increment, best['layer_units'] + increment)
            }
            all_pars.append(lstm_pars)

    pars = []
    for j in range(25):
        for h in range(population_size):
            pars.append((all_pars[h], tickers[j].strip()))

    if log_neptune:
        run["LSTM_Pars"].log(pars)

    pool = Pool(processes=10)
    print('starting processes')
    result = pool.starmap_async(run_individual, pars)

    results = result.get()

    if log_neptune:
        run["Results"].log(results)
    stocks = tickers[:25]
    low_mape = None
    for i in range(population_size):
        mapes = [results[h*population_size+i] for h in range(10)]
        average_mape = mean(mapes)
        paramaters = pars[i][0]
        if log_neptune:
            run["Average_MAPE"].log(f"with pars: {paramaters},\nAverage mape is: {average_mape}")
        if low_mape is None or average_mape < low_mape:
            low_mape = average_mape
            low_pars = paramaters

    lowest_mape = low_mape
    best = low_pars
    pool.close()
    end = time.time() - start
    if log_neptune:
        run[f"Generations/gen{gen}"].log(f"gen: {gen}/{gens} ran in {end} seconds\n best pars: {best}\nmape: {lowest_mape}\n for stocks: {stocks}")
    print(f"generation: {gen}/{gens} ran in {end} seconds\nbest paramaters: {best}\nmape: {lowest_mape}\nfor stocks: {stocks}")

    gen += 1


def run_individual(lstm_pars, ticker):
    i = Individual.Individual(lstm_pars, ticker)
    return i.calculate_fitness()


if __name__ == '__main__':
    load_dotenv()
    NEPTUNE_API_TOKEN = os.getenv('NEPTUNE-API-TOKEN')
    ALPHA_VANTAGE_TOKEN = os.getenv('ALPHA-VANTAGE-API-TOKEN')

    population_size = 5
    gen = 1
    gens = 10

    best = None
    lowest_mape = 1000

    with open('tickers.txt', 'r') as f:
        tickers = f.readlines()
    blacklist = ['AMZN', 'GOOG', 'NVDA', 'TSLA', 'NKE']
    tickers = [ticker.strip() for ticker in tickers if ticker.strip() not in blacklist]

    dateTimeObj = datetime.now()
    custom_id = 'EXP-' + dateTimeObj.strftime("%d-%b-%Y-(%H:%M:%S)")

    run = neptune.init(
        project="elitheknight/GeneticStock",
        api_token=NEPTUNE_API_TOKEN,
        custom_run_id=custom_id
    )

    last_get = pandas.read_csv('Data/AAPL_data.csv')['timestamp'].iloc[0]
    current_get = get_data('AAPL', ALPHA_VANTAGE_TOKEN)['timestamp'].iloc[0]
    if last_get != current_get:
        print('getting new stock data')
        for ticker in tickers:
            get_data_to_file(ticker, ALPHA_VANTAGE_TOKEN)
            time.sleep(16)
        print('done getting stock data')

    while gen <= gens:
        run_generation()
