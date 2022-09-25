import requests
import numpy as np
import pandas
import csv


def get_new_data():
    alpha_vantage_apikey = 'ZLGDWBU4HK5QHIXT'
    stock_symbol = 'IBM'

    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={stock_symbol}&apikey={alpha_vantage_apikey}&datatype=csv"

    try:
        response = requests.get(url)
    except Exception as e:
        print(e)
        exit(1)

    lines = response.text.splitlines()
    reader = csv.reader(lines)

    f = open('data.csv', 'w')
    csv_writer = csv.writer(f)
    for row in reader:
        csv_writer.writerow(row)

    f.close()


def read_data():
    return pandas.read_csv('data.csv')


# get closing price for each day
data = read_data()[['close']]
# reverses pandas data
data = data.iloc[::-1]

#sets up test sets
test_ratio = 0.2
train_ratio = 1 - test_ratio
train_size = int(train_ratio * len(data))
test_size = int(test_ratio * len(data))
train = data[:train_size]
test = data[train_size:]

