import tensorflow as tf
from tensorflow.python.keras import Sequential, Model, Input
from tensorflow.python.keras.layers import Dense
import pandas
from sklearn.preprocessing import StandardScaler
import numpy as np
from pathlib import Path
from time import time
from Stock_Model import StockModel

csv_data = pandas.read_csv(f"Data/AAPL_data.csv")
flippedData = csv_data.copy().loc[::-1].reset_index(drop=True)
test_ratio = 0.2
train_ratio = 1 - test_ratio
train_size = int(train_ratio * len(csv_data))
test_size = int(test_ratio * len(csv_data))
train = flippedData[:train_size]
test = flippedData[train_size:]

scaler = StandardScaler()
scaled_data = scaler.fit_transform(csv_data[['close']])
scaled_data_train = scaled_data[:train.shape[0]]
window_size = 50
X, y = [], []

for i in range(window_size, len(scaled_data_train)):
    X.append(scaled_data_train[i - window_size:i])
    y.append(scaled_data_train[i])

x_train, y_train = np.array(X), np.array(y)
n_features = x_train.shape[1]

# model = Sequential()
# model.add(Dense(1024, input_shape=(n_features, 1)))
# model.add(Dense(2048))
# model.add(Dense(1024))
# model.add(Dense(512))
# model.add(Dense(256))
# model.add(Dense(128))
# model.add(Dense(64))
# model.add(Dense(32))
# model.add(Dense(16))
# model.add(Dense(8))
# model.add(Dense(4))
# model.add(Dense(2))
# model.add(Dense(1))
#
#
# # x_in = Input(shape=(8,))
# # x = Dense(100)(x_in)
# # model_functional = Model()
#
# model.compile(loss='mean_squared_error', optimizer='adam')
#
# model.fit(x_train, y_train, epochs=100, batch_size=20, verbose=0)
#
# savable = StockModel('Models/first_model', model=model)
# savable.save_model()
# print('saved')

loadable = StockModel('Models/first_model')
model = loadable.model
print(model)

raw = flippedData['close'][len(flippedData) - len(test) - window_size:].values
raw = raw.reshape(-1, 1)
raw = scaler.transform(raw)

print(raw)

x_test = []

for i in range(window_size, raw.shape[0]):
    x_test.append(raw[i - window_size:i, 0])

x_test = np.array(x_test)
print(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


predicted_price_ = model.predict(x_test)
predicted_price = scaler.inverse_transform(predicted_price_)

y_true, y_pred = np.array(test['close']), np.array(predicted_price)
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

print(mape)