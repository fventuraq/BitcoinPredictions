import os
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Input, Model, models, layers
#from tensorflow import set_random_seed
import keras as kr
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from datetime import datetime
#from python-dateutil import dateutil.parser
import dateutil.parser as parser
import random as rn
import json

pd.options.display.max_rows = 100
np.random.seed(1)
tf.random.set_seed(1)
rn.seed(1)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

df = pd.read_csv('BTC_USD_2013-10-01_2020-12-04-CoinDesk.csv')
print(df)

df = df.sort_values(by='Date')
print(df)

df = df.drop(len(df)-1, axis=0)
df = df.drop(len(df)-1, axis=0)
print(df)

df['Date']=[parser.parse(x) for x in list(df['Date'])]
print(df)

df.index = df['Date']
df = df.drop('Date', axis=1)
print(df)

df=df.rename(columns={'Closing Price (USD)':'Price'})
print(df)

df = df[1000:]
df = df[df.index < datetime(2020, 12, 2)]
print(df)

Obs=20
sc = StandardScaler()
df['Price']=sc.fit_transform(df['Price'].values.reshape(-1,1))
print(df)

#Transformamos la columna PRICE a un array
Data = np.asarray(df['Price'])
print(Data)

#transformamos el array a una matriz de 1xN
Data = np.atleast_2d(Data)
print(Data)

#Transformamos la matriz nx1 a una matriz de Nx1
Data = Data.T
print(Data)

#"X” es creado con varios grupos de n (20) observaciones utilizando la data
X = np.atleast_3d(np.array([Data[start:start + Obs] for start in range(0, Data.shape[0] - Obs)]))
print(X)

Y = Data[Obs:]
print(Y)

print (len(X), len(Y))

model = Sequential()
model.add(LSTM(units=200, input_dim=1, return_sequences=True))
model.add(LSTM(units=200, input_dim=1, return_sequences=False))

model.add(Dense(1, input_shape=(1,)))
model.add(Activation('linear'))
model.compile(loss="mape", optimizer="rmsprop")

NN = model.fit(X, Y, epochs=200, batch_size=50, verbose=2, shuffle=False)

pd.DataFrame(NN.history).to_csv('BTC_nominal_loss.csv')

model_params = json.dumps(NN.params)
with open("BTC_Nominal_params.json", "w") as json_file:
  json_file.write(model_params)

model_json = model.to_json()
with open("BTC_Nominal.json", "w") as json_file:
  json_file.write(model_json)

model.save_weights("BTC_Nominal.h5")

Predictions = [model.predict(np.asarray(df['Price'])[i:i+Obs].reshape(1,Obs,1)) for i in range(len(df)-Obs)]
Predictions = [df['Price'].iloc[0]]*Obs + [val[0][0] for val in Predictions]
print(Predictions)

df['Predictions'] = Predictions
df['Price'] = sc.inverse_transform(df['Price'])
df['Predictions'] = sc.inverse_transform(df['Predictions'])
print(df)

#gráfico para comprar las predicciones con la data real

'''---plt.xlabel('Date')
plt.ylabel('Price')
plt.plot(df['Price'], 'b', label='Price')
plt.plot(df['Predictions'], 'r', label='Predictions')
plt.legend(loc='upper left', shadow=True, fontsize='x-large')'''

#Validacion de resuñtados

def ratios(x, y, z):
  predictions = list(df['Predictions'])[:-1]
  price = list(df['Price'])[:-1]
  corrects = []
  for i in range(len(predictions) - y, len(predictions) - z):
    if (predictions[i] * (1.0 + (x / 100.0))) > price[i] > (predictions[i] * (1.0 - (x / 100.0))):
      corrects.append(1.0)
    else:
      corrects.append(0.0)
  return np.average(corrects) * 100

def mape(y_true, y_pred):
  y_true, y_pred = np.array(y_true), np.array(y_pred)
  return np.mean(np.abs((y_true - y_pred)/ y_true))*100

MAPE = [mape(df['Price'][-500:], df['Predictions'][-500:]),
        mape(df['Price'][-1000:-500], df['Predictions'][-1000:-500]),
        mape(df['Price'][-1500:-1000], df['Predictions'][-1500:-1000]),
        mape(df['Price'][-2000:-1500], df['Predictions'][-2000:-1500])
        ]
print(MAPE)

Col1 = [ratios(15, i, i-500) for i in range(500,2001,500)]
Col2 = [ratios(10, i, i-500) for i in range(500,2001,500)]
Col3 = [ratios(5, i, i-500) for i in range(500,2001,500)]

print (pd.DataFrame([MAPE, Col1, Col2, Col3],
                    index=["MAPE", "Ratio 15%", "Ratio 10%", "Ratio 5%"],
                    columns=["0-500","500-1000","1000-1500","1500-2000"]))

df['Vol'] = (df['Price'].pct_change()*100.0).rolling(30).std(ddof=0)
print(df)

plt.xlabel('Date')
plt.ylabel('Volatility of returns')
plt.plot(df.index, df['Vol'], label='Vol')
plt.legend(loc='upper center', shadow=True, fontsize='large')