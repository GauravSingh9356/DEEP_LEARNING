# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 11:30:42 2021

@author: gs9356
"""

#importing libraries

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt




#importint dataset and dividing into training and testing 

data = pd.read_csv("Google_Stock_Price_Train.csv")
training_set = data[data['Date']<'2019-01-01'].copy()
training_set = training_set.drop(['Date', 'Adj Close', 'High', 'Low', 'Close', 'Volume'], axis = 1)
data_test = data[data['Date']>='2019-01-01'].copy()
data_test = data_test.drop(['Date', 'Adj Close', 'High', 'Low', 'Close', 'Volume'], axis = 1)


#feature scaling

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)



#creating with 60 timestamps and 1 output

X_train = []
y_train = []
for i in range(60, training_set_scaled.shape[0]):
    X_train.append(training_set_scaled[i-60:i])
    y_train.append(training_set_scaled[i,0])
    
X_train,y_train = np.array(X_train),np.array(y_train)






#building the RNN


#importing the keras libraries and packages

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


#initialising the RNN 

regressor = Sequential()

#adding the first LSTM layer and dropout regularization

regressor.add(LSTM(units = 50,activation="relu", return_sequences = True, input_shape=(X_train.shape[1], 1)))
regressor.add(Dropout(0.2))


#adding second LSTM layer with dropout regularization

regressor.add(LSTM(units=60, activation="relu", return_sequences = True))
regressor.add(Dropout(0.2))

#adding the third LSTM layer and dropout regularization

regressor.add(LSTM(units=80, activation="relu", return_sequences = True))
regressor.add(Dropout(0.2))

#adding the fourth LSTM layer and dropout regularization

regressor.add(LSTM(units=120, activation="relu", return_sequences = False))
regressor.add(Dropout(0.2))

#adding the output layer

regressor.add(Dense(units=1))


#compiling the RNN

regressor.compile(optimizer = "adam", loss="mean_squared_error", metrics=[tf.keras.metrics.Accuracy()])


#fitting the RNN to the training set

regressor.fit(X_train, y_train, epochs=100, batch_size=32)

#getting the data for testing


past_60_days = training_set.tail(60)

df = past_60_days.append(data_test, ignore_index = True)


inputs = sc.transform(df)

X_test = []
y_test=[]

for i in range(60, inputs.shape[0]):
    X_test.append(inputs[i-60:i])
    y_test.append(inputs[i, 0])
    
X_test, y_test = np.array(X_test), np.array(y_test)

predicted_stock_price = regressor.predict(X_test)


#Inverse transform to scale it back
scale = 1/8.18605127e-04

predicted_stock_price = predicted_stock_price*scale
real_stock_price = y_test*scale
    
#visualizing the results

plt.figure(figsize=(14,5))
plt.plot(real_stock_price, color="red", label="Real Google Stock Price")
plt.plot(predicted_stock_price, color="blue", label="Predicted Google Stock Price")
plt.title("Google Stock Price Prediction")
plt.xlabel("Time")
plt.ylabel("Google Stock Price")
plt.legend()
plt.show()



#RMSE

import math
from sklearn.metrics import mean_squared_error

rmse = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))

print(rmse)
























