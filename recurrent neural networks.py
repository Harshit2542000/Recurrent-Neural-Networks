# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 07:04:24 2020

@author: harshit
"""
#data preprocessing

#part-1 importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#part-2 importing the training set
dataset_train=pd.read_csv("Google_Stock_Price_Train.csv")
training_set=dataset_train.iloc[:,1:2].values
#part-3 feature scaling
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler(feature_range=(0,1))
training_set_scaled=sc.fit_transform(training_set)
#part-3 choosing the number of timesteps and 1 output
X_train=[]
Y_train=[]
for i in range(60,1258):
    X_train.append(training_set_scaled[i-60:i,0])
    Y_train.append(training_set_scaled[i,0])
X_train=np.array(X_train)
Y_train=np.array(Y_train)
#part-4adding a new dimension to the X_train as keras module expects a 3-D array with new dimension as the predictor
X_train=np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))
#Building The RNN
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
#part-1 Initialising The RNN
regressor=Sequential()
#part-2 Adding The First LSTM Layer And The Dropout Regularisation Layer
regressor.add(LSTM(units=50,return_sequences=True,input_shape=(X_train.shape[1],1)))
regressor.add(Dropout(0.20))
#part-3 Adding Another LSTM Layer And The Dropout Regularisation Layer
regressor.add(LSTM(units=50,return_sequences=True))
regressor.add(Dropout(0.20))
#part-4 Adding Another LSTM Layer And The Dropout Regularisation Layer
regressor.add(LSTM(units=50,return_sequences=True))
regressor.add(Dropout(0.20))
#part-5 Adding The LSTM Layer And The Dropout Regularisation Layer
regressor.add(LSTM(units=50))
regressor.add(Dropout(0.20))
#part-6 Adding The Output Layer 
regressor.add(Dense(units=1))
#Compiling The RNN
regressor.compile(optimizer='adam',loss='mean_squared_error')
#training the RNN on the training set
regressor.fit(X_train,Y_train,epochs=100,batch_size=32)
#predicting the test set results and visualizing the predictions
#part-1 getting the real stock price of january 2017 for the test set,so,that later we can visualize the results.
dataset_test=pd.read_csv("Google_Stock_Price_Test.csv")
real_stock_price=dataset_test.iloc[:,1:2].values
#part-2 predicting the stock price of january 2017
dataset_total=pd.concat((dataset_train['Open'],dataset_test['Open']),axis=0)
inputs=dataset_total[len(dataset_total)-len(dataset_test)-60: ].values
inputs=inputs.reshape(-1,1)
inputs=sc.transform(inputs)
X_test=[]
for i in range(60,80):
    X_test.append(inputs[i-60:i,0])
X_test=np.array(X_test)
X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
predicted_stock_price=regressor.predict(X_test)
predicted_stock_price=sc.inverse_transform(predicted_stock_price)
#Visualizing The Prediction Results
plt.plot(real_stock_price,color='red',label='Real Google Stock Price')
plt.plot(predicted_stock_price,color='blue',label='Predicted Google Stock Price')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
#evaluating the RNN
import math
from sklearn.metrics import mean_squared_error
RMSE=math.sqrt(mean_squared_error(real_stock_price,predicted_stock_price))
print("Root Mean Squared Error:- ",RMSE)
print("Absolute/Relative Error:- ",RMSE/800)