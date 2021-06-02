# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 11:29:50 2021

@author: gs9356
"""

#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#dataset import
dataset = pd.read_csv("Credit_Card_Applications.csv")
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values



#feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
X = sc.fit_transform(X)


#training the SOM
from minisom import MiniSom
som = MiniSom(10, 10, 15,sigma=1.0, learning_rate=0.5)
som.random_weights_init(X)
som.train_random(data=X, num_iteration=100)



#visualizing the results
from pylab import bone, pcolor,colorbar, plot, show   
bone()
pcolor(som.distance_map().T)
colorbar()
markers=['o', 's']
colors = ['r', 'g']
for i,x in enumerate(X):
    w = som.winner(x)
    plot(w[0]+0.5,w[1]+0.5, markers[y[i]],markeredgecolor=colors[y[i]], markerfacecolor='None', markeredgewidth=2)
    show()
    
    
    
    
