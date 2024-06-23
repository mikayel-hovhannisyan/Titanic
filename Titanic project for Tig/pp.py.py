import pandas as pd
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import relu,linear
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.keras.backend.set_floatx('float64')

tf.autograph.set_verbosity(0)

dataframe = pd.read_csv("train.csv")
A_train = np.asarray(dataframe)
Y_train = []
X_train = []

for x in range(len(A_train)):
    Y_train.append(A_train[x][1])
    X_train.append([])
    X_train[x].append(A_train[x][0])
    for y in range(2, len(A_train[x])):
        X_train[x].append(A_train[x][y])


tickets = {
    "1" : [],
}
for x in range(len(X_train)):
    tl = X_train[x][7].split(" ")
    tn = tl[-1]
    for y in tickets:
        tickets[y].append("0")
    if len(tl) > 1:
        if len(tl) > 2:
            del tl[-1]
            tl = [" ".join(tl)]
        if not tl[0] in tickets:
            tickets[tl[0]] = []
            for asd in tickets["1"]:
                tickets[tl[0]].append(0)
        tickets[tl[0]][-1] = tn
        
del tickets["1"]
ticketsArr = []

for x in tickets:

    ticketsArr.append([])
    for y in tickets[x]:
        ticketsArr[-1].append(y)
ticketsArr = np.rot90(ticketsArr)
ticketsArr = ticketsArr[::-1]

X_train = np.array(X_train)
Y_train = np.array(Y_train)

X_train = np.append(X_train, ticketsArr, axis=1)
for row in X_train:
    
    # Name #
    row[2] = 0
    
    # Sex #
    if row[3] == "male":
        row[3] = 1
    elif row[3] == "female":
        row[3] = 0
    else:
        row[3] = 999

    # Age #
    if row[4] == "nan":
        row[4] = 999

    row[5] = 0
    row[6] = 0
    row[7] = 0
    row[8] = 0
    row[9] = 0
    

    
    # Embarked #
    if row[10] == "S":
        row[10] = 0
    elif row[10] == "C":
        row[10] = 1
    elif row[10] == "Q":
        row[10] = 2
    else:
        row[10] = 999


    # Cabin #
    row[9] = 0
   
    # for i in row:
    #     if type(i) is int:
    #         print("")
    #     else:
    #         print(type(i))
    


# COLUMN = 10
# for ROW in X_train:
#     print(ROW[COLUMN])

def eval_mse(y, yhat):

    m = len(y)
    err = 0.0
    for i in range(m):
       err += ((yhat[i]-y[i])**2)/(2*m)

    
    return(err)

def eval_cat_err(y, yhat):

    m = len(y)
    incorrect = 0
    for i in range(m):
        if yhat[i] != y[i]:
            incorrect += 1
    cerr = incorrect / m

    
    return(cerr)

Test_train = []

for x in X_train:
    Test_train.append([])
    for y in x:
        Test_train[-1].append(y)
    # print(Test_train[-1])
    

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

tf.random.set_seed(1234)
model = Sequential(
    [
        Dense(120,activation = "relu"),
        Dense(40,activation = "relu"),
        Dense(6,activation = "linear")

    ], name="Complex"
)
model.compile(
    loss=SparseCategoricalCrossentropy(from_logits=True),
    optimizer = Adam(0.01),
)
    
model.fit(
    Test_train, Y_train,
    epochs=1000
)