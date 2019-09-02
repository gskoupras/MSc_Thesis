import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Read data from csv files
offers = pd.read_csv('Offers.csv', parse_dates=True, index_col=0)
bids = pd.read_csv('Bids.csv', parse_dates=True, index_col=0)
X = pd.read_csv('Features.csv', parse_dates=True, index_col=0)

# Get rid of extreme values
offers = offers[offers < 2000]
bids = bids[bids > -250]

# Connect all together
data = pd.concat([X, offers, bids], axis=1, sort=True)

# Keep only data from 2018 (for simplicity)
data = data.loc[data.index > 2018000000, :]

# Handle missing data
data.fillna(data.median(), inplace=True)

# Predict 1h ahead instead of same time
data['Offers'] = data['Offers'].shift(-2)
data['Offers'].fillna(method='ffill', inplace=True)

# Divide features and output
# X = data.loc[:, ['Prev']]
X = data.iloc[:, :-2]
y = data['Offers']

r = 2

X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, shuffle=False)

lin = LinearRegression()
lin.fit(X_train, y_train)
y_pred = lin.predict(X_test)

# Function that calculates the direction accuracy
# Accepts both row and column vectors
def direction_accuracy(real, pred):
    if real.shape[0] == 1:
        real = real.reshape(real.shape[1], 1)
    if pred.shape[0] == 1:
        pred = pred.reshape(pred.shape[1], 1)
    real_diff = np.diff(real, axis=0)
    pred_diff = np.diff(pred, axis=0)
    true = 0
    false = 0
    for i in range(0, len(real_diff)):
        if (real_diff[i] > 0) and (pred_diff[i] > 0):
            true = true + 1
        elif (real_diff[i] == 0) and (pred_diff[i] == 0):
            true = true + 1
        elif (real_diff[i] < 0) and (pred_diff[i] < 0):
            true = true + 1
        else:
            false = false + 1
    acc = true / (true + false)
    return acc

da = round(direction_accuracy(y_test, y_pred), r)

print('Direction Accuracy:', da)
