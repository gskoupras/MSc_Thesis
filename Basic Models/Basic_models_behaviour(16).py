# =============================================================================
# Basic_models_behaviour.py
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read data from csv files
offers = pd.read_csv('Offers.csv', parse_dates=True, index_col=0)
bids = pd.read_csv('Bids.csv', parse_dates=True, index_col=0)
X = pd.read_csv('Features.csv', parse_dates=True, index_col=0)

# Get rid of extreme values
offers = offers[offers < 2000]
bids = bids[bids > -250]

# Connect all together
data = pd.concat([X, offers], axis=1, sort=True)

# Sort data
data.sort_index(inplace=True)

# Keep only data from 2018 (for simplicity)
data = data.loc[data.index > 2018000000, :]

# Handle missing data
data.fillna(data.median(), inplace=True)

test_size = 0.1
y = data['Offers']

# Predict with Median value
train_size = round((1-test_size)*len(y))
y_train = y.iloc[:train_size]
y_test = y.iloc[train_size:]
median = y_train.median()*np.ones(shape=(len(y_test), 1))
y_pred1 = pd.DataFrame(median, index=y_test.index,
                       columns=['Predictions'])

# Predict with same as previous price (1 hour before)
y_pred2 = y_test.shift(2)
y_pred2.fillna(method='bfill', inplace=True)

# Predict with the mean of 3 previous prices (1 hour before)
y_pred3 = pd.DataFrame(data=y_test)
for i in range(4, len(y_test)):
    y_pred3.iloc[i] = (y_test.iloc[i-4]+y_test.iloc[i-3]+y_test.iloc[i-2])/3

# Plots
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)

fig = plt.figure(1, figsize=(7, 3.5), dpi=150)
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
axes.plot(range(len(y_test)), y_test, zorder=1, lw=2,
          color='red', label='Real Price')
axes.plot(range(len(y_test)), y_pred1, zorder=2, lw=2,
          color='blue', label='Median Value of Dataset')
axes.plot(range(len(y_test)), y_pred2, zorder=1, lw=2,
          color='green', label='Previous Value (1H)')
axes.plot(range(len(y_test)), y_pred3, zorder=0, lw=2,
          color='#edbc4a', label='Mean of Previous 3 (1H)')
axes.set_title('Basic Models Behaviour', fontsize=18)
axes.set_xlabel('Day and SP', fontsize=16)
axes.set_ylabel('Offer Price', fontsize=16)
axes.legend(loc='best', fontsize=16)
axes.grid(True)
axes.autoscale()

fig = plt.figure(2, figsize=(7, 3.5), dpi=150)
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
axes.plot(range(len(y_test)), y_pred1.iloc[:, 0]-y_test, zorder=0,
          color='blue', label='Median Value of Dataset')
axes.plot(range(len(y_test)), y_pred2-y_test, zorder=0,
          color='green', label='Previous Value (1H)')
axes.plot(range(len(y_test)), y_pred3.iloc[:, 0]-y_test, zorder=0,
          color='#edbc4a', label='Mean of Previous 3 (1H)')
axes.set_title('Basic Models Behaviour - Residuals', fontsize=18)
axes.set_xlabel('Day and SP', fontsize=16)
axes.set_ylabel('Offer Price Residual', fontsize=16)
axes.legend(loc='best', fontsize=16)
axes.grid(True)
axes.autoscale()
