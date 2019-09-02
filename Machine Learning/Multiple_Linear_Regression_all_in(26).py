# =============================================================================
# Multiple_Linear_Regression_all_in.py
# =============================================================================
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


# Function that calculates the direction accuracy
# Accepts both row and column vectors
def direction_accuracy(real, pred):
    real = pd.DataFrame(real)
    pred = pd.DataFrame(pred, index=real.index)
    if real.shape[0] == 1:
        real = real.transpose()
    if pred.shape[0] == 1:
        pred = pred.transpose()
    real_diff = real.shift(-2)-real
    real_diff = real_diff.shift(2)
    pred_diff = pred.shift(-2)-real.values
    pred_diff = pred_diff.shift(2)
    real_diff.fillna(real_diff.median(), inplace=True)
    pred_diff.fillna(pred_diff.median(), inplace=True)
    true = 0
    false = 0
    for i in range(0, len(real_diff)):
        if (real_diff.iloc[i, 0] > 0) and (pred_diff.iloc[i, 0] > 0):
            true = true + 1
        elif (real_diff.iloc[i, 0] == 0) and (pred_diff.iloc[i, 0] == 0):
            true = true + 1
        elif (real_diff.iloc[i, 0] < 0) and (pred_diff.iloc[i, 0] < 0):
            true = true + 1
        else:
            false = false + 1
    acc = true / (true + false)
    return acc


# Calculate running time
start_time = time.time()

r = 2

X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, shuffle=False)

lin = LinearRegression()
lin.fit(X_train, y_train)
y_pred = lin.predict(X_test)

mae = round(metrics.mean_absolute_error(y_test, y_pred), r)
rmse = round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)), r)
mse = round(metrics.mean_squared_error(y_test, y_pred), r)
da = round(direction_accuracy(y_test, y_pred), r)
rt = round(time.time() - start_time, r)

#fig = plt.figure(1, figsize=(7, 3.5), dpi=150)
#axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
#axes.scatter(X_test['APXP'], y_test, marker="o", s=2, color="blue",
#             label='Real')
#axes.scatter(X_test['APXP'], y_pred, marker="o", s=2, color="red",
#             label='Predicted')
#axes.set_xlabel('Electricity Price')
#axes.set_ylabel('Offer Price')
#axes.set_title('Multiple Linear Regression (Real vs Predicted values)')
#axes.legend(loc='best')
#axes.grid(True)
#axes.autoscale()
## fig.savefig('Multiple Linear Regression (Real vs Predicted values)',
##             bbox_inches='tight', dpi=800)

# Plots
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)

fig = plt.figure(2, figsize=(7, 3.5), dpi=150)
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
axes.plot(range(len(y_test[700:900])), y_test[700:900], zorder=1, lw=3,
          color='red', label='Real Price')
axes.plot(y_pred[700:900], zorder=2, lw=3,
          color='blue', label='LR Predicted Price')
axes.plot(y_pred[700:900]-y_test[700:900].values, zorder=0, lw=3,
          color='green', label='Residual Error')
axes.set_title('Multiple Linear Regression, "All-in" Approach',
               fontsize=18)
axes.set_xlabel('Day and SP', fontsize=16)
axes.set_ylabel('Offer price and Residual Error', fontsize=16)
axes.legend(loc='best', fontsize=16)
axes.grid(True)
axes.autoscale()

print('')
print("   ******** Evaluation Metrics ********    ")
print("Mean Absolute Error:")
print(mae)
print("Mean Squared Error:")
print(mse)
print("Root Mean Squared Error:")
print(rmse)
print('Direction Accuracy:', da)
print('Running time:', rt)
