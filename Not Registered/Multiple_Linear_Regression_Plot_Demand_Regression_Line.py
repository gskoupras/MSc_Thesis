import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.model_selection import train_test_split
from sklearn import metrics
import statsmodels.formula.api as sm

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
X = data.iloc[:, :-2]
# X = data.loc[:, ['TSDF']]
y = data['Offers']
# y = data['Bids']


# Function that calculates the direction accuracy
# Accepts both row and column vectors
def direction_accuracy(real, pred):
    real=pd.DataFrame(real)
    pred=pd.DataFrame(pred, index=real.index)
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
        if (real_diff.iloc[i,0] > 0) and (pred_diff.iloc[i,0] > 0):
            true = true + 1
        elif (real_diff.iloc[i,0] == 0) and (pred_diff.iloc[i,0] == 0):
            true = true + 1
        elif (real_diff.iloc[i,0] < 0) and (pred_diff.iloc[i,0] < 0):
            true = true + 1
        else:
            false = false + 1
    acc = true / (true + false)
    return acc


# Calculate running time
start_time = time.time()

r = 2

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.1,
                                                    shuffle=False)

# Linear Regression
lin = LinearRegression()
lin.fit(X_train, y_train)
y_pred = lin.predict(X_test)

mae = round(metrics.mean_absolute_error(y_test, y_pred), r)
rmse = round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)), r)
mse = round(metrics.mean_squared_error(y_test, y_pred), r)
da = round(direction_accuracy(y_test, y_pred), r)
rt = round(time.time() - start_time, r)

# print(lin.intercept_[0],lin.coef_[0][0],lin.coef_[0][1],lin.coef_[0][2])
print(lin.intercept_, lin.coef_[4])  # 4 for demand

fig = plt.figure(1, figsize=(7, 3.5), dpi=150)
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
axes.scatter(X['TSDF'], y, marker='.', color="blue", label="Real")
axes.scatter(X['TSDF'], lin.predict(X),
             marker='.', color="red", label="Predicted")
axes.plot(X['TSDF'],
          115+lin.intercept_+X['TSDF']*lin.coef_[4], lw=3,
          color="green", label="Fitted Line")
axes.set_xlim(xmin=18000, xmax=51000)
axes.set_ylim(-40, 310)
axes.set_xlabel('Demand')
axes.set_ylabel('Price')
axes.set_title('Demand on Offer Price')
axes.legend(loc='best')
axes.grid(True)
# axes.autoscale()
# fig.savefig('Effect of TimeSeries Cross Validation on '
#            'Multiple Linear Regression Performance',
#            bbox_inches='tight', dpi=800)

fig = plt.figure(2, figsize=(7, 3.5), dpi=150)
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
axes.plot(range(len(y_test)), y_test, color='red', label='Real')
axes.plot(y_pred, color='blue', label='Predicted')
axes.set_title('Multiple Linear Regression (Real vs Predicted values)')
axes.set_xlabel('Day and SP')
axes.set_ylabel('Offer price')
axes.legend()
axes.grid(True)
axes.autoscale()
# fig.savefig('Multiple Linear Regression (Real vs Predicted values),
#            bbox_inches='tight', dpi=800)

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

