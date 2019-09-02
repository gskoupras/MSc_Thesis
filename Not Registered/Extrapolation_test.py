import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import time

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

plt.rc('xtick', labelsize=30)
plt.rc('ytick', labelsize=30)
x = [1, 2]
y = [100, 110]
z = np.polyfit(x, y, 1)
f = np.poly1d(z)
x_all = list(np.arange(0.5, 2.5, 0.1))
plt.plot(x_all, f(x_all), 'g', zorder=0, label='Linear Equation', lw=15)
#for i in np.arange(0.5, 3.5, 0.1):
#    plt.plot(i, f(i), 'go', zorder=0)
plt.scatter(x, y, color='red', s=1500, label='Price Points', zorder=1)
plt.legend(loc='best', fontsize=35)
plt.grid(True)
plt.show()
print(z)
# print(f(5))

plt.rc('xtick', labelsize=30)
plt.rc('ytick', labelsize=30)
x = [1, 2, 3]
y = [100, 110, 110]
z = np.polyfit(x, y, 2)
f = np.poly1d(z)
x_all = list(np.arange(0.5, 3.5, 0.1))
plt.plot(x_all, f(x_all), 'g', zorder=0, label='Quadratic Equation', lw=15)
#for i in np.arange(0.5, 3.5, 0.1):
#    plt.plot(i, f(i), 'go', zorder=0)
plt.scatter(x, y, color='red', s=1500, label='Price Points', zorder=1)
plt.legend(loc='best', fontsize=35)
plt.grid(True)
plt.show()
print(z)
# print(f(5))

mae = np.array([])
rmse = np.array([])
mse = np.array([])
da = np.array([])
rt = np.array([])
r=2
    
# 1st degree
y_pred = pd.DataFrame(data=data['Offers'])
x = [1, 2]
start_time = time.time()
for i in range(3, len(data['Offers'])):
    y = [data['Offers'].iloc[i-3],
         data['Offers'].iloc[i-2]]
    z = np.polyfit(x, y, 1)
    f = np.poly1d(z)
    y_pred.iloc[i] = f(4)
mae = np.append(mae, round(metrics.mean_absolute_error(data['Offers'], y_pred), r))
rmse = np.append(rmse, round(np.sqrt(metrics.mean_squared_error(data['Offers'], y_pred)), r))
mse = np.append(mse, round(metrics.mean_squared_error(data['Offers'], y_pred), r))
da = np.append(da, round(direction_accuracy(data['Offers'], y_pred), r))
rt = np.append(rt, round(time.time() - start_time, r))

# 2nd degree
y_pred = pd.DataFrame(data=data['Offers'])
x = [1, 2, 3]
for i in range(4, len(data['Offers'])):
    y = [data['Offers'].iloc[i-4],
         data['Offers'].iloc[i-3],
         data['Offers'].iloc[i-2]]
    z = np.polyfit(x, y, 2)
    f = np.poly1d(z)
    y_pred.iloc[i] = f(5)
mae = np.append(mae, round(metrics.mean_absolute_error(data['Offers'], y_pred), r))
rmse = np.append(rmse, round(np.sqrt(metrics.mean_squared_error(data['Offers'], y_pred)), r))
mse = np.append(mse, round(metrics.mean_squared_error(data['Offers'], y_pred), r))
da = np.append(da, round(direction_accuracy(data['Offers'], y_pred), r))
rt = np.append(rt, round(time.time() - start_time, r))

# 3rd degree
y_pred = pd.DataFrame(data=data['Offers'])
x = [1, 2, 3, 4]
for i in range(5, len(data['Offers'])):
    y = [data['Offers'].iloc[i-5],
         data['Offers'].iloc[i-4],
         data['Offers'].iloc[i-3],
         data['Offers'].iloc[i-2]]
    z = np.polyfit(x, y, 3)
    f = np.poly1d(z)
    y_pred.iloc[i] = f(6)
mae = np.append(mae, round(metrics.mean_absolute_error(data['Offers'], y_pred), r))
rmse = np.append(rmse, round(np.sqrt(metrics.mean_squared_error(data['Offers'], y_pred)), r))
mse = np.append(mse, round(metrics.mean_squared_error(data['Offers'], y_pred), r))
da = np.append(da, round(direction_accuracy(data['Offers'], y_pred), r))
rt = np.append(rt, round(time.time() - start_time, r))

## 9th: Predict with linear extrapolation(1 hour before). Based on:
## https://www.geeksforgeeks.org/program-to-implement-linear-extrapolation/
#def predict_with_linear_extrapolation_1h(y):
#    start_time = time.time()
#    y_pred = pd.DataFrame(data=y)
#    for i in range(3, len(y)):
#        y_pred.iloc[i] = y.iloc[i-3]+3*(y.iloc[i-2]-y.iloc[i-3])
#    r = 2
#    mae = round(metrics.mean_absolute_error(y, y_pred), r)
#    rmse = round(np.sqrt(metrics.mean_squared_error(y, y_pred)), r)
#    mse = round(metrics.mean_squared_error(y, y_pred), r)
#    da = round(direction_accuracy(y, y_pred), r)
#    rt = round(time.time() - start_time, r)
#    return mae, rmse, mse, da, rt

