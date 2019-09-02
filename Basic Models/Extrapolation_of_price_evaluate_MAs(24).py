# =============================================================================
# Extrapolation_of_price_evaluate_MAs.py
# =============================================================================
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


def predict_with_extrapolation_1h(y, smas, degree=1):
    """
    smas is a list in the form of a range
    degree is an integer
    """

    r = 2
    mae = np.array([])
    rmse = np.array([])
    mse = np.array([])
    da = np.array([])
    rt = np.array([])
    for i in smas:
        start_time = time.time()
        y_new = y.rolling(window=i).mean()
        y_new.fillna(y_new.median(), inplace=True)
        y_pred = pd.DataFrame(data=y_new)
        x = list(range(1, degree+2))
        for j in range(degree+2, len(y_new)):
            y_c = np.array([])
            for k in range(degree+2, 1, -1):
                y_c = np.append(y_c, y_new.iloc[j-k])
            z = np.polyfit(x, y_c, degree)
            f = np.poly1d(z)
            y_pred.iloc[j] = f(degree+3)
        mae = np.append(mae, round(metrics.mean_absolute_error(y, y_pred), r))
        rmse = np.append(rmse, round(np.sqrt(metrics.mean_squared_error(y, y_pred)), r))
        mse = np.append(mse, round(metrics.mean_squared_error(y, y_pred), r))
        da = np.append(da, round(direction_accuracy(y, y_pred), r))
        rt = np.append(rt, round(time.time() - start_time, r))
    return mae, rmse, mse, da, rt


sma = list(range(1, 81))
deg = 1
mae, rmse, mse, da, rt = predict_with_extrapolation_1h(data['Offers'],
                                                       smas=sma,
                                                       degree=deg)


# Plots
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)

size = 20
fig = plt.figure(1, figsize=(7, 3.5), dpi=150)
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
axes2 = axes.twinx()
for i in range(1, len(sma)+1):
    axes.scatter(sma[i-1], mae[i-1], marker="o",
                 color='red', s=50, zorder=11)
axes.plot(sma, mae, color='red',
          lw=4, zorder=11)
axes.plot(sma, mae, color='red',
          lw=2, zorder=0, label='Mean Absolute Error')
axes.plot(range(1, len(sma)+20),
          min(mae)*np.ones((len(list(range(1, len(sma)+20))), 1)),
          color='black', lw=2, zorder=5)
axes.text(60, min(mae)+0.01, str(min(mae)), fontsize=16, zorder=5)
for i in range(1, len(sma)+1):
    axes2.scatter(sma[i-1], rmse[i-1], marker="o",
                  color='blue', s=size, zorder=10)
axes2.plot(sma, rmse, color='blue',
           lw=2, zorder=10, label='Root Mean Squared Error')
axes2.plot(range(1, len(sma)+20),
           min(rmse)*np.ones((len(list(range(1, len(sma)+20))), 1)),
           color='black', lw=2, zorder=5)
axes2.text(20, min(rmse)+0.02, str(min(rmse)), fontsize=16, zorder=5)
axes.set_xlabel('SMA used', fontsize=16)
axes.set_ylabel('Mean Absolute Error', fontsize=16)
axes2.set_ylabel('Root Mean Squared Error', fontsize=16)
axes.set_title('Linear Method Performance on SMAs', fontsize=18)
axes.set_xticks(range(min(sma), max(sma)+1), 5)
fig.legend(bbox_to_anchor=(1.02, 0.18),
           bbox_transform=axes.transAxes, fontsize=16)
axes.grid(True)
axes.autoscale()
# fig.savefig('Polynomial Methods Performance, MAE and RMSE',
#            bbox_inches='tight', dpi=800)


size = 20
fig = plt.figure(2, figsize=(7, 3.5), dpi=150)
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
axes2 = axes.twinx()
for i in range(1, len(sma)+1):
    axes.scatter(sma[i-1], rt[i-1], marker="o",
                 color='red', s=size, zorder=1)
axes.plot(sma, rt, color='red',
          lw=2, zorder=0, label='Running Time')
for i in range(1, len(sma)+1):
    axes2.scatter(sma[i-1], da[i-1], marker="o",
                  color='blue', s=size, zorder=1)
axes2.plot(sma, da, color='blue',
           lw=2, zorder=2, label='Direction Accuracy')
axes.set_xlabel('SMA used', fontsize=16)
axes.set_ylabel('Running Time (s)', fontsize=16)
axes2.set_ylabel('Direction Accuracy', fontsize=16)
axes.set_title('Linear Method Performance on SMAs', fontsize=18)
axes.set_xticks(range(min(sma), max(sma)+1), 5)
fig.legend(bbox_to_anchor=(0.97, 0.85),
           bbox_transform=axes.transAxes, fontsize=16)
axes.grid(True)
axes.autoscale()
# fig.savefig('Polynomial Methods Performance, RT and DA',
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

