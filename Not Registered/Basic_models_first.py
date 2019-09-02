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

# Handle missing data
# data.interpolate(method='linear', inplace=True)
data.fillna(data.median(), inplace=True)

# Keep only data from 2018 (for simplicity)
data = data.loc[data.index > 2018000000, :]


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


# 1st: Mean value
def predict_with_mean(y, test_size=0.1, cv=True):
    start_time = time.time()
    r = 2
    if cv is True:
        mae = np.array([])
        rmse = np.array([])
        mse = np.array([])
        da = np.array([])
        for i in np.arange(0.1, 1, 0.1):
            train_size = int(round((1-i)*len(y)))
            y_train = y.iloc[:train_size]
            y_test = y.iloc[train_size:]
            avg = y_train.mean()*np.ones(shape=(len(y_test), 1))
            y_pred = pd.DataFrame(avg, index=y_test.index,
                                  columns=['Predictions'])
            mae = np.append(mae, round(metrics.mean_absolute_error(y_test,
                                                                   y_pred), r))
            rmse = np.append(rmse,
                             round(np.sqrt(metrics.mean_squared_error(y_test,
                                                                      y_pred)), r))
            mse = np.append(mse, round(metrics.mean_squared_error(y_test,
                                                                  y_pred), r))
            da = np.append(da, round(direction_accuracy(y_test, y_pred), r))
        mae = round(np.mean(mae), r)
        rmse = round(np.mean(rmse), r)
        mse = round(np.mean(mse), r)
        da = round(np.mean(da), r)
        rt = round(time.time() - start_time, r)
        return mae, rmse, mse, da, rt
    else:
        train_size = round((1-test_size)*len(y))
        y_train = y.iloc[:train_size]
        y_test = y.iloc[train_size:]
        avg = y_train.mean()*np.ones(shape=(len(y_test), 1))
        y_pred = pd.DataFrame(avg, index=y_test.index,
                              columns=['Predictions'])
        mae = round(metrics.mean_absolute_error(y_test, y_pred), r)
        rmse = round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)), r)
        mse = round(metrics.mean_squared_error(y_test, y_pred), r)
        da = round(direction_accuracy(y_test, y_pred), r)
        rt = round(time.time() - start_time, r)
        return mae, rmse, mse, da, rt


# 2st: Median value
def predict_with_median(y, test_size=0.1, cv=True):
    start_time = time.time()
    r = 2
    if cv is True:
        mae = np.array([])
        rmse = np.array([])
        mse = np.array([])
        da = np.array([])
        for i in np.arange(0.1, 1, 0.1):
            train_size = int(round((1-i)*len(y)))
            y_train = y.iloc[:train_size]
            y_test = y.iloc[train_size:]
            median = y_train.median()*np.ones(shape=(len(y_test), 1))
            y_pred = pd.DataFrame(median, index=y_test.index,
                                  columns=['Predictions'])
            mae = np.append(mae, round(metrics.mean_absolute_error(y_test,
                                                                   y_pred), r))
            rmse = np.append(rmse,
                             round(np.sqrt(metrics.mean_squared_error(y_test,
                                                                      y_pred)), r))
            mse = np.append(mse, round(metrics.mean_squared_error(y_test,
                                                                  y_pred), r))
            da = np.append(da, round(direction_accuracy(y_test, y_pred), r))
        mae = round(np.mean(mae), r)
        rmse = round(np.mean(rmse), r)
        mse = round(np.mean(mse), r)
        da = round(np.mean(da), r)
        rt = round(time.time() - start_time, r)
        return mae, rmse, mse, da, rt
    else:
        train_size = round((1-test_size)*len(y))
        y_train = y.iloc[:train_size]
        y_test = y.iloc[train_size:]
        median = y_train.median()*np.ones(shape=(len(y_test), 1))
        y_pred = pd.DataFrame(median, index=y_test.index,
                              columns=['Predictions'])
        mae = round(metrics.mean_absolute_error(y_test, y_pred), r)
        rmse = round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)), r)
        mse = round(metrics.mean_squared_error(y_test, y_pred), r)
        da = round(direction_accuracy(y_test, y_pred), r)
        rt = round(time.time() - start_time, r)
        return mae, rmse, mse, da, rt


# 3rd: Predict with same as previous price (1 hour before)
def predict_with_previous_1h(y):
    start_time = time.time()
    y_pred = y.shift(2)
    y_pred.fillna(method='bfill', inplace=True)
    r = 2
    mae = round(metrics.mean_absolute_error(y, y_pred), r)
    rmse = round(np.sqrt(metrics.mean_squared_error(y, y_pred)), r)
    mse = round(metrics.mean_squared_error(y, y_pred), r)
    da = round(direction_accuracy(y, y_pred), r)
    rt = round(time.time() - start_time, r)
    return mae, rmse, mse, da, rt


# 4th: Predict with same as previous price (2 hour before)
def predict_with_previous_2h(y):
    start_time = time.time()
    y_pred = y.shift(4)
    y_pred.fillna(method='bfill', inplace=True)
    r = 2
    mae = round(metrics.mean_absolute_error(y, y_pred), r)
    rmse = round(np.sqrt(metrics.mean_squared_error(y, y_pred)), r)
    mse = round(metrics.mean_squared_error(y, y_pred), r)
    da = round(direction_accuracy(y, y_pred), r)
    rt = round(time.time() - start_time, r)
    return mae, rmse, mse, da, rt


# 5th: Predict with the mean of 3 previous prices(1 hour before)
def predict_with_mean_of_3_previous_1h(y):
    start_time = time.time()
    y_pred = pd.DataFrame(data=y)
    for i in range(4, len(y)):
        y_pred.iloc[i] = (y.iloc[i-4]+y.iloc[i-3]+y.iloc[i-2])/3
    r = 2
    mae = round(metrics.mean_absolute_error(y, y_pred), r)
    rmse = round(np.sqrt(metrics.mean_squared_error(y, y_pred)), r)
    mse = round(metrics.mean_squared_error(y, y_pred), r)
    da = round(direction_accuracy(y, y_pred), r)
    rt = round(time.time() - start_time, r)
    return mae, rmse, mse, da, rt


# 6th: Predict with the mean of 3 previous prices(2 hours before)
def predict_with_mean_of_3_previous_2h(y):
    start_time = time.time()
    y_pred = pd.DataFrame(data=y)
    for i in range(6, len(y)):
        y_pred.iloc[i] = (y.iloc[i-6]+y.iloc[i-5]+y.iloc[i-4])/3
    r = 2
    mae = round(metrics.mean_absolute_error(y, y_pred), r)
    rmse = round(np.sqrt(metrics.mean_squared_error(y, y_pred)), r)
    mse = round(metrics.mean_squared_error(y, y_pred), r)
    da = round(direction_accuracy(y, y_pred), r)
    rt = round(time.time() - start_time, r)
    return mae, rmse, mse, da, rt


# 7th: Predict with the mean of 3 previous prices(1 hour before)
def predict_with_median_of_3_previous_1h(y):
    start_time = time.time()
    y_pred = pd.DataFrame(data=y)
    for i in range(4, len(y)):
        y_pred.iloc[i] = np.median([y.iloc[i-4], y.iloc[i-3], y.iloc[i-2]])
    r = 2
    mae = round(metrics.mean_absolute_error(y, y_pred), r)
    rmse = round(np.sqrt(metrics.mean_squared_error(y, y_pred)), r)
    mse = round(metrics.mean_squared_error(y, y_pred), r)
    da = round(direction_accuracy(y, y_pred), r)
    rt = round(time.time() - start_time, r)
    return mae, rmse, mse, da, rt


# 8th: Predict with the mean of 3 previous prices(1 hour before)
def predict_with_median_of_3_previous_2h(y):
    start_time = time.time()
    y_pred = pd.DataFrame(data=y)
    for i in range(6, len(y)):
        y_pred.iloc[i] = np.median([y.iloc[i-6], y.iloc[i-5], y.iloc[i-4]])
    r = 2
    mae = round(metrics.mean_absolute_error(y, y_pred), r)
    rmse = round(np.sqrt(metrics.mean_squared_error(y, y_pred)), r)
    mse = round(metrics.mean_squared_error(y, y_pred), r)
    da = round(direction_accuracy(y, y_pred), r)
    rt = round(time.time() - start_time, r)
    return mae, rmse, mse, da, rt


# Gather results
mae = np.array([])
rmse = np.array([])
mse = np.array([])
da = np.array([])
rt = np.array([])

for func in (predict_with_mean, predict_with_median):
    mae_, rmse_, mse_, da_, rt_ = func(data['Offers'],
                                       test_size=0.1, cv=True)
    mae = np.append(mae, mae_)
    rmse = np.append(rmse, rmse_)
    mse = np.append(mse, mse_)
    da = np.append(da, da_)
    rt = np.append(rt, rt_)

for func in (predict_with_previous_1h, predict_with_previous_2h,
             predict_with_mean_of_3_previous_1h,
             predict_with_mean_of_3_previous_2h,
             predict_with_median_of_3_previous_1h,
             predict_with_median_of_3_previous_2h):
    mae_, rmse_, mse_, da_, rt_ = func(data['Offers'])
    mae = np.append(mae, mae_)
    rmse = np.append(rmse, rmse_)
    mse = np.append(mse, mse_)
    da = np.append(da, da_)
    rt = np.append(rt, rt_)

# Plots
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)

size = 150
labels = ['Mean Value of dataset', 'Median Value of dataset',
          'Previous Value (1H)', 'Previous Value (2H)',
          'Mean of Previous 3 (1H)', 'Mean of Previous 3 (2H)',
          'Median of Previous 3 (1H)', 'Median of Previous 3 (2H)']
fig = plt.figure(3, figsize=(7, 3.5), dpi=150)
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
for i in range(0, len(labels)):
    axes.scatter(i, mae[i], marker="o", s=size, label=labels[i])
axes.set_xlabel('Basic Method')
axes.set_ylabel('Mean Absolute Error')
axes.set_title('Basic Methods Performance')
axes.set_xticks(list(range(0, len(labels))))
axes.set_xticklabels(list(range(1, len(labels)+1)))
axes.legend(loc='best')
axes.grid(True)
axes.autoscale()

fig = plt.figure(4, figsize=(7, 3.5), dpi=150)
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
for i in range(0, len(labels)):
    axes.scatter(i, rmse[i], marker="o", s=size, label=labels[i])
axes.set_xlabel('Basic Method')
axes.set_ylabel('Root Mean Squared Error')
axes.set_title('Basic Methods Performance')
axes.set_xticks(list(range(0, len(labels))))
axes.set_xticklabels(list(range(1, len(labels)+1)))
axes.legend(loc='best')
axes.grid(True)
axes.autoscale()

fig = plt.figure(5, figsize=(7, 3.5), dpi=150)
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
for i in range(0, len(labels)):
    axes.scatter(i, rt[i], marker="o", s=size, label=labels[i])
axes.set_xlabel('Basic Method')
axes.set_ylabel('Running Time (s)')
axes.set_title('Basic Methods Performance')
axes.set_xticks(list(range(0, len(labels))))
axes.set_xticklabels(list(range(1, len(labels)+1)))
axes.legend(loc='best')
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
