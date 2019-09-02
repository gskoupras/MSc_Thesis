# =============================================================================
# Correlation_thresh_eval_CV.py
# =============================================================================
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import cross_validate

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

# Predict 1h ahead instead of same time
data['Offers'] = data['Offers'].shift(-2)
data['Offers'].fillna(method='ffill', inplace=True)


# 1sh Correlation Function: Threshold
def corr_thresh(df, thr=0.1):
    """df: last column is the output"""
    corr = df.corr()
    features = list(corr['Offers'].iloc[:-1][abs(corr['Offers'].iloc[:-1]) < thr].index)
    for j in df.columns[:-1]:
        if j in features:
            df.drop(j, axis=1, inplace=True)
    return df


r = 2
max_thresh = 6
mae = np.array([])
rmse = np.array([])
mse = np.array([])
da = np.array([])
rt = np.array([])
for i in range(0, max_thresh):

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

    # Predict 1h ahead instead of same time
    data['Offers'] = data['Offers'].shift(-2)
    data['Offers'].fillna(method='ffill', inplace=True)

    print(data.shape[1])
    data_loop = corr_thresh(df=data, thr=i/10)
    print(data.shape[1], data_loop.shape[1])
    X = data_loop.iloc[:, :-1]
    y = data_loop['Offers']

    start_time = time.time()

    r = 2

    # Perform Split on TimeSeries data
    tscv = TimeSeriesSplit(n_splits=11)
    
    # Perform cross validation
    lr = LinearRegression()
    scores = cross_validate(lr, X, y, cv=tscv.split(X),
                            scoring=('neg_mean_squared_error',
                            'neg_mean_absolute_error'))

    mae = np.append(mae,
                    round(-scores['test_neg_mean_absolute_error'].mean(), r))
    rmse = np.append(rmse,
                     round(np.sqrt(-scores['test_neg_mean_squared_error'].mean()), r))
    mse = np.append(mse,
                    round(-scores['test_neg_mean_squared_error'].mean(), r))
    rt = np.append(rt, round(time.time() - start_time, r))

# Plots
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)

size = 150
fig = plt.figure(1, figsize=(7, 3.5), dpi=150)
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
axes2 = axes.twinx()
for i in range(0, max_thresh):
    axes.scatter(list(range(0, max_thresh))[i-1], mae[i-1], marker="o",
                 color='red', s=size, zorder=1)
axes.plot(range(0, max_thresh), mae, color='red',
          lw=2, zorder=1, label='Mean Absolute Error')
axes.plot(range(0, max_thresh), min(mae)*np.ones((max_thresh, 1)),
          color='black', lw=1, zorder=0)
axes.text(np.argmin(mae)-0.18, min(mae)+0.3, str(min(mae)), fontsize=16)
for i in range(0, max_thresh):
    axes2.scatter(list(range(0, max_thresh))[i-1], rmse[i-1], marker="o",
                  color='blue', s=size, zorder=1)
axes2.plot(range(0, max_thresh), rmse, color='blue',
           lw=2, zorder=0, label='Root Mean Squared Error')
axes.set_xlabel('Correlation Threshold considered', fontsize=16)
axes.set_ylabel('Mean Absolute Error', fontsize=16)
axes2.set_ylabel('Root Mean Squared Error', fontsize=16)
axes.set_title('Multiple Linear Regression with Correlation '
               'Threshold considered - CV', fontsize=18)
axes.set_xticks(list(range(0, max_thresh)))
axes.set_xticklabels(list(np.around(np.arange(0, max_thresh/10, 0.1), 1)))
fig.legend(bbox_to_anchor=(0.95, 1),
           bbox_transform=axes.transAxes, fontsize=16)
axes.grid(True)
axes.autoscale()

size = 150
fig = plt.figure(2, figsize=(7, 3.5), dpi=150)
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
for i in range(0, max_thresh):
    axes.scatter(list(range(0, max_thresh))[i-1], rt[i-1], marker="o",
                 color='red', s=size, zorder=1)
axes.plot(range(0, max_thresh), rt, color='red', lw=2,
          zorder=0, label='Running Time')
axes.set_xlabel('Correlation Threshold considered', fontsize=16)
axes.set_ylabel('Running Time (s)', fontsize=16)
axes.set_title('Multiple Linear Regression with Correlation '
               'Threshold considered - CV', fontsize=18)
axes.set_xticks(list(range(0, max_thresh)))
axes.set_xticklabels(list(np.around(np.arange(0, max_thresh/10, 0.1), 1)))
fig.legend(bbox_to_anchor=(1, 1),
           bbox_transform=axes.transAxes, fontsize=16)
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
print('Running Time:', rt)
