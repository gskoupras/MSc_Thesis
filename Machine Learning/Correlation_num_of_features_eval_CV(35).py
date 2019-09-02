# =============================================================================
# Correlation_num_of_features_eval_CV.py
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


# 2nd Correlation Function: Number of Features
def corr_num_of_feat(df, num=10):
    """df: last column is the output"""
    corr = df.corr()
    features = list(abs(corr['Offers'].iloc[:-1]).sort_values(ascending=False).index)
    features = features[num:]
    for i in df.columns[:-1]:
        if i in features:
            df.drop(i, axis=1, inplace=True)
    return df


r = 2
max_feat = 22
mae = np.array([])
rmse = np.array([])
mse = np.array([])
da = np.array([])
rt = np.array([])
for i in range(max_feat, 0, -1):

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

    data_loop = corr_num_of_feat(df=data, num=i)
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
for i in range(1, max_feat+1):
    axes.scatter(list(range(1, max_feat+1))[i-1], mae[i-1], marker="o",
                 color='red', s=size, zorder=1)
axes.plot(range(1, max_feat+1), mae, color='red',
          lw=2, zorder=1, label='Mean Absolute Error')
axes.plot(range(1, max_feat+1), min(mae)*np.ones((max_feat, 1)),
          color='black', lw=1, zorder=0)
axes.annotate(str(min(mae)),
              xy=(np.argmin(mae)+1.1, min(mae)+0.1), xycoords='data',
              xytext=(np.argmin(mae)+1.5, min(mae)+0.5), textcoords='data',
              arrowprops=dict(arrowstyle="<-", facecolor='black'),
              horizontalalignment='center', verticalalignment='top',
              fontsize=16)
for i in range(1, max_feat+1):
    axes2.scatter(list(range(1, max_feat+1))[i-1], rmse[i-1], marker="o",
                  color='blue', s=size, zorder=1)
axes2.plot(range(1, max_feat+1), rmse, color='blue',
           lw=2, zorder=0, label='Root Mean Squared Error')
axes2.annotate(str(min(rmse)),
               xy=(np.argmin(rmse)+0.85, min(rmse)+0.01), xycoords='data',
               xytext=(np.argmin(rmse)-0.2, min(rmse)+0.05), textcoords='data',
               arrowprops=dict(arrowstyle="<-", facecolor='black'),
               horizontalalignment='center', verticalalignment='top',
               fontsize=16)
axes.set_xlabel('Number of best features considered', fontsize=16)
axes.set_ylabel('Mean Absolute Error', fontsize=16)
axes2.set_ylabel('Root Mean Squared Error', fontsize=16)
axes.set_title('Multiple Linear Regression with Correlation - '
               'Number of features considered', fontsize=18)
axes.set_xticks(list(range(1, max_feat+1)))
axes.set_xticklabels(list(range(max_feat, 0, -1)))
fig.legend(bbox_to_anchor=(0.9, 0.9),
           bbox_transform=axes.transAxes, fontsize=16)
axes.grid(True)
axes.autoscale()

size = 150
fig = plt.figure(2, figsize=(7, 3.5), dpi=150)
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
for i in range(1, max_feat+1):
    axes.scatter(list(range(1, max_feat+1))[i-1], rt[i-1], marker="o",
                 color='red', s=size, zorder=1)
axes.plot(range(1, max_feat+1), rt, color='red', lw=2,
          zorder=0, label='Running Time')
axes.set_xlabel('Number of previous observations considered', fontsize=16)
axes.set_ylabel('Running Time (s)', fontsize=16)
axes.set_title('Multiple Linear Regression with Correlation - '
               'Number of features considered', fontsize=18)
axes.set_xticks(range(1, max_feat+1))
axes.set_xticklabels(list(range(max_feat, 0, -1)))
fig.legend(bbox_to_anchor=(1, 0.6),
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
print('Direction Accuracy:', da)
print('Running Time:', rt)
