# =============================================================================
# Multiple_Linear_Regression_CV_missing_data.py
# =============================================================================
import pandas as pd
import numpy as np
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
data = pd.concat([X, offers, bids], axis=1, sort=True)

# Keep only data from 2018 (for simplicity)
data = data.loc[data.index > 2018000000, :]

mae = np.array([])
rmse = np.array([])

for i in range(0, 11):
    if i == 0:
        data_new = data.dropna()
    elif i == 1:
        data_new = data.interpolate(method='linear')
    elif i == 2:
        data_new = data.fillna(method='pad')
    elif i == 3:
        data_new = data.fillna(method='bfill')
    elif i == 4:
        data_new = data.fillna(data.mean())
    elif i == 5:
        data_new = data.interpolate(method='quadratic')
    elif i == 6:
        data_new = data.interpolate(method='cubic')
    elif i == 7:
        data_new = data.interpolate(method='polynomial', order=5)
    elif i == 8:
        data_new = data.interpolate(method='slinear')
    elif i == 9:
        data_new = data.interpolate(method='zero')
    elif i == 10:
        data_new = data.fillna(data.median())

    data_new = data_new.dropna()   # For remaining null values

    # Predict 1h ahead instead of same time
    data_new['Offers'] = data_new['Offers'].shift(-2)
    data_new['Offers'].fillna(method='ffill', inplace=True)

    X = data_new.iloc[:, :-2]
    y = data_new['Offers']

    # Perform Split on TimeSeries data
    tscv = TimeSeriesSplit(n_splits=11)

    # Perform cross validation
    lr = LinearRegression()
    scores = cross_validate(lr, X, y, cv=tscv.split(X),
                            scoring=('neg_mean_squared_error',
                            'neg_mean_absolute_error'))

    mae = np.append(mae, -scores['test_neg_mean_absolute_error'].mean())
    rmse = np.append(rmse,
                     np.sqrt(-scores['test_neg_mean_squared_error'].mean()))

# Plots
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)

size = 150
labels = ['Drop Values', 'Linear Interpolation', 'Forward Propagation',
          'Backward Propagation', 'Mean Value', 'Quadratic Interpolation',
          'Cubic Interpolation', 'Polynomial Interpolation (5th)',
          'Spline Linear Interpolation', 'Zero Value', 'Medial Value']
fig = plt.figure(1, figsize=(7, 3.5), dpi=150)
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
for i in range(0, len(labels)):
    axes.scatter(i, mae[i], marker="o", s=size, label=labels[i])
    if i == 7:
        axes.text(i-0.35, mae[i]-0.08, str(round(mae[i], 2)),
                  fontsize=16)
    else:
        axes.text(i-0.35, mae[i]+0.04, str(round(mae[i], 2)),
                  fontsize=16)
axes.plot(list(range(0, len(labels))),
          min(mae)*np.ones((len(mae), 1)),
          color='black', lw=1, zorder=0)
axes.set_xlabel('Method of handling missing values', fontsize=16)
axes.set_ylabel('Mean Absolute Error', fontsize=16)
axes.set_title('Linear Regression and Missing Data', fontsize=18)
axes.set_xticks(range(0, len(labels)))
axes.set_xticklabels(range(1, len(labels)+1))
axes.legend(loc='upper left', fontsize=13)
axes.grid(True)
axes.autoscale()

fig = plt.figure(2, figsize=(7, 3.5), dpi=150)
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
for i in range(0, len(labels)):
    axes.scatter(i, rmse[i], marker="o", s=size, label=labels[i])
    if i == 7:
        axes.text(i-0.35, rmse[i]-0.28, str(round(rmse[i], 2)),
                  fontsize=16)
    else:
        axes.text(i-0.35, rmse[i]+0.12, str(round(rmse[i], 2)),
                  fontsize=16)
axes.plot(list(range(0, len(labels))),
          min(rmse)*np.ones((len(rmse), 1)),
          color='black', lw=1, zorder=0)
axes.set_xlabel('Method of handling missing values', fontsize=16)
axes.set_ylabel('Root Mean Square Error', fontsize=16)
axes.set_title('Linear Regression and Missing Data', fontsize=18)
axes.set_xticks(range(0, len(labels)))
axes.set_xticklabels(range(1, len(labels)+1))
axes.legend(loc='upper left', fontsize=13)
axes.grid(True)
axes.autoscale()
