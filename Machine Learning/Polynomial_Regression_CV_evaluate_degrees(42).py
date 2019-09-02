# =============================================================================
# Polynomial_Regression_CV_evaluate_degrees.py
# =============================================================================
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import PolynomialFeatures

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

# ********************* Feature Selection *********************

if 'Bids' in data.columns:
    data.drop('Bids', axis=1, inplace=True)


# Based on Correlation threshold
def corr_thresh(df, thr=0.1):
    """df: last column is the output"""
    corr = df.corr()
    features = list(corr['Offers'].iloc[:-1][abs(corr['Offers'].iloc[:-1]) < thr].index)
    for j in df.columns[:-1]:
        if j in features:
            df.drop(j, axis=1, inplace=True)
    return df


data = corr_thresh(df=data, thr=0.5)


## Based on number of correlated features
#def corr_num_of_feat(df, num=10):
#    """df: last column is the output"""
#    corr = df.corr()
#    features = list(abs(corr['Offers'].iloc[:-1]).sort_values(ascending=False).index)
#    features = features[num:]
#    for i in df.columns[:-1]:
#        if i in features:
#            df.drop(i, axis=1, inplace=True)
#    return df
#
#
#data = corr_num_of_feat(df=data, num=3)

## Based on BE on MAE
#data = data.loc[:, ['Ren_R', 'DRM', 'LOLP', 'Gr1', 'Gr2', 'Prev', 'SMA20',
#                    'EMA10', 'RSI', 'BB', 'Spk', 'Med', 'Offers']]

## Based on BE on MAE (CV)
#data = data.loc[:, ['Ren_R', 'TSDF', 'Gr2', 'SMA20',
#                    'BB', 'BBw', 'Med', 'Offers']]

# Divide features and output
X = data.iloc[:, :-1]
y = data['Offers']

# *************************************************************

# Perform Split on TimeSeries data
tscv = TimeSeriesSplit(n_splits=11)

r = 2
deg = list(range(1, 4))
mae = np.array([])
rmse = np.array([])
mse = np.array([])
rt = np.array([])

for i in deg:
    # Prepare the poly regressor
    poly_reg = PolynomialFeatures(degree=i)
    X_poly = poly_reg.fit_transform(X)
#    X_norm = (X-X_poly.mean())/X_poly.std()

    # Calculate running time
    start_time = time.time()

    # Make the predictor FOR THE TEST SET
    lr = LinearRegression()
    scores = cross_validate(lr, X_poly, y, cv=tscv.split(X),
                            scoring=('neg_mean_squared_error',
                            'neg_mean_absolute_error'))

    mae = np.append(mae, round(-scores['test_neg_mean_absolute_error'].mean(),
                               r))
    rmse = np.append(rmse,
                     round(np.sqrt(-scores['test_neg_mean_squared_error'].mean()), r))
    mse = np.append(mse,
                    round(-scores['test_neg_mean_squared_error'].mean(), r))
    rt = np.append(rt, round(time.time() - start_time, r))

# Plots
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)

#size = 100
#fig = plt.figure(1, figsize=(7, 3.5), dpi=150)
#axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
#axes.plot(
#          mae, lw=1.5, color='red', label='Mean Absolute Error')
#axes.scatter(list(range(0, len(mae))),
#             mae, marker="o", color='blue',
#             s=size, label='Mean Absolute Error (points)')
#axes.scatter(mae.argmin(),
#             min(mae), marker="o", color='green',
#             s=size, label='Best')
#axes.annotate('Best Degree = {}'.format(mae.argmin()+1),
#              xy=(mae.argmin(), 1.001*min(mae)), xycoords='data',
#              xytext=(mae.argmin(), 1.01*min(mae)), textcoords='data',
#              arrowprops=dict(arrowstyle="->", facecolor='black'),
#              horizontalalignment='center', verticalalignment='top')
#axes.set_xlabel('Degree', fontsize=16)
#axes.set_ylabel('Mean Absolute Error', fontsize=16)
#axes.set_title('Effect of degree on Polynomial Regression Performance',
#               fontsize=18)
#axes.legend(loc='best', fontsize=16)
#axes.set_xticks(list(range(0, len(mae), 2)))
#axes.set_xticklabels(list(range(1, len(mae)+1, 2)))
#axes.grid(True)
#axes.autoscale()
#
#fig = plt.figure(2, figsize=(7, 3.5), dpi=150)
#axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
#axes.plot(
#          rmse, lw=1.5, color='red', label='Root Mean Squared Error')
#axes.scatter(list(range(0, len(rmse))),
#             rmse, marker="o", color='blue',
#             s=size, label='Root Mean Squared Error (points)')
#axes.scatter(rmse.argmin(),
#             min(rmse), marker="o", color='green',
#             s=size, label='Best')
#axes.annotate('Best Degree = {}'.format(rmse.argmin()+1),
#              xy=(rmse.argmin(), 1.0005*min(rmse)), xycoords='data',
#              xytext=(0.9*rmse.argmin(), 1.004*min(rmse)), textcoords='data',
#              arrowprops=dict(arrowstyle="->", facecolor='black'),
#              horizontalalignment='center', verticalalignment='top')
#axes.set_xlabel('Degree', fontsize=16)
#axes.set_ylabel('Root Mean Squared Error', fontsize=16)
#axes.set_title('Effect of degree on Polynomial Regression Performance',
#               fontsize=16)
#axes.legend(loc='best', fontsize=16)
#axes.set_xticks(list(range(0, len(rmse), 2)))
#axes.set_xticklabels(list(range(1, len(rmse)+1, 2)))
#axes.grid(True)
#axes.autoscale()

size = 150
fig = plt.figure(3, figsize=(7, 3.5), dpi=150)
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
axes2 = axes.twinx()
for i in range(0, len(mae)):
    axes.scatter(list(range(0, len(mae)))[i-1], mae[i-1], marker="o",
                 color='red', s=size, zorder=1)
axes.plot(range(0, len(mae)), mae, color='red',
          lw=2, zorder=1, label='Mean Absolute Error')
axes.plot(range(0, len(mae)), min(mae)*np.ones((len(mae), 1)),
          color='black', lw=1, zorder=0)
axes.text(np.argmin(mae)-0.07, min(mae)+7,
          str(round(min(mae), r)), fontsize=16)
for i in range(0, len(rmse)):
    axes2.scatter(list(range(0, len(rmse)))[i-1], rmse[i-1], marker="o",
                  color='blue', s=size, zorder=1)
axes2.plot(range(0, len(rmse)), rmse, color='blue',
           lw=2, zorder=0, label='Root Mean Squared Error')
axes.set_xlabel('Degree', fontsize=16)
axes.set_ylabel('Mean Absolute Error', fontsize=16)
axes2.set_ylabel('Root Mean Squared Error', fontsize=16)
axes.set_title('Effect of degree on Polynomial Regression Performance - CV',
               fontsize=18)
axes.set_xticks(list(range(0, len(mae))))
axes.set_xticklabels(list(range(1, len(mae)+1)))
fig.legend(bbox_to_anchor=(0.5, 1),
           bbox_transform=axes.transAxes, fontsize=16)
axes.grid(True)
axes.autoscale()

size = 150
fig = plt.figure(4, figsize=(7, 3.5), dpi=150)
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
axes2 = axes.twinx()
for i in range(0, len(rt)):
    axes.scatter(list(range(0, len(rt)))[i-1], rt[i-1], marker="o",
                 color='red', s=size, zorder=1)
axes.plot(range(0, len(rt)), rt, color='red', lw=2,
          zorder=0, label='Running Time')
axes.set_xlabel('Degree', fontsize=16)
axes.set_ylabel('Running Time (s)', fontsize=16)
axes.set_title('Effect of degree on Polynomial Regression Performance - CV',
               fontsize=18)
axes.set_xticks(list(range(0, len(rt))))
axes.set_xticklabels(list(range(1, len(rt)+1)))
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
