# =============================================================================
# Support_Vector_Regression_CV_grid_search.py
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit

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


## Based on Correlation threshold
#def corr_thresh(df, thr):
#    """df: last column is the output"""
#    corr = df.corr()
#    features = list(corr['Offers'].iloc[:-1][abs(corr['Offers'].iloc[:-1]) < thr].index)
#    for j in df.columns[:-1]:
#        if j in features:
#            df.drop(j, axis=1, inplace=True)
#    return df
#
#
#data = corr_thresh(df=data, thr=0.3)


# Based on number of correlated features
def corr_num_of_feat(df, num):
    """df: last column is the output"""
    corr = df.corr()
    features = list(abs(corr['Offers'].iloc[:-1]).sort_values(ascending=False).index)
    features = features[num:]
    for i in df.columns[:-1]:
        if i in features:
            df.drop(i, axis=1, inplace=True)
    return df


data = corr_num_of_feat(df=data, num=5)

## Based on BE
#data = data.loc[:, ['Ren_R', 'Rene', 'TSDF', 'NIV',
#                    'Im_Pr', 'In_gen', 'DRM', 'Gr1', 'Ret', 'SMA20',
#                    'EMA10', 'RSI', 'SRSID', 'BB', 'BBw', 'Med', 'Offers']]

## Based on BE on MAE
#data = data.loc[:, ['Ren_R', 'DRM', 'LOLP', 'Gr1', 'Gr2', 'Prev', 'SMA20',
#                    'EMA10', 'RSI', 'BB', 'Spk', 'Med', 'Offers']]

## Based on BE on MAE (CV)
#data = data.loc[:, ['Ren_R', 'TSDF', 'Gr2', 'SMA20',
#                    'BB', 'BBw', 'Med', 'Offers']]

# All-in
X = data.iloc[:, :-1]
y = data['Offers']

# *************************************************************

# Calculate running time
start_time = time.time()

r = 2

params = {
    'C': list(range(1, 11)),
    'epsilon': (0.05, 0.1, 0.2, 0.3, 0.4, 0.5)}

# Perform Split on TimeSeries data
tscv = TimeSeriesSplit(n_splits=11)

# Perform cross validation
svr = SVR(kernel='rbf', verbose=1)
grid = GridSearchCV(svr, params, cv=tscv,
                    scoring=(
                            'neg_mean_squared_error',
                            'neg_mean_absolute_error'),
                    return_train_score=False,
                    refit='neg_mean_absolute_error', n_jobs=-1)
grid.fit(X, y)

results = pd.DataFrame(grid.cv_results_)

rt = round(time.time() - start_time, r)

# Plots
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)


print('Minimum of MAE:',
      min(-grid.cv_results_['mean_test_neg_mean_absolute_error']))
# print(-grid.best_score_)
print('This is given for a C of',
      grid.best_params_['C'], 'and an epsilon of',
      grid.best_params_['epsilon'])

fig = plt.figure(1)
ax = plt.axes(projection='3d')
ax.plot3D(results['param_alpha'], results['param_l1_ratio'],
          -results['mean_test_neg_mean_absolute_error'], '-',
          lw=2, color='red')

print('Mean of MAE:',
      round(min(-results['test_neg_mean_absolute_error'].mean()), r))
print('Mean of RMSE:',
      round(np.sqrt(min(-results['test_neg_mean_squared_error'].mean())), r))
print('Mean of MSE:',
      round(min(-results['test_neg_mean_squared_error'].mean(), r)))
print('Running Time:', rt)
