# =============================================================================
# Support_Vector_Regression_grid_search.py
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn import metrics
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d
from matplotlib import cm

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


mae = np.array([])
rmse = np.array([])
mse = np.array([])
da = np.array([])
rt = np.array([])

params = {
        "C": list(range(10, 470, 50)),
        "epsilon": np.around(list(np.arange(0.01, 0.8, 0.05)), 2)}

for i in params[list(params)[0]]:
    for j in params[list(params)[1]]:
        # Calculate running time
        start_time = time.time()

        r = 2

        X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.1, shuffle=False)

        X_train_norm = (X_train-X_train.mean())/X_train.std()
        X_test_norm = (X_test-X_train.mean())/X_train.std()

        svr = SVR(C=i, kernel='rbf', gamma='scale', epsilon=j, verbose=1)
        svr.fit(X_train_norm, y_train)
        y_pred = svr.predict(X_test_norm)

        mae = np.append(mae,
                        round(metrics.mean_absolute_error(y_test, y_pred), r))
        rmse = np.append(rmse,
                         round(np.sqrt(metrics.mean_squared_error(y_test,
                                                                  y_pred)), r))
        mse = np.append(mse,
                        round(metrics.mean_squared_error(y_test, y_pred), r))
        da = np.append(da, round(direction_accuracy(y_test, y_pred), r))
        rt = np.append(rt, round(time.time() - start_time, r))
        print(i, j)


mae = mae.reshape(len(params[list(params)[0]]), len(params[list(params)[1]]))
mae = pd.DataFrame(data=mae,
                   index=params[list(params)[0]],
                   columns=params[list(params)[1]])

rmse = rmse.reshape(len(params[list(params)[0]]), len(params[list(params)[1]]))
rmse = pd.DataFrame(data=rmse,
                    index=params[list(params)[0]],
                    columns=params[list(params)[1]])

# Plots
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)

param_x, param_y = np.meshgrid(mae.columns, mae.index)

#fig = plt.figure(1)
#ax = plt.axes(projection='3d')
#ax.scatter3D(param_x, param_y, 
#              mae, '-', lw=2, color='red')

fig = plt.figure(2, dpi=100)
ax = plt.axes(projection='3d')
surf = ax.plot_surface(param_x, param_y, mae, rstride=1, cstride=1,
                       cmap=cm.rainbow,
                       linewidth=4, antialiased=False)
fig.colorbar(surf, shrink=1, aspect=10)
ax.set_title('SVR', fontsize=16)
ax.set_xlabel(list(params)[1], fontsize=16, labelpad=20)
ax.set_ylabel(list(params)[0], fontsize=16, labelpad=20)
ax.set_zlabel('MAE', fontsize=16, labelpad=15)
ax.locator_params(axis='z', nbins=5)
ax.tick_params(axis='z', pad=5)

fig = plt.figure(3, dpi=100)
ax = plt.axes(projection='3d')
surf = ax.plot_surface(param_x, param_y, rmse, rstride=1, cstride=1,
                       cmap=cm.rainbow,
                       linewidth=4, antialiased=False)
fig.colorbar(surf, shrink=1, aspect=10)
ax.set_title('SVR', fontsize=16)
ax.set_xlabel(list(params)[1], fontsize=16, labelpad=20)
ax.set_ylabel(list(params)[0], fontsize=16, labelpad=20)
ax.set_zlabel('RMSE', fontsize=16, labelpad=15)
ax.locator_params(axis='z', nbins=5)
ax.tick_params(axis='z', pad=5)

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
print('')
print('Best MAE: ')
print(mae.min(axis=0).min())
print('For', list(params)[0], ':', mae.min(axis=1).idxmin())
print('and ', list(params)[1], ':', mae.min(axis=0).idxmin())
print('Best RMSE: ')
print(rmse.min(axis=0).min())
print('For', list(params)[0], ':', rmse.min(axis=1).idxmin())
print('and ', list(params)[1], ':', rmse.min(axis=0).idxmin())
