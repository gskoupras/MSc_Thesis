# =============================================================================
# Multiple_Linear_Regression_Remove_features_based_on_RMSE.py
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

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


# Divide features and output
X = data.iloc[:, :-2]
y = data['Offers']
# y = data['Bids'] #If you want to evaluate Bids

features = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, shuffle=False)

# Model with all the features
lin = LinearRegression()
lin.fit(X_train, y_train)
y_pred = lin.predict(X_test)

r = 2
mae_best = round(metrics.mean_absolute_error(y_test, y_pred), r)
rmse_best = round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)), r)
mae_evo = np.array([])
rmse_evo = np.array([])

terminal = 1
feat_removed = np.array([])
X_opt = X
start_time = time.time()

while terminal == 1:

    mae_evo = np.append(mae_evo, mae_best)
    rmse_evo = np.append(rmse_evo, rmse_best)
    mae = np.array([])
    rmse = np.array([])

    for i in range(0, len(features)):

        X_test = X_opt.loc[:, X_opt.columns != features[i]]

        X_train, X_test_temp, y_train, y_test = train_test_split(
                         X_test, y, test_size=0.1, shuffle=False)

        lr = LinearRegression()
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_test_temp)

        mae = np.append(mae, round(metrics.mean_absolute_error(y_test,
                                                               y_pred), r))
        rmse = np.append(rmse,
                         round(np.sqrt(metrics.mean_squared_error(y_test,
                                                                  y_pred)), r))

    mae = dict(zip(features, mae))
    rmse = dict(zip(features, rmse))
    print('')
    print(mae)
    print('')
    print('Minimum of MAE:')
    print(min(mae, key=lambda k: mae[k]))
    print(min(mae.values()))
    print('')
    print(rmse)
    print('')
    print('Minimum of RMSE:')
    print(min(rmse, key=lambda k: rmse[k]))
    print(min(rmse.values()))
    print('')

    if rmse_best > min(rmse.values()):
        print('Best model has not been found yet!')
        print('')
        mae_best = min(mae.values())
        rmse_best = min(rmse.values())
        weak_feat = str(min(rmse, key=lambda k: rmse[k]))
        print('Worst feature: ', weak_feat)
        features.remove(weak_feat)
        feat_removed = np.append(feat_removed, weak_feat)
        print('New list of features:')
        print(features)
        X_opt = X_opt.loc[:, features]
        if len(features) == 1:
            print('The model is down to 1 feature!')
            print('')
            mae_evo = np.append(mae_evo, mae_best)
            rmse_evo = np.append(rmse_evo, rmse_best)
            terminal = 0
    else:
        terminal = 0

print('   *****************  Best model has been found!  *****************')
print('')
print('Best Mean Absolute Error:')
print(mae_best)
print('Best Root Mean Squared Error:')
print(min(rmse_evo))
print('')
print('How MAE has evolved:')
print(mae_evo)
print('')
print('List of best features:')
print(features)
print('')
print('List of removed features (in order of removal):')
print(feat_removed)
print('')
print('Running Time:', round(time.time() - start_time, r))
print('')
print('Direction Accuracy:', round(direction_accuracy(y_test, y_pred), r))

# Plots
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)

size = 150
fig = plt.figure(3, figsize=(7, 3.5), dpi=150)
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
axes2 = axes.twinx()
for i in range(0, len(mae_evo)):
    axes.scatter(list(range(0, len(mae_evo)))[i-1], mae_evo[i-1], marker="o",
                 color='red', s=size, zorder=1)
axes.plot(range(0, len(mae_evo)), mae_evo, color='red',
          lw=2, zorder=1, label='Mean Absolute Error')
axes.plot(range(0, len(mae_evo)), min(mae_evo)*np.ones((len(mae_evo), 1)),
          color='black', lw=1, zorder=0)
axes.annotate(str(min(mae_evo)),
              xy=(np.argmin(mae_evo), min(mae_evo)+0.1), xycoords='data',
              xytext=(np.argmin(mae_evo), min(mae_evo)+0.5), textcoords='data',
              arrowprops=dict(arrowstyle="<-", facecolor='black'),
              horizontalalignment='center', verticalalignment='top',
              fontsize=16)
for i in range(0, len(mae_evo)):
    axes2.scatter(list(range(0, len(mae_evo)))[i-1], rmse_evo[i-1], marker="o",
                  color='blue', s=size, zorder=1)
axes2.plot(range(0, len(mae_evo)), rmse_evo, color='blue',
           lw=2, zorder=0, label='Root Mean Squared Error')
axes2.annotate(str(min(rmse_evo)),
               xy=(np.argmin(rmse_evo)*0.99, min(rmse_evo)*1.0005),
               xycoords='data',
               xytext=(np.argmin(rmse_evo)-0.8, min(rmse_evo)+0.09),
               textcoords='data',
               arrowprops=dict(arrowstyle="<-", facecolor='black'),
               horizontalalignment='center', verticalalignment='top',
               fontsize=16)
axes.set_xlabel('Number of features removed', fontsize=16)
axes.set_ylabel('Mean Absolute Error', fontsize=16)
axes2.set_ylabel('Root Mean Squared Error', fontsize=16)
axes.set_title('How metrics evolve '
               'when removing features based on RMSE', fontsize=18)
axes.set_xticks(list(range(0, len(mae_evo))))
axes.set_xticklabels(list(range(0, len(mae_evo))))
fig.legend(bbox_to_anchor=(0.85, 0.95),
           bbox_transform=axes.transAxes, fontsize=16)
axes.grid(True)
axes.autoscale()
