import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import cross_val_score, cross_validate
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

# Divide features and output
X = data.iloc[:, :-2]
y = data['Offers']
# y = data['Bids'] #If you want to evaluate Bids

features = list(X.columns)

tscv = TimeSeriesSplit(n_splits=11)

# Model with all the features
dtr = DecisionTreeRegressor(
        criterion='mse', splitter="best", random_state=0)
scores = cross_validate(dtr, X, y, cv=tscv.split(X),
                        scoring=('neg_mean_squared_error',
                        'neg_mean_absolute_error'))

mae_best = -scores['test_neg_mean_absolute_error'].mean()
rmse_best = np.sqrt(-scores['test_neg_mean_squared_error'].mean())
mae_evo = np.array([])
rmse_evo = np.array([])

terminal = 1
feat_removed = np.array([])
X_opt = X

while terminal == 1:

    mae_evo = np.append(mae_evo, mae_best)
    rmse_evo = np.append(rmse_evo, rmse_best)
    mae = np.array([])
    rmse = np.array([])

    for i in range(0, len(features)):

        X_test = X_opt.loc[:, X_opt.columns != features[i]]

        dtr = DecisionTreeRegressor(
                                    criterion='mse',
                                    splitter="best", random_state=0)
        scores = cross_validate(dtr, X_test, y, cv=tscv.split(X_test),
                                scoring=('neg_mean_squared_error',
                                'neg_mean_absolute_error'))

        mae = np.append(mae, -scores['test_neg_mean_absolute_error'].mean())
        rmse = np.append(rmse,
                         np.sqrt(-scores[
                                 'test_neg_mean_squared_error'].mean()))

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

    if mae_best > min(mae.values()):
        print('Best model has not been found yet!')
        print('')
        mae_best = min(mae.values())
        rmse_best = min(rmse.values())
        weak_feat = str(min(mae, key=lambda k: mae[k]))
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
print('')
print('How MAE has evolved:')
print(mae_evo)
print('')
print('List of best features:')
print(features)
print('')
print('List of removed features (in order of removal):')
print(feat_removed)

fig = plt.figure(1, figsize=(7, 3.5), dpi=150)
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
axes.plot(list(range(0, len(mae_evo))),
          mae_evo, lw=1.5, color='red', label='Mean Absolute Error')
axes.scatter(list(range(0, len(mae_evo))),
             mae_evo, marker="o", color='blue',
             s=100, label='Mean Absolute Error (points)')
axes.set_xlabel('Number of iterations')
axes.set_ylabel('Metrics')
axes.set_title('How Mean Absolute Error evolves '
               'when removing features based on it')
axes.legend(loc='best')
axes.grid(True)
axes.autoscale()
# fig.savefig('How Mean Absolute Error evolves '
#             'when removing features based on it',
#            bbox_inches='tight', dpi=800)


fig = plt.figure(2, figsize=(7, 3.5), dpi=150)
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
axes.plot(list(range(0, len(rmse_evo))),
          rmse_evo, lw=1.5, color='red', label='Root Mean Squared Error')
axes.scatter(list(range(0, len(rmse_evo))),
             rmse_evo, marker="o", color='blue',
             s=100, label='Root Mean Squared Error (points)')
axes.set_xlabel('Number of iterations')
axes.set_ylabel('Metrics')
axes.set_title('How Root Mean Squared Error evolves '
               'when removing features based on MAE')
axes.legend(loc='best')
axes.grid(True)
axes.autoscale()
# fig.savefig('How Root Mean Squared Error evolves '
#             'when removing features based on MAE',
#            bbox_inches='tight', dpi=800)
