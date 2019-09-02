# =============================================================================
# Artificial_Neural_Network_CV_grid_search_number_of_neurons.py
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn import metrics
import statsmodels.formula.api as sm
from sklearn.preprocessing import StandardScaler
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor

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
# X = data.loc[:, ['TSDF']]
y = data['Offers']
# y = data['Bids']

X = (X-X.mean())/X.std()

tscv = TimeSeriesSplit(n_splits=3)


def build_regressor(n_neurons, optimizer='Adamax', n_layers=8):
    nn = Sequential()
    nn.add(Dense(units=n_neurons,
                 kernel_initializer='uniform',
                 activation='relu', input_dim=len(X.columns)))
    for i in range(0, n_layers):
        nn.add(Dense(units=n_neurons,
                     kernel_initializer='uniform',
                     activation='relu'))
    nn.add(Dense(units=1,
                 kernel_initializer='uniform',
                 activation='linear'))
    nn.compile(optimizer=optimizer,
               loss='mae',
               metrics=['mse', 'mae'])
    return nn


nn = KerasRegressor(build_fn=build_regressor, batch_size=100, epochs=15)

params = {
    'n_neurons': list(range(10, 16))}

grid = GridSearchCV(nn, params, cv=tscv,
                    scoring=(
                            'neg_mean_squared_error',
                            'neg_mean_absolute_error'),
                    return_train_score=False,
                    refit='neg_mean_absolute_error')

grid.fit(X, y)
results = pd.DataFrame(grid.cv_results_)

print('')
print('Minimum of MAE:',
      min(-grid.cv_results_['mean_test_neg_mean_absolute_error']))
# print(-grid.best_score_)  # Alternative way to do it
print('Minimum of RMSE:',
      np.sqrt(min(-grid.cv_results_['mean_test_neg_mean_squared_error'])))
# print(-grid.best_score_)  # Alternative way to do it
print('')
print('The minimum MAE is given for',
      grid.best_params_['n_neurons'], 'neurons')
print('The minimum RMSE is given for',
      grid.cv_results_['param_n_neurons']
      [grid.cv_results_['rank_test_neg_mean_squared_error'] == 1][0],
      'neurons')
print('')
means = -grid.cv_results_['mean_test_neg_mean_absolute_error']
stds = grid.cv_results_['std_test_neg_mean_absolute_error']
params = grid.cv_results_['param_n_neurons']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
print('')
means = np.sqrt(-grid.cv_results_['mean_test_neg_mean_squared_error'])
stds = np.sqrt(grid.cv_results_['std_test_neg_mean_squared_error'])
params = grid.cv_results_['param_n_neurons']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

size = 150
fig = plt.figure(1, figsize=(7, 3.5), dpi=150)
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
for i in range(0, len(params)):
    axes.scatter(i,
                 -grid.cv_results_['mean_test_neg_mean_absolute_error'][i],
                 marker="o", s=size, label=params[i])
axes.set_xlabel('Number of Neurons')
axes.set_ylabel('Mean Absolute Error')
axes.set_title('Effect of Number of Neurons on '
               'Artificial Neural Network Performance')
axes.set_xticks(list(range(0, len(params))))
axes.set_xticklabels(list(range(params[0], len(params)+params[0])))
# axes.legend(loc='best')
axes.grid(True)
axes.autoscale()
# fig.savefig('Effect of Number of Neurons on '
#             'Artificial Neural Network Performance',
#            bbox_inches='tight', dpi=800)

fig = plt.figure(2, figsize=(7, 3.5), dpi=150)
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
for i in range(0, len(params)):
    axes.scatter(i,
                 np.sqrt(-grid.cv_results_[
                         'mean_test_neg_mean_squared_error'][i]),
                 marker="o", s=size, label=params[i])
axes.set_xlabel('Number of Neurons')
axes.set_ylabel('Root Mean Squared Error')
axes.set_title('Effect of Number of Neurons on '
               'Artificial Neural Network Performance')
axes.set_xticks(list(range(0, len(params))))
axes.set_xticklabels(list(range(params[0], len(params)+params[0])))
# axes.legend(loc='best')
axes.grid(True)
axes.autoscale()
# fig.savefig('Effect of Number of Neurons on '
#             'Artificial Neural Network Performance',
#            bbox_inches='tight', dpi=800)
