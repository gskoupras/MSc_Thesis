# =============================================================================
# Artificial_Neural_Network_CV_grid_search_all_three_variables.py
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

tscv = TimeSeriesSplit(n_splits=11)


def build_regressor(optimizer, n_neurons, n_layers):
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


nn = KerasRegressor(build_fn=build_regressor, batch_size=20, epochs=100)

params = {
    'optimizer': ['SGD', 'RMSprop', 'Adagrad',
                  'Adadelta', 'Adam', 'Adamax', 'Nadam'],
    'n_neurons': list(range(1, 16)),
    'n_layers': list(range(1, 11))
    }

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
print(-grid.best_score_)  # Alternative way to do it
print('Minimum of RMSE:',
      np.sqrt(min(-grid.cv_results_['mean_test_neg_mean_squared_error'])))
print('')
print('The minimum MAE is given for the',
      grid.best_params_['optimizer'], 'optimizer,',
      grid.best_params_['n_neurons'], 'neurons and',
      grid.best_params_['n_layers'], 'hidden layers')
print('The minimum RMSE is given for the',
      grid.cv_results_['param_optimizer']
      [grid.cv_results_['rank_test_neg_mean_squared_error'] == 1][0],
      'optimizer,', grid.cv_results_['param_n_neurons']
      [grid.cv_results_['rank_test_neg_mean_squared_error'] == 1][0],
      'neurons and', grid.cv_results_['param_n_layers']
      [grid.cv_results_['rank_test_neg_mean_squared_error'] == 1][0],
      'hidden layers')
print('')

## Plot number of neurons and layers for the best optimizer
#fig = plt.figure(1)
#ax = plt.axes(projection='3d')
#ax.plot3D(grid.cv_results_['param_alpha'], grid.cv_results_['param_l1_ratio'],
#          -grid.cv_results_['mean_test_neg_mean_absolute_error'], '-',
#          lw=2, color='red')








