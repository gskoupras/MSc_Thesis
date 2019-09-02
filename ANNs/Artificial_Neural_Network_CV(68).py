# =============================================================================
# Artificial_Neural_Network_CV.py
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import cross_val_score, cross_validate
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


def build_regressor(n_neurons=6, n_layers=1, optimizer='Adagrad'):
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


nn = KerasRegressor(build_fn=build_regressor, batch_size=50, epochs=30)
scores = cross_validate(nn, X, y, cv=tscv,
                        scoring=('neg_mean_squared_error',
                        'neg_mean_absolute_error'))

print('Mean of MAE(test set):', -scores['test_neg_mean_absolute_error'].mean())
print('Standard deviation of MAE:',
      -scores['test_neg_mean_absolute_error'].std())
print('Mean of RMSE(test set):',
      np.sqrt(-scores['test_neg_mean_squared_error'].mean()))
print('Standard deviation of RMSE:',
      -scores['test_neg_mean_squared_error'].std())

fig = plt.figure(1, figsize=(7, 3.5), dpi=150)
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
axes.plot(-scores['test_neg_mean_absolute_error'],
          '-', lw=1.5, color='red', label='Mean Absolute Error')
axes.set_xlabel('Cross validation Split')
axes.set_ylabel('Metrics')
axes.set_title('Effect of TimeSeries Cross Validation on '
               'Artificial Neural Network Performance')
axes.legend(loc='best')
axes.grid(True)
axes.autoscale()

fig = plt.figure(2, figsize=(7, 3.5), dpi=150)
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
axes.plot(np.sqrt(-scores['test_neg_mean_squared_error']),
          '-', lw=1.5, color='blue', label='Root Mean Squared Error')
axes.set_xlabel('Cross validation Split')
axes.set_ylabel('Metrics')
axes.set_title('Effect of TimeSeries Cross Validation on '
               'Artificial Neural Network Performance')
axes.legend(loc='best')
axes.grid(True)
axes.autoscale()
