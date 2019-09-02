import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn import metrics
import statsmodels.formula.api as sm

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
# X = data.iloc[:, :-2]
X = data.loc[:, ['TSDF']]
y = data['Offers']
# y = data['Bids']

# Perform Split on TimeSeries data
tscv = TimeSeriesSplit(n_splits=11)

# params = {
#    'l1_ratio': (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1)}

params = {
    'alpha': list(range(5, 400000, 10000))}

en = ElasticNet()
grid = GridSearchCV(en, params, cv=tscv,
                    scoring=(
                            'neg_mean_squared_error',
                            'neg_mean_absolute_error'),
                    return_train_score=False,
                    refit='neg_mean_absolute_error')
grid.fit(X, y)

print('Minimum of MAE: ',
      min(-grid.cv_results_['mean_test_neg_mean_absolute_error']))
# print(-grid.best_score_)
print('This is given for an alpha of:', grid.best_params_['alpha'])

fig = plt.figure(1, figsize=(7, 3.5), dpi=150)
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
axes.plot(params['alpha'],
          -grid.cv_results_['mean_test_neg_mean_absolute_error'], '-',
          lw=2, color='red', label='Mean Absolute Error')
axes.set_xlabel('Alpha Ratio')
axes.set_ylabel('Mean Absolute Error')
axes.set_title('Effect of Alpha Ratio on '
               'Elastic Net Regression Performance')
axes.legend(loc='best')
axes.grid(True)
axes.autoscale()
# fig.savefig('Effect of Alpha Ratio on '
#             'Elastic Net Regression Performance',
#            bbox_inches='tight', dpi=800)

fig = plt.figure(2, figsize=(7, 3.5), dpi=150)
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
axes.plot(params['alpha'],
          np.sqrt(-grid.cv_results_['mean_test_neg_mean_squared_error']), '-',
          lw=2, color='red', label='Root Mean Squared Error')
axes.set_xlabel('Alpha Ratio')
axes.set_ylabel('Root Mean Squared Error')
axes.set_title('Effect of Alpha Ratio on '
               'Elastic Net Regression Performance')
axes.legend(loc='best')
axes.grid(True)
axes.autoscale()
# fig.savefig('Effect of Alpha Ratio on '
#             'Elastic Net Regression Performance',
#            bbox_inches='tight', dpi=800)