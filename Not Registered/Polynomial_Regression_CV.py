import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
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

# Divide features and output
# X = data.loc[:, ['TSDF']]
X = data.iloc[:, :-2]
y = data['Offers']

# Perform Split on TimeSeries data
tscv = TimeSeriesSplit(n_splits=11)

# Prepare the poly regressor
poly_reg = PolynomialFeatures(degree=2)
X_poly = poly_reg.fit_transform(X)
X_norm = (X-X_poly.mean())/X_poly.std()

start_time = time.time()
r = 2

# Perform cross validation
lr = LinearRegression()
scores = cross_validate(lr, X_norm, y, cv=tscv.split(X),
                        scoring=('neg_mean_squared_error',
                        'neg_mean_absolute_error'))

rt = round(time.time() - start_time, r)

fig = plt.figure(2, figsize=(7, 3.5), dpi=150)
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
axes.plot(-scores['test_neg_mean_absolute_error'],
          '-', lw=1.5, color='red', label='Mean Absolute Error')
axes.plot(np.sqrt(-scores['test_neg_mean_squared_error']),
          '-', lw=1.5, color='blue', label='Root Mean Squared Error')
axes.set_xlabel('Cross validation Split')
axes.set_ylabel('Metrics')
axes.set_title('Effect of TimeSeries Cross Validation on '
               'Polynomial Regression Performance')
axes.legend(loc='best')
axes.grid(True)
axes.autoscale()

print('Mean of MAE:', round(-scores['test_neg_mean_absolute_error'].mean(), r))
print('Mean of RMSE:',
      round(np.sqrt(-scores['test_neg_mean_squared_error'].mean()), r))
print('Running Time:', rt)
