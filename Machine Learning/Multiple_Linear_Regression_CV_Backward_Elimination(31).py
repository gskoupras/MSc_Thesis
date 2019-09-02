# =============================================================================
# Multiple_Linear_Regression_CV_Backward_Elimination.py
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import time
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import cross_validate

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
# y = data['Bids']


# Function that removes features based on p-values
def backward_elimination(X):
    # Include the constant column for the intercept
    cons = np.ones(shape=(len(X), 1))
    cons = pd.DataFrame(cons, index=X.index, columns=['Constant'])
    X = pd.concat([cons, X], axis=1, sort=True)

    terminal = 0
    r = 2

    while terminal == 0:
        regressor_OLS = sm.OLS(y, X).fit()
        if regressor_OLS.pvalues[regressor_OLS.pvalues.idxmax()] > 0.05:
            print('Drop feature:',
                  regressor_OLS.pvalues.idxmax(), 'with p-value:',
                  round(regressor_OLS.pvalues[regressor_OLS.pvalues.idxmax()], r))
            X.drop(regressor_OLS.pvalues.idxmax(), axis=1, inplace=True)
        else:
            print('Best model has been found!')
            print('Features left:')
            print(X.columns)
            terminal = 1

    X.drop('Constant', axis=1, inplace=True)
    return X


X = backward_elimination(X)

# Calculate running time
start_time = time.time()

r = 2

# Perform Split on TimeSeries data
tscv = TimeSeriesSplit(n_splits=11)

# Perform cross validation
lr = LinearRegression()
scores = cross_validate(lr, X, y, cv=tscv.split(X),
                        scoring=('neg_mean_squared_error',
                        'neg_mean_absolute_error'))

rt = round(time.time() - start_time, r)

# Plots
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)

s = 50
fig = plt.figure(2, figsize=(7, 3.5), dpi=150)
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
axes2 = axes.twinx()
axes.plot(-scores['test_neg_mean_absolute_error'],
          '-', lw=2, color='red',
          label='Mean Absolute Error', zorder=10)
axes.scatter(list(range(0, len(-scores['test_neg_mean_absolute_error']))),
             -scores['test_neg_mean_absolute_error'],
             marker="o",
             color='red', s=s, zorder=10)
axes2.plot(np.sqrt(-scores['test_neg_mean_squared_error']),
           '-', lw=2, color='blue',
           label='Root Mean Squared Error', zorder=10)
axes2.scatter(list(range(0, len(-scores['test_neg_mean_squared_error']))),
              np.sqrt(-scores['test_neg_mean_squared_error']),
              marker="o",
              color='blue', s=s, zorder=10)
axes.set_xlabel('Cross validation Split', fontsize=16)
axes.set_ylabel('Mean Absolute Error', fontsize=16)
axes2.set_ylabel('Root Mean Squared Error', fontsize=16)
axes.set_title('Effect of Timeseries Cross Validation on '
               'Multiple Linear Regression Performance', fontsize=18)
axes.set_xticks(range(0, len(-scores['test_neg_mean_absolute_error'])))
axes.set_xticklabels(range(1, len(-scores['test_neg_mean_absolute_error'])+1))
fig.legend(bbox_to_anchor=(0.8, 0.9),
           bbox_transform=axes.transAxes, fontsize=16)
axes.grid(True, zorder=10)
axes.autoscale()
# fig.savefig('Effect of TimeSeries Cross Validation on '
#            'Multiple Linear Regression Performance',
#            bbox_inches='tight', dpi=800)

print('Mean of MAE:', round(-scores['test_neg_mean_absolute_error'].mean(), r))
print('Mean of RMSE:',
      round(np.sqrt(-scores['test_neg_mean_squared_error'].mean()), r))
print('Mean of MSE:',
      round(-scores['test_neg_mean_squared_error'].mean(), r))
print('Running Time:', rt)
