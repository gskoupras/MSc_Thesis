import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.model_selection import train_test_split
from sklearn import metrics
import statsmodels.formula.api as sm
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
# X = data.iloc[:, :-2]
X = data.loc[:, ['Prev', 'EMA10']]
y = data['Offers']

# Perform Split on TimeSeries data
tscv = TimeSeriesSplit(n_splits=11)

deg = list(range(1, 6))
mae = np.array([])
rmse = np.array([])

for i in deg:
    # Prepare the poly regressor
    poly_reg = PolynomialFeatures(degree=i)
    X_poly = poly_reg.fit_transform(X)

    # Make the predictor FOR THE TEST SET
    lr = LinearRegression()
    scores = cross_validate(lr, X_poly, y, cv=tscv.split(X),
                            scoring=('neg_mean_squared_error',
                            'neg_mean_absolute_error'))

    mae = np.append(mae, -scores['test_neg_mean_absolute_error'].mean())
    rmse = np.append(rmse,
                     np.sqrt(-scores['test_neg_mean_squared_error'].mean()))

size = 100
fig = plt.figure(1, figsize=(7, 3.5), dpi=150)
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
axes.plot(
          mae, lw=1.5, color='red', label='Mean Absolute Error')
axes.scatter(list(range(0, len(mae))),
             mae, marker="o", color='blue',
             s=size, label='Mean Absolute Error (points)')
axes.scatter(mae.argmin(),
             min(mae), marker="o", color='green',
             s=size, label='Best')
axes.annotate('Best Degree = {}'.format(mae.argmin()+1),
              xy=(mae.argmin(), 1.001*min(mae)), xycoords='data',
              xytext=(mae.argmin(), 1.01*min(mae)), textcoords='data',
              arrowprops=dict(arrowstyle="->", facecolor='black'),
              horizontalalignment='center', verticalalignment='top')
axes.set_xlabel('Degree')
axes.set_ylabel('Mean Absolute Error')
axes.set_title('Effect of degree on Polynomial Regression Performance')
axes.legend(loc='best')
axes.set_xticks(list(range(0, len(mae), 2)))
axes.set_xticklabels(list(range(1, len(mae)+1, 2)))
axes.grid(True)
axes.autoscale()
# fig.savefig('Effect of degree on Polynomial Regression Performance',
#            bbox_inches='tight', dpi=800)

fig = plt.figure(2, figsize=(7, 3.5), dpi=150)
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
axes.plot(
          rmse, lw=1.5, color='red', label='Root Mean Squared Error')
axes.scatter(list(range(0, len(rmse))),
             rmse, marker="o", color='blue',
             s=size, label='Root Mean Squared Error (points)')
axes.scatter(rmse.argmin(),
             min(rmse), marker="o", color='green',
             s=size, label='Best')
axes.annotate('Best Degree = {}'.format(rmse.argmin()+1),
              xy=(rmse.argmin(), 1.0005*min(rmse)), xycoords='data',
              xytext=(0.9*rmse.argmin(), 1.004*min(rmse)), textcoords='data',
              arrowprops=dict(arrowstyle="->", facecolor='black'),
              horizontalalignment='center', verticalalignment='top')
axes.set_xlabel('Degree')
axes.set_ylabel('Root Mean Squared Error')
axes.set_title('Effect of degree on Polynomial Regression Performance')
axes.legend(loc='best')
axes.set_xticks(list(range(0, len(rmse), 2)))
axes.set_xticklabels(list(range(1, len(rmse)+1, 2)))
axes.grid(True)
axes.autoscale()
## fig.savefig('Effect of degree on Polynomial Regression Performance',
##            bbox_inches='tight', dpi=800)
