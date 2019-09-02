import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import cross_val_score, cross_validate
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

mae = np.array([])
rmse = np.array([])
r = 2

for i in range(0, 11):
    if i == 0:
        data_new = data.dropna()
    elif i == 1:
        data_new = data.interpolate(method='linear')
    elif i == 2:
        data_new = data.fillna(method='pad')
    elif i == 3:
        data_new = data.fillna(method='bfill')
    elif i == 4:
        data_new = data.fillna(data.mean())
    elif i == 5:
        data_new = data.interpolate(method='quadratic')
    elif i == 6:
        data_new = data.interpolate(method='cubic')
    elif i == 7:
        data_new = data.interpolate(method='polynomial', order=5)
    elif i == 8:
        data_new = data.interpolate(method='slinear')
    elif i == 9:
        data_new = data.interpolate(method='zero')
    elif i == 10:
        data_new = data.fillna(data.median())

    data_new = data_new.dropna()   # For remaining null values

    # Predict 1h ahead instead of same time
    data_new['Offers'] = data_new['Offers'].shift(-2)
    data_new['Offers'].fillna(method='ffill', inplace=True)

    X = data_new.iloc[:, :-2]
    y = data_new['Offers']

    X_train, X_test, y_train, y_test = train_test_split(
                         X, y, test_size=0.1, shuffle=False)

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)

    mae = np.append(mae, round(metrics.mean_absolute_error(y_test,
                                                           y_pred), r))
    rmse = np.append(rmse,
                     round(np.sqrt(metrics.mean_squared_error(y_test,
                                                              y_pred)), r))

size = 150
labels = ['Drop Values', 'Linear Interpolation', 'Forward Propagation',
          'Backward Propagation', 'Mean Value', 'Quadratic Interpolation',
          'Cubic Interpolation', 'Polynomial Interpolation (5th)',
          'Spline Linear Interpolation', 'Zero Value', 'Medial Value']
fig = plt.figure(1, figsize=(7, 3.5), dpi=150)
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
for i in range(0, len(labels)):
    axes.scatter(i, mae[i], marker="o", s=size, label=labels[i])
axes.set_xlabel('Method of handling missing values')
axes.set_ylabel('Mean Absolute Error')
axes.set_title('Effect of '
               'handling of missing data on Linear Regression Performance')
axes.legend(loc='best')
axes.grid(True)
axes.autoscale()

fig = plt.figure(2, figsize=(7, 3.5), dpi=150)
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
for i in range(0, len(labels)):
    axes.scatter(i, rmse[i], marker="o", s=size, label=labels[i])
axes.set_xlabel('Method of handling missing values')
axes.set_ylabel('Root Mean Square Error')
axes.set_title('Effect of '
               'handling of missing data on Linear Regression Performance')
axes.legend(loc='best')
axes.grid(True)
axes.autoscale()

print('')
print("   ******** Evaluation Metrics ********    ")
print("Mean Absolute Error:")
print(mae)
print("Root Mean Squared Error:")
print(rmse)
