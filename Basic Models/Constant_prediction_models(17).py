# =============================================================================
# Constant_prediction_models.py
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

# Divide features and output
X = data.iloc[:, :-2]
y = data['Offers']

X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, shuffle=False)

mae = np.array([])
rmse = np.array([])
r = 2

for i in np.arange(95.0, 114.0, 0.1):
    avg = i*np.ones(shape=(len(y_test), 1))
    y_pred = pd.DataFrame(avg, index=y_test.index, columns=['Predictions'])
    mae = np.append(mae, round(metrics.mean_absolute_error(y_test, y_pred), r))
    rmse = np.append(rmse, round(np.sqrt(metrics.mean_squared_error(y_test,
                                                                    y_pred)), r))
# Plot
size = 10
fig = plt.figure(1, figsize=(7, 3.5), dpi=150)
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
axes2 = axes.twinx()
for i in np.arange(950, 1140):
    axes.scatter(i/10, mae[i-950], marker="o",
                 color='red', s=size, zorder=1)
axes.plot(np.arange(95.0, 114.0, 0.1), mae, color='red',
          lw=1, zorder=0, label='Mean Absolute Error')
axes.plot(np.arange(95.0, 114.0, 0.1), min(mae)*np.ones((190, 1)),
          color='black', lw=1, zorder=0)
axes.text(94.5, min(mae)+0.1, str(min(mae)), fontsize=16)
for i in np.arange(950, 1140):
    axes2.scatter(i/10, rmse[i-950], marker="o",
                  color='blue', s=size, zorder=1)
axes2.plot(np.arange(95.0, 114.0, 0.1), rmse, color='blue',
           lw=1, zorder=0, label='Root Mean Squared Error')
axes2.text(113.2, min(rmse)+0.01, str(min(rmse)), fontsize=16)
axes.set_xlabel('Constant Prediction', fontsize=16)
axes.set_ylabel('Mean Absolute Error', fontsize=16)
axes2.set_ylabel('Root Mean Squared Error', fontsize=16)
axes.set_title('Constant Prediction Methods Performance',
               fontsize=18)
axes.set_xticks(np.arange(95.0, 114.0, 0.1), 10)
fig.legend(bbox_to_anchor=(0.6, 0.9),
           bbox_transform=axes.transAxes, fontsize=16)
axes.grid(True)
axes.autoscale()
# fig.savefig('Constant Prediction Methods Performance',
#            bbox_inches='tight', dpi=800)






