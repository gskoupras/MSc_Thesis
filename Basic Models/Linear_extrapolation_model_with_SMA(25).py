# =============================================================================
# Linear_extrapolation_model_with_SMA.py
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read data from csv files
offers = pd.read_csv('Offers.csv', parse_dates=True, index_col=0)
bids = pd.read_csv('Bids.csv', parse_dates=True, index_col=0)
X = pd.read_csv('Features.csv', parse_dates=True, index_col=0)

# Get rid of extreme values
offers = offers[offers < 2000]
bids = bids[bids > -250]

# Connect all together
data = pd.concat([X, offers], axis=1, sort=True)

# Sort data
data.sort_index(inplace=True)

# Keep only data from 2018 (for simplicity)
data = data.loc[data.index > 2018110000, :]

# Handle missing data
data.fillna(data.median(), inplace=True)

# Model
y = data['Offers'].rolling(window=30).mean()
y.fillna(y.mean(), inplace=True)
y_pred = pd.DataFrame(data=y)

i = 1

y_pred = pd.DataFrame(data=y)
x = list(range(1, i+2))
for j in range(i+2, len(y)):
    y_c = np.array([])
    for k in range(i+2, 1, -1):
        y_c = np.append(y_c, y.iloc[j-k])
    z = np.polyfit(x, y_c, i)
    f = np.poly1d(z)
    y_pred.iloc[j] = f(i+3)

# Plots
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)

fig = plt.figure(1, figsize=(7, 3.5), dpi=150)
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
axes.plot(range(len(y)), data['Offers'], zorder=1, lw=3,
          color='red', label='Real Price')
axes.plot(range(len(y)), y, zorder=3, lw=3,
          color='#7bf016', label='30 SMA')
axes.plot(range(len(y)), y_pred, zorder=2, lw=3,
          color='blue', label='Linear Extrapolation on SMA')
axes.set_title('Linear Extrapolation Model Behaviour on 30 SMA', fontsize=18)
axes.set_xlabel('Day and SP', fontsize=16)
axes.set_ylabel('Offer Price', fontsize=16)
axes.legend(loc='best', fontsize=16)
axes.grid(True)
axes.autoscale()
