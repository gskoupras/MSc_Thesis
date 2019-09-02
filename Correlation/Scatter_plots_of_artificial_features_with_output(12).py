# =============================================================================
# Scatter_plots_of_artificial_features_with_output.py
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read data from csv files
offers = pd.read_csv('Offers.csv', parse_dates=True, index_col=0)
bids = pd.read_csv('Bids.csv', parse_dates=True, index_col=0)
X = pd.read_csv('Artificial Features.csv', parse_dates=True, index_col=0)

# Get rid of extreme values
offers = offers[offers < 2000]
bids = bids[bids > -250]

# Connect all together
data = pd.concat([X, offers], axis=1, sort=True)

# Sort data
data.sort_index(inplace=True)

# Keep only data from 2018 (for simplicity)
# data = data.loc[data.index > 2018000000, :]

# Handle missing data
data.fillna(data.median(), inplace=True)

# Predict 1h ahead instead of same time
data['Offers'] = data['Offers'].shift(-2)
data['Offers'].fillna(method='ffill', inplace=True)

# Plot
plt.rc('xtick', labelsize=7)
plt.rc('ytick', labelsize=7)

fig, axs = plt.subplots(3, 4, figsize=(20, 14), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace=0.3, wspace=0.15)

axs = axs.ravel()

for i in range(0, len(data.columns)):

    axs[i].scatter(data['Offers'], data.iloc[:, i], s=5)
    axs[i].set_title(data.columns[i], fontsize=16)
    axs[i].set_xlim([0, 1000])
    if i == 0:
        axs[i].set_ylim([-1000, 1000])
    if i == 1:
        axs[i].set_ylim([-2, 10])
    if i == 2:
        axs[i].set_ylim([-1000, 1000])
    if i == 3:
        axs[i].set_ylim([-100, 1550])
    if i == 4:
        axs[i].set_ylim([-100, 1500])
    if i == 5:
        axs[i].set_ylim([-100, 1250])
    if i == 6:
        axs[i].set_ylim([0, 105])
    if i == 7:
        axs[i].set_ylim([-5, 105])
    if i == 10:
        axs[i].set_ylim([-70, 1750])
    if i == 11:
        axs[i].set_ylim([-50, 1200])
    z = np.polyfit(data['Offers'], data.iloc[:, i], 1)
    print(z)
    p = np.poly1d(z)
    axs[i].plot(range(0, int(max(data['Offers'])+1)),
                p(range(0, int(max(data['Offers'])+1))), "r--", lw=3)

# fig.savefig('Features_vs_Output',
#             bbox_inches='tight', dpi=800)
