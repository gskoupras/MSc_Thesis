# =============================================================================
# Scatter_plots_of_system_features_with_output.py
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read data from csv files
offers = pd.read_csv('Offers.csv', parse_dates=True, index_col=0)
bids = pd.read_csv('Bids.csv', parse_dates=True, index_col=0)
X = pd.read_csv('System Features.csv', parse_dates=True, index_col=0)

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

# Change order for visual purposes
data = data[['Ren_R', 'APXP', 'APXV', 'Rene', 'TSDF',
             'LOLP', 'Im_Pr', 'In_gen', 'DRM', 'NIV', 'Offers']]

# Plot
plt.rc('xtick', labelsize=7)
plt.rc('ytick', labelsize=7)

fig, axs = plt.subplots(2, 5, figsize=(15, 8), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace=0.2, wspace=0.25)

axs = axs.ravel()

for i in range(0, len(data.columns)-1):

    axs[i].scatter(data['Offers'], data.iloc[:, i], s=5)
    axs[i].set_title(data.columns[i], fontsize=16)
    axs[i].set_xlim([0, 700])
    if i == 1:
        axs[i].set_ylim([-50, 320])
    if i == 5:
        axs[i].set_ylim([-0.00005, 0.001])
    if i == 6:
        axs[i].set_ylim([-210, 600])
    if i == 9:
        axs[i].set_ylim([-2500, 2000])
    z = np.polyfit(data['Offers'], data.iloc[:, i], 1)
    print(z)
    p = np.poly1d(z)
    axs[i].plot(range(0, int(max(data['Offers'])+1)),
                p(range(0, int(max(data['Offers'])+1))), "r--", lw=3)

#fig.savefig('Features_vs_Output',
#            bbox_inches='tight', dpi=800)
