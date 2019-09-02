# =============================================================================
# Correlation_all_features.py
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
# data = data.loc[data.index > 2018000000, :]

# Handle missing data
data.fillna(data.median(), inplace=True)

# Predict 1h ahead instead of same time
data['Offers'] = data['Offers'].shift(-2)
data['Offers'].fillna(method='ffill', inplace=True)

# Calculate the correlation matrix
corr = data.corr()

corr_h = corr.copy()
corr_l = corr.copy()
corr_h['Offers'].iloc[:-1] = corr_h['Offers'].iloc[:-1][abs(corr_h['Offers'].iloc[:-1])>0.1]
corr_l['Offers'].iloc[:-1] = corr_l['Offers'].iloc[:-1][abs(corr_l['Offers'].iloc[:-1])<0.1]

# Plots
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)

fig = plt.figure(1)
ax = fig.add_subplot(111)
cax = ax.matshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0, len(data.columns), 1)
ax.set_xticks(ticks)
plt.xticks(rotation=60)
ax.set_yticks(ticks)
ax.set_xticklabels(data.columns)
ax.set_yticklabels(data.columns)
plt.show()
# fig.savefig('Correlation of System Features',
#             bbox_inches='tight', dpi=800)

fig = plt.figure(2, figsize=(7, 3.5), dpi=150)
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
# axes.bar(np.arange(len(corr['Offers'].iloc[:-1])), corr['Offers'].iloc[:-1])
axes.bar(np.arange(len(corr_h['Offers'].iloc[:-1])),
         corr_h['Offers'].iloc[:-1], color='green',
         label='Highly Correlated Features')
axes.bar(np.arange(len(corr_l['Offers'].iloc[:-1])),
         corr_l['Offers'].iloc[:-1], color='red',
         label='Lowly Correlated Features')
axes.axhline(y=0.1, linewidth=1.5, color='#006eff')
axes.axhline(y=-0.1, linewidth=1.5, color='#006eff')
axes.set_xticks(np.arange(len(corr['Offers'])))
axes.set_xticklabels(corr['Offers'].index, fontsize=16, rotation=0)
axes.set_ylabel('Correlation', fontsize=16)
axes.set_title('Correlation of All Features with Offer Prices', fontsize=18)
axes.legend(loc='upper left', fontsize=14)
axes.grid(True)
axes.autoscale()
# fig.savefig('Correlation bar chart: All features',
#             bbox_inches='tight', dpi=800)

# Using Pearson Correlation
plt.figure(3, figsize=(12, 10))
ax = sns.heatmap(corr, annot=True,
                 lw=1, fmt=".2f",
                 annot_kws={"size": 14},
                 cmap='RdBu_r', vmin=-1, vmax=1)
cax = plt.gcf().axes[-1]
cax.tick_params(labelsize=18)
ax.set_title('Correlation Matrix: All Features', fontsize=20)
plt.yticks(rotation=0, fontsize=18)
plt.xticks(fontsize=18)
plt.show()
# plt.savefig('Correlation heat map: All Features',
#             bbox_inches='tight', dpi=800)
