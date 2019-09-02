# =============================================================================
# Plots_of_output_data_together.py
# =============================================================================
import pandas as pd
import matplotlib.pyplot as plt

# Get data
offers = pd.read_csv('Offers.csv', parse_dates=True, index_col=0)
bids = pd.read_csv('Bids.csv', parse_dates=True, index_col=0)

# Means and Medians
print('Offers mean and median before:')
print(round(offers.iloc[:, 0].mean(), 1), offers.iloc[:, 0].median())
print('Bids mean and median before:')
print(round(bids.iloc[:, 0].mean(), 1), bids.iloc[:, 0].median())

# Limit values
offers = offers[offers < 2000]
bids = bids[bids > -250]

# Keep only the year 2018
offers = offers.loc[offers.index > 2018000000, :]
bids = bids.loc[bids.index > 2018000000, :]

# Means and Medians
print('Offers mean and median after:')
print(round(offers.iloc[:, 0].mean(), 1), offers.iloc[:, 0].median())
print('Bids mean and median after:')
print(round(bids.iloc[:, 0].mean(), 1), bids.iloc[:, 0].median())

# Plots
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)

fig = plt.figure(1, figsize=(7, 3.5), dpi=150)
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
axes.plot(range(len(offers)), offers['Offers'], label='Highest Offers')
axes.plot(range(len(bids)), bids['Bids'], label='Lowest Bids')
axes.set_xlabel('Date and Settlement Period', fontsize=9)
axes.set_ylabel('Offer/Bid Price (GBP/MWh)', fontsize=9)
axes.set_title('Highest Offers and Lowest Bids Accepted', fontsize=10)
axes.grid(True)
axes.legend(loc='best', fontsize=10)
axes.autoscale()
fig.savefig('Highest Offers and Lowest Bids Accepted',
            bbox_inches='tight', dpi=800)
