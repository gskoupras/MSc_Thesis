# =============================================================================
# Plots_of_output_data_separate.py
# =============================================================================
import pandas as pd
import matplotlib.pyplot as plt

# Get data
offers = pd.read_csv('Offers.csv', parse_dates=True, index_col=0)
bids = pd.read_csv('Bids.csv', parse_dates=True, index_col=0)

bids = bids[bids > -250]
offers = offers[offers < 2000]

# offers = offers.loc[offers.index > 2018000000, :]
# bids = bids.loc[bids.index > 2018000000, :]

# Plots
plt.rc('xtick', labelsize=7)
plt.rc('ytick', labelsize=7)

fig = plt.figure(1, figsize=(7, 3.5), dpi=150)
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
axes.plot(range(len(offers)), offers['Offers'])
axes.set_xlabel('Date and Settlement Period', fontsize=9)
axes.set_ylabel('Offer Price (GBP/MWh)', fontsize=9)
axes.set_title('Highest Offers Accepted', fontsize=10)
axes.grid(True)
axes.autoscale()
fig.savefig('Highest Offers Accepted',
            bbox_inches='tight', dpi=800)


fig = plt.figure(2, figsize=(7, 3.5), dpi=150)
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
axes.plot(range(len(bids)), bids['Bids'])
axes.set_xlabel('Date and Settlement Period', fontsize=9)
axes.set_ylabel('Bid Price (GBP/MWh)', fontsize=9)
axes.set_title('Lowest Bids Accepted', fontsize=10)
axes.grid(True)
axes.autoscale()
fig.savefig('Lowest Bids Accepted',
            bbox_inches='tight', dpi=800)
