# =============================================================================
# plot_MAs.py
# =============================================================================
import pandas as pd
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

# Handle missing data
data.fillna(data.median(), inplace=True)

# Keep only data from 2018 (for simplicity)
data = data.loc[data.index > 2018000000, :]

smas = [5, 20]
emas = [5, 20]
SMA = pd.DataFrame()
EMA = pd.DataFrame()
plot_together = True

for i in smas:
    SMA.loc[:, '{}_sma'.format(i)] = data['Offers'].rolling(window=i).mean()
    SMA.loc[:, '{}_sma'.format(i)].fillna(SMA.loc[:, '{}_sma'.format(i)].mean(), inplace=True)

for i in emas:
    EMA.loc[:, '{}_ema'.format(i)] = data['Offers'].ewm(span=i, adjust=False).mean()
    EMA.loc[:, '{}_ema'.format(i)].fillna(EMA.loc[:, '{}_ema'.format(i)].mean(), inplace=True)

cor = pd.DataFrame()
for i in smas:
    cor = pd.concat([cor, SMA.loc[:, '{}_sma'.format(i)]], axis=1, sort=True)
for i in emas:
    cor = pd.concat([cor, EMA.loc[:, '{}_ema'.format(i)]], axis=1, sort=True)
corr = cor.corr()

# Plots
plt.rc('xtick', labelsize=20)
plt.rc('ytick', labelsize=20)

if plot_together is True:
    plt.figure(1)
    plt.plot(range(len(data['Offers'])), data['Offers'],
             label='Offer Price', lw=2)
    for i in smas:
        plt.plot(range(len(SMA.loc[:, '{}_sma'.format(i)])),
                 SMA.loc[:, '{}_sma'.format(i)], label='{} SMA'.format(i), lw=2)
    for i in emas:
        plt.plot(range(len(EMA.loc[:, '{}_ema'.format(i)])),
                 EMA.loc[:, '{}_ema'.format(i)], label='{} EMA'.format(i), lw=2)
    plt.ylabel('Offer Price (GBP/MWh)', fontsize=20)
    plt.title('Highest Offers Accepted, SMAs and EMAs', fontsize=22)
    plt.xlabel('Date and Settlement Period', fontsize=20)
    plt.legend(loc='best', fontsize=20)
    plt.grid(True)
    plt.autoscale()
    plt.tight_layout()
#    plt.savefig("Highest Offers Accepted, SMAs and EMAs.png")
    plt.show()

else:
    plt.figure(1)
    plt.plot(range(len(data['Offers'])), data['Offers'], label='Offer Price')
    for i in smas:
        plt.plot(range(len(SMA.loc[:, '{}_sma'.format(i)])),
                 SMA.loc[:, '{}_sma'.format(i)], label='{} SMA'.format(i))
    plt.ylabel('Offer Price (GBP/MWh)')
    plt.title('Highest Offers Accepted and SMAs')
    plt.xlabel('Date and Settlement Period')
    plt.legend(loc='best')
    plt.grid(True)
    plt.autoscale()
    plt.tight_layout()
#    plt.savefig("Highest Offers Accepted and SMAs.png")
    plt.show()

    plt.figure(2)
    plt.plot(range(len(data['Offers'])), data['Offers'], label='Offer Price')
    for i in emas:
        plt.plot(range(len(EMA.loc[:, '{}_ema'.format(i)])),
                 EMA.loc[:, '{}_ema'.format(i)], label='{} EMA'.format(i))
    plt.ylabel('Offer Price (GBP/MWh)')
    plt.title('Highest Offers Accepted and EMAs')
    plt.xlabel('Date and Settlement Period')
    plt.legend(loc='best')
    plt.grid(True)
    plt.autoscale()
    plt.tight_layout()
#    plt.savefig("Highest Offers Accepted and EMAs.png")
    plt.show()
