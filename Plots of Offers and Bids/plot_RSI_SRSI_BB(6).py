# =============================================================================
# plot_RSI_SRSI_BB.py
# =============================================================================
import pandas as pd
import matplotlib.pyplot as plt
from talib import RSI, STOCHRSI
from talib import BBANDS as BB

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

# BB
up, mid, low = BB(data['Offers'].values, timeperiod=15,
                  nbdevup=2, nbdevdn=2, matype=0)

# RSI
rsi = RSI(data['Offers'].values, timeperiod=14)
rsi = pd.DataFrame(rsi, columns=['RSI'])
rsi.fillna(rsi.mean(), inplace=True)
rsi_up = rsi[rsi > 70]
rsi_low = rsi[rsi < 35]

# StochRSI
fastk, fastd = STOCHRSI(data['Offers'].values, timeperiod=100,
                        fastk_period=100, fastd_period=15, fastd_matype=0)
fastk = pd.DataFrame(fastk, columns=['fastk'])
fastd = pd.DataFrame(fastd, columns=['fastd'])
fastk.fillna(fastk.mean(), inplace=True)
fastd.fillna(fastd.mean(), inplace=True)

# Plots
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)

fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True, figsize=(20, 14), dpi=150)
ax0.plot(range(len(data['Offers'])), data['Offers'],
         label='Offer Price', lw=1)
ax0.set_xlabel('Date and Settlement Period', fontsize=12)
ax0.set_ylabel('Offer Price (GBP/MWh)', fontsize=12)
ax0.set_title('Highest Offers Accepted and RSI', fontsize=14)
# ax0.legend(loc='best')
ax0.grid(True)
ax1.plot(range(len(data['Offers'])), rsi, label='RSI', lw=1, color='#d499ff')
ax1.scatter(range(len(data['Offers'])), rsi_up, label='High RSI',
            s=20, color='#ff6b61')
ax1.scatter(range(len(data['Offers'])), rsi_low, label='Low RSI',
            s=20, color='#87ff8d')
ax1.fill_between(range(len(data['Offers'])),
                 y1=35, y2=70, color='#8ce0ff', alpha='0.3')
ax1.set_xlabel('Date and Settlement Period', fontsize=12)
ax1.set_ylabel('RSI', fontsize=12)
ax1.legend(loc='best')
ax1.grid(True)
# fig.savefig('Highest Offers Accepted and RSI',
#            bbox_inches='tight', dpi=800)

fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True, figsize=(20, 14), dpi=150)
ax0.plot(range(len(data['Offers'])), data['Offers'],
         label='Offer Price', lw=1)
ax0.set_xlabel('Date and Settlement Period', fontsize=12)
ax0.set_ylabel('Offer Price (GBP/MWh)', fontsize=12)
ax0.set_title('Highest Offers Accepted and StochRSI', fontsize=14)
ax0.legend(loc='best')
ax0.grid(True)
ax1.plot(range(len(data['Offers'])), fastk,
         label='Fast K', lw=1, color='#d499ff')
ax1.plot(range(len(data['Offers'])), fastd,
         label='Fast D', lw=1, color='#e0417e')
ax1.fill_between(range(len(data['Offers'])),
                 y1=30, y2=70, color='#8ce0ff', alpha='0.3')
ax1.set_xlabel('Date and Settlement Period', fontsize=12)
ax1.set_ylabel('StochRSI', fontsize=12)
ax1.legend(loc='best')
ax1.grid(True)
# fig.savefig('Highest Offers Accepted and StochRSI',
#            bbox_inches='tight', dpi=800)

fig = plt.figure(3, figsize=(7, 3.5), dpi=150)
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
axes.plot(range(len(data['Offers'])), data['Offers'],
          label='Offer Price', lw=1)
axes.plot(range(len(data['Offers'])), mid,
          '-', lw=0.5, color='red', label='BB mid')
axes.plot(range(len(data['Offers'])), up,
          '-', lw=0.5, color='#2c51bf', label='BB up')
axes.plot(range(len(data['Offers'])), low,
          '-', lw=0.5, color='#2c51bf', label='BB low')
axes.fill_between(range(len(data['Offers'])),
                  y1=low, y2=up, color='#8ce0ff', alpha='0.3')
#axes.plot(range(len(data['Offers'])), data['Offers'].values - mid,
#          '-', lw=1, color='green', label='BB indicator')
axes.plot(range(len(data['Offers'])), up-low,
          '-', lw=1, color='#7c9c00', label='Volatility Indicator')
axes.set_xlabel('Date and Settlement Period', fontsize=16)
axes.set_ylabel('Offer Price (GBP/MWh)', fontsize=16)
axes.set_title('Highest Offers Accepted and Bollinger Bands', fontsize=18)
axes.legend(loc='best')
axes.grid(True)
axes.autoscale()
# fig.savefig('Highest Offers Accepted and Bollinger Bands',
#            bbox_inches='tight', dpi=800)
