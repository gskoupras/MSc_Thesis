# =============================================================================
# Create_artificial_features.py
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from talib import RSI, STOCHRSI
from talib import BBANDS as BB

# Read data from csv files
offers = pd.read_csv('Offers.csv', parse_dates=True, index_col=0)

# Get rid of extreme values
offers = offers[offers < 2000]

# Handle missing data
offers.fillna(offers.median(), inplace=True)

# 1st feature (gradient of 1st order):
diff = offers.diff(periods=1, axis=0)
diff.columns = ['Gr1']

# 2nd feature (return):
ret = pd.DataFrame.copy(diff)
for i in range(1, len(offers['Offers'])):
    if offers.iloc[i-1, 0] != 0:
        ret.iloc[i, 0] = ret.iloc[i, 0]/offers.iloc[i-1, 0]
    else:
        ret.iloc[i, 0] = ret.iloc[i, 0]/20
ret.columns = ['Ret']

# Plot
fig = plt.figure(1, figsize=(7, 3.5), dpi=150)
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
axes2 = axes.twinx()
axes.plot(range(len(offers['Offers'])), offers['Offers'],
          label='Offer Price', lw=2)
axes2.plot(range(len(offers['Offers'])), ret['Ret'],
           '-', lw=2, color='red', label='Return')
axes.set_xlabel('Date and Settlement Period', fontsize=16)
axes.set_ylabel('Offer Price (GBP/MWh)', fontsize=16)
axes2.set_ylabel('Return p.u.', fontsize=16)
axes.set_title('Highest Offers Accepted and Returns', fontsize=18)
fig.legend(loc="upper right", bbox_to_anchor=(1, 1),
           bbox_transform=axes.transAxes, fontsize=16)
axes.grid(True)
axes.autoscale()

# 3rd feature (gradient of 2nd order):
grad = pd.DataFrame.copy(offers)
x = [1, 2, 3]
for i in range(2, len(offers['Offers'])):
    y = [offers['Offers'].iloc[i-2],
         offers['Offers'].iloc[i-1],
         offers['Offers'].iloc[i]]
    z = np.polyfit(x, y, 2)
    grad.iloc[i] = z[0]
grad.iloc[0] = grad.median()
grad.iloc[1] = grad.median()
grad.columns = ['Gr2']

# 4th feature (actual offer - previous value):
offer_prev = pd.DataFrame.copy(offers)
offer_prev.columns = ['Prev']

# 5th feature (SMA):
sma = offers.rolling(window=20).mean()
sma.columns = ['SMA20']

# 6th feature (EMA):
ema = offers.ewm(span=10, adjust=False).mean()
ema.columns = ['EMA10']

# 7th feature (Relative Strength Index)
rsi = RSI(offers['Offers'].values, timeperiod=10)
rsi = pd.DataFrame(rsi, columns=['RSI'], index=offers['Offers'].index)
rsi.fillna(rsi.median(), inplace=True)

# 8th feature (STOCHRSI):
fastk, fastd = STOCHRSI(offers['Offers'].values, timeperiod=20,
                        fastk_period=20, fastd_period=10, fastd_matype=0)
fastk = pd.DataFrame(fastk, columns=['StochRSI_K'],
                     index=offers['Offers'].index)
fastd = pd.DataFrame(fastd, columns=['SRSID'],
                     index=offers['Offers'].index)
fastk.fillna(fastk.median(), inplace=True)
fastd.fillna(fastd.median(), inplace=True)

# 9th feature (Bollinger Bands)
up, mid, low = BB(offers['Offers'].values, timeperiod=15,
                  nbdevup=2, nbdevdn=2, matype=0)
BB = offers['Offers'].values - mid
BB = pd.DataFrame(BB, columns=['BB'], index=offers['Offers'].index)
BB.fillna(BB.median(), inplace=True)

# 10th feature (BB index of standard deviation):
BB_width = up-low
BB_width = pd.DataFrame(BB_width,
                        columns=['BBw'], index=offers['Offers'].index)
BB_width.fillna(BB_width.mean(), inplace=True)

# 11th feature (Timer until next spike):
sp_tim = pd.DataFrame.copy(offers)
sp_tim.columns = ['Spk']
n = 0
for i in range(0, len(offers)):
    if offers['Offers'].iloc[i] >= 200:
        n = 0
    else:
        n = n+1
    sp_tim['Spk'].iloc[i] = n

# Plot
fig = plt.figure(2, figsize=(7, 3.5), dpi=150)
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
axes.plot(range(len(offers['Offers'])), offers['Offers'],
          label='Offer Price', lw=2.5)
axes.plot(range(len(offers['Offers'])), sp_tim['Spk'],
          '-', lw=2, color='red', label='Spike Timer')
axes.set_xlabel('Date and Settlement Period', fontsize=16)
axes.set_ylabel('Offer Price (GBP/MWh)', fontsize=16)
axes.set_title('Highest Offers Accepted and Spike Timer ', fontsize=18)
axes.legend(loc='best', fontsize=14)
axes.grid(True)
axes.autoscale()

# 12th feature (Median of previous prices)
med = pd.DataFrame.copy(offers)
for i in range(2, len(offers['Offers'])):
    med.iloc[i, 0] = offers['Offers'].iloc[i-2:i+1].median()
med.columns = ['Med']

# Create final array of features
X = pd.concat([diff, ret, grad, offer_prev,
               sma, ema, rsi, fastd, BB, BB_width,
               sp_tim, med], axis=1)

# Save to csv file
X.to_csv('Artificial Features.csv')
