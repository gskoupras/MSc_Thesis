# =============================================================================
# Create_all_features.py
# =============================================================================
import pandas as pd
import numpy as np
from talib import RSI, STOCHRSI
from talib import BBANDS as BB

# System Features

# Get the files
gpt = pd.read_csv('actual_aggregated_generation_per_type.csv',
                  parse_dates=True, index_col=0)
apx = pd.read_csv('apx_day_ahead.csv',
                  parse_dates=True, index_col=0)
genfor = pd.read_csv('day_ahead_generation_forecast_wind_and_solar.csv',
                     parse_dates=True, index_col=0)
demfor = pd.read_csv('forecast_day_and_day_ahead_demand_data.csv',
                     parse_dates=True, index_col=0)
imb = pd.read_csv('derived_system_wide_data.csv',
                  parse_dates=True, index_col=0)
inter = pd.read_csv('interconnectors.csv',
                    parse_dates=True, index_col=0)
prob = pd.read_csv('loss_of_load_probability.csv',
                   parse_dates=True, index_col=0)
outturn = pd.read_csv('initial_demand_outturn.csv',
                      parse_dates=True, index_col=0)

# Sort the files in ascending order based on the index (timestamp)
gpt = gpt.sort_index(ascending=True)
apx = apx.sort_index(ascending=True)
genfor = genfor.sort_index(ascending=True)
demfor = demfor.sort_index(ascending=True)
imb = imb.sort_index(ascending=True)
inter = inter.sort_index(ascending=True)
prob = prob.sort_index(ascending=True)
outturn = outturn.sort_index(ascending=True)

# Make necessary arrangements to derive the features
gpt['Fossil'] = gpt['FossilGas']+gpt['FossilHardCoal']+gpt['FossilOil']+gpt['Other']
gpt['RENE'] = gpt['Biomass']+gpt['HydroPumpedStorage']+gpt['HydroRunOfRiver']+gpt['Nuclear']+gpt['OffWind']+gpt['OnWind']+gpt['Solar']                                               
gpt['RENE Ratio'] = (gpt['RENE']/(gpt['Fossil']+gpt['RENE']))

genfor['Total_RENE'] = (genfor['solar']+genfor['wind_off']+genfor['wind_on'])

inter['Total_inter_gen'] = inter['intewGeneration']+inter['intfrGeneration']+inter['intirlGeneration']+inter['intnedGeneration']

# Isolate features
rene_ratio = gpt['RENE Ratio']
price = apx['APXPrice']
price_vol = apx['APXVolume']
rene_gen = genfor['Total_RENE']
dem = demfor['TSDF']
niv = imb['indicativeNetImbalanceVolume']
buy_pri = imb['systemBuyPrice']
inter_gen = inter['Total_inter_gen']
drm = prob['drm2HourForecast']
lol_p = prob['lolp1HourForecast']

# Artificial Features

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

# 12th feature (Median of previous prices)
med = pd.DataFrame.copy(offers)
for i in range(2, len(offers['Offers'])):
    med.iloc[i, 0] = offers['Offers'].iloc[i-2:i+1].median()
med.columns = ['Med']

# Create final array of features
X = pd.concat([rene_ratio, price, price_vol, rene_gen,
               dem, niv, buy_pri, inter_gen, drm,
               lol_p, diff, ret, grad, offer_prev,
               sma, ema, rsi, fastd, BB, BB_width,
               sp_tim, med], axis=1, sort=True)

# Change names for better chart
X.rename(columns={'RENE Ratio': 'Ren_R',
                  'APXPrice': 'APXP',
                  'APXVolume': 'APXV',
                  'Total_RENE': 'Rene',
                  'indicativeNetImbalanceVolume': 'NIV',
                  'systemBuyPrice': 'Im_Pr',
                  'Total_inter_gen': 'In_gen',
                  'drm2HourForecast': 'DRM',
                  'lolp1HourForecast': 'LOLP'}, inplace=True)

# Save to csv file
X.to_csv('Features.csv')
