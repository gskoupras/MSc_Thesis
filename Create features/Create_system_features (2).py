# =============================================================================
# Create_system_features.py
# =============================================================================
import pandas as pd

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

# Create final array of features
X = pd.concat([rene_ratio, price, price_vol, rene_gen,
               dem, niv, buy_pri, inter_gen, drm,
               lol_p], axis=1, sort=True)

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
X.to_csv('System Features.csv')