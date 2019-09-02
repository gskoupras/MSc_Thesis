# =============================================================================
# Multiple_Linear_Regression_test_size.py
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Read data from csv files
offers = pd.read_csv('Offers.csv', parse_dates=True, index_col=0)
bids = pd.read_csv('Bids.csv', parse_dates=True, index_col=0)
X = pd.read_csv('Features.csv', parse_dates=True, index_col=0)

# Get rid of extreme values
offers = offers[offers < 2000]
bids = bids[bids > -250]

# Connect all together
data = pd.concat([X, offers, bids], axis=1, sort=True)

# Keep only data from 2018 (for simplicity)
data = data.loc[data.index > 2018000000, :]

# Predict 1h ahead instead of same time
data['Offers'] = data['Offers'].shift(-2)
data['Offers'].fillna(method='ffill', inplace=True)

# #Handling missing values:
df_drop = data.dropna()

df_inter = data.interpolate(method='linear')
df_inter.dropna(inplace=True)

df_forw = data.fillna(method='pad')
df_forw.dropna(inplace=True)

df_back = data.fillna(method='bfill')
df_back.dropna(inplace=True)

df_mean = data.fillna(data.mean())

df_median = data.fillna(data.median())

# Divide features and output
X = data.iloc[:, :-2]
y = data['Offers']

X_drop = df_drop.iloc[:, :-2]
y_drop = df_drop['Offers']
X_inter = df_inter.iloc[:, :-2]
y_inter = df_inter['Offers']
X_forw = df_forw.iloc[:, :-2]
y_forw = df_forw['Offers']
X_mean = df_mean.iloc[:, :-2]
y_mean = df_mean['Offers']
X_back = df_back.iloc[:, :-2]
y_back = df_back['Offers']
X_median = df_median.iloc[:, :-2]
y_median = df_median['Offers']

# Different test splits that I want to evaluate
splits = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

# Initialise arrays:
mae_drop = np.array([])
rmse_drop = np.array([])
mae_inter = np.array([])
rmse_inter = np.array([])
mae_forw = np.array([])
rmse_forw = np.array([])
mae_mean = np.array([])
rmse_mean = np.array([])
mae_back = np.array([])
rmse_back = np.array([])
mae_median = np.array([])
rmse_median = np.array([])

for i in splits:
    #Train-test split
    X_train_drop, X_test_drop, y_train_drop, y_test_drop = train_test_split(X_drop, y_drop, test_size = i, shuffle=False)
    X_train_inter, X_test_inter, y_train_inter, y_test_inter = train_test_split(X_inter, y_inter, test_size = i, shuffle=False)  
    X_train_forw, X_test_forw, y_train_forw, y_test_forw = train_test_split(X_forw, y_forw, test_size = i, shuffle=False)
    X_train_mean, X_test_mean, y_train_mean, y_test_mean = train_test_split(X_mean, y_mean, test_size = i, shuffle=False)
    X_train_back, X_test_back, y_train_back, y_test_back = train_test_split(X_back, y_back, test_size = i, shuffle=False)
    X_train_median, X_test_median, y_train_median, y_test_median = train_test_split(X_median, y_median, test_size = i, shuffle=False)

    # Create an instance of the class Linear Regression
    lin1 = LinearRegression()
    lin2 = LinearRegression()
    lin3 = LinearRegression()
    lin4 = LinearRegression()
    lin5 = LinearRegression()
    lin6 = LinearRegression()

    # Fit the model to the training data (learn the coefficients)
    lin1.fit(X_train_drop, y_train_drop)
    lin2.fit(X_train_inter, y_train_inter)
    lin3.fit(X_train_forw, y_train_forw)
    lin4.fit(X_train_mean, y_train_mean)
    lin5.fit(X_train_back, y_train_back)
    lin6.fit(X_train_median, y_train_median)
    
    # Make predictions on the testing set
    y_pred_drop = lin1.predict(X_test_drop)
    y_pred_inter = lin2.predict(X_test_inter)
    y_pred_forw = lin3.predict(X_test_forw)
    y_pred_mean = lin4.predict(X_test_mean)
    y_pred_back = lin5.predict(X_test_back)
    y_pred_median = lin6.predict(X_test_median)
    
    mae_drop=np.append(mae_drop, metrics.mean_absolute_error(y_test_drop, y_pred_drop))
    rmse_drop=np.append(rmse_drop, np.sqrt(metrics.mean_squared_error(y_test_drop, y_pred_drop)))
    mae_inter=np.append(mae_inter, metrics.mean_absolute_error(y_test_inter, y_pred_inter))
    rmse_inter=np.append(rmse_inter, np.sqrt(metrics.mean_squared_error(y_test_inter, y_pred_inter)))
    mae_forw=np.append(mae_forw, metrics.mean_absolute_error(y_test_forw, y_pred_forw))
    rmse_forw=np.append(rmse_forw, np.sqrt(metrics.mean_squared_error(y_test_forw, y_pred_forw)))
    mae_mean=np.append(mae_mean, metrics.mean_absolute_error(y_test_mean, y_pred_mean))
    rmse_mean=np.append(rmse_mean, np.sqrt(metrics.mean_squared_error(y_test_mean, y_pred_mean)))
    mae_back=np.append(mae_back, metrics.mean_absolute_error(y_test_back, y_pred_back))
    rmse_back=np.append(rmse_back, np.sqrt(metrics.mean_squared_error(y_test_back, y_pred_back)))
    mae_median=np.append(mae_median, metrics.mean_absolute_error(y_test_median, y_pred_median))
    rmse_median=np.append(rmse_median, np.sqrt(metrics.mean_squared_error(y_test_median, y_pred_median)))

# Plots
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)

splits = [i * 100 for i in splits]
li_wi = 3
plt.figure(1)
plt.plot(splits, mae_drop, '-', lw=li_wi, label="Drop Values")
plt.plot(splits, mae_inter, '-', lw=li_wi, label="Linear Interpolation")
plt.plot(splits, mae_forw, '-', lw=li_wi, label="Forward Propagation")
plt.plot(splits, mae_mean, '-', lw=li_wi, label="Mean Value")
plt.plot(splits, mae_back, '-', lw=li_wi, label="Backward Propagation")
plt.plot(splits, mae_median, '-', lw=li_wi, label="Median Value")
plt.ylabel('Mean Absolute Error')
plt.title('Linear Regression - Test Set Size')
plt.xlabel('Test size (%)')
plt.legend(loc='best')
plt.grid(True)
plt.autoscale()
plt.show()

plt.figure(2)
plt.plot(splits, rmse_drop, '-', lw=li_wi, label="Drop Values")
plt.plot(splits, rmse_inter, '-', lw=li_wi, label="Linear Interpolation")
plt.plot(splits, rmse_forw, '-', lw=li_wi, label="Forward Propagation")
plt.plot(splits, rmse_mean, '-', lw=li_wi, label="Mean Value")
plt.plot(splits, rmse_back, '-', lw=li_wi, label="Backward Propagation")
plt.plot(splits, rmse_median, '-', lw=li_wi, label="Median Value")
plt.ylabel('Root Mean Squared Error')
plt.title('Linear Regression - Test Set Size')
plt.xlabel('Test size (%)')
plt.legend(loc='best')
plt.grid(True)
plt.autoscale()
plt.show()

s = 50
fig = plt.figure(3, figsize=(7, 3.5), dpi=150)
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
axes2 = axes.twinx()
axes.plot(splits, mae_median,
          '-', lw=2, color='red',
          label='Mean Absolute Error', zorder=10)
axes.scatter(splits, mae_median,
             marker="o",
             color='red', s=s, zorder=10)
axes2.plot(splits, rmse_median,
           '-', lw=2, color='blue',
           label='Root Mean Squared Error', zorder=10)
axes2.scatter(splits, rmse_median,
              marker="o",
              color='blue', s=s, zorder=10)
axes.set_xlabel('Test Set Size (%)', fontsize=16)
axes.set_ylabel('Mean Absolute Error', fontsize=16)
axes2.set_ylabel('Root Mean Squared Error', fontsize=16)
axes.set_title('Linear Regression - Test Set Size', fontsize=18)
axes.set_xticks(splits)
fig.legend(bbox_to_anchor=(0.8, 0.5),
           bbox_transform=axes.transAxes, fontsize=16)
axes.grid(True, zorder=10)
axes.autoscale()
