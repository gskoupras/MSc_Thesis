# =============================================================================
# GRU_Correlation_thresh_eval.py
# =============================================================================
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, GRU, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from numpy.random import seed
from tensorflow import set_random_seed
from keras.layers.advanced_activations import LeakyReLU

# To get the same results every time
seed(1)
set_random_seed(1)

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
data = data.loc[data.index > 2018000000, :]

# Handle missing data
data.fillna(data.median(), inplace=True)

# Predict 1h ahead instead of same time
data['Offers'] = data['Offers'].shift(-2)
data['Offers'].fillna(method='ffill', inplace=True)


# Function that calculates the direction accuracy
# Accepts both row and column vectors
def direction_accuracy(real, pred):
    real = pd.DataFrame(real)
    pred = pd.DataFrame(pred, index=real.index)
    if real.shape[0] == 1:
        real = real.transpose()
    if pred.shape[0] == 1:
        pred = pred.transpose()
    real_diff = real.shift(-2)-real
    real_diff = real_diff.shift(2)
    pred_diff = pred.shift(-2)-real.values
    pred_diff = pred_diff.shift(2)
    real_diff.fillna(real_diff.median(), inplace=True)
    pred_diff.fillna(pred_diff.median(), inplace=True)
    true = 0
    false = 0
    for i in range(0, len(real_diff)):
        if (real_diff.iloc[i, 0] > 0) and (pred_diff.iloc[i, 0] > 0):
            true = true + 1
        elif (real_diff.iloc[i, 0] == 0) and (pred_diff.iloc[i, 0] == 0):
            true = true + 1
        elif (real_diff.iloc[i, 0] < 0) and (pred_diff.iloc[i, 0] < 0):
            true = true + 1
        else:
            false = false + 1
    acc = true / (true + false)
    return acc


# 1sh Correlation Function: Threshold
def corr_thresh(df, thr=0.1):
    """df: last column is the output"""
    corr = df.corr()
    features = list(corr['Offers'].iloc[:-1][abs(corr['Offers'].iloc[:-1]) < thr].index)
    for j in df.columns[:-1]:
        if j in features:
            df.drop(j, axis=1, inplace=True)
    return df


# Function that creates the 3D array that the GRU needs
def build_timeseries(mat, y_col_index, time_steps):

    io_pairs = mat.shape[0] - time_steps
    features = mat.shape[1]
    x = np.zeros((io_pairs, time_steps, features))
    y = np.zeros((io_pairs,))

    for i in range(io_pairs):
        x[i] = mat[i:time_steps+i]
        y[i] = mat[time_steps+i, y_col_index]
    print("Length of time-series i/o", x.shape, y.shape)
    return x, y


# Function that trims the dataset to be divisible by the batch size
def trim_dataset(mat, batch_size):
    no_of_rows_drop = mat.shape[0] % batch_size
    if(no_of_rows_drop > 0):
        return mat[:-no_of_rows_drop]
    else:
        return mat


time_st = 96  # SPs to look back to predict the price in next day
batch_s = 100  # Number of time steps checked before weight update
epoch = 5000  # Number of epochs
unit = 100  # Number of neurons in the output of a hidden layer
r = 2  # For rounding
a = 0.1  # For LeakyReLU

max_thresh = 6
mae = np.array([])
rmse = np.array([])
mse = np.array([])
da = np.array([])
rt = np.array([])
for i in range(0, max_thresh):

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
    data = data.loc[data.index > 2018000000, :]

    # Handle missing data
    data.fillna(data.median(), inplace=True)

    # Predict 1h ahead instead of same time
    data['Offers'] = data['Offers'].shift(-2)
    data['Offers'].fillna(method='ffill', inplace=True)

    print(data.shape[1])
    data_loop = corr_thresh(df=data, thr=i/10)
    print(data.shape[1], data_loop.shape[1])
    X = data_loop.iloc[:, :-1]
    y = data_loop['Offers']

    df_train, df_test = train_test_split(data, test_size=0.2,
                                         random_state=0, shuffle=False)

    # scale the feature MinMax, build array
    x = df_train.loc[:, :].values
    min_max_scaler = MinMaxScaler()
    x_train = min_max_scaler.fit_transform(x)
    x_test = min_max_scaler.transform(df_test.loc[:, :])

    y_col = data.columns.get_loc("Offers")  # Output column

    # Create training set
    X_train, y_train = build_timeseries(x_train, y_col, time_st)
    X_train = trim_dataset(X_train, batch_s)
    y_train = trim_dataset(y_train, batch_s)

    # Create validation and test set
    X_temp, y_temp = build_timeseries(x_test, y_col, time_st)
    X_val, X_test_t = np.split(trim_dataset(X_temp, batch_s), 2)
    y_val, y_test_t = np.split(trim_dataset(y_temp, batch_s), 2)

    # Calculate running time
    start_time = time.time()

    # Build the model
    gru = Sequential()
    gru.add(GRU(units=unit, batch_input_shape=(batch_s, time_st,
                X_train.shape[2]), stateful=True,
                kernel_initializer='random_uniform'))
    gru.add(Dropout(0.3))
    gru.add(Dense(unit, activation='linear'))
    gru.add(LeakyReLU(alpha=a))
    gru.add(Dense(1, activation='linear'))
    gru.compile(loss='mae', optimizer='Nadam',
                metrics=['mse', 'mae'])

    # Callbacks
    callbacks = [EarlyStopping(monitor='val_loss', patience=10),
                 ModelCheckpoint(filepath='best_model.h5',
                                 monitor='val_loss',
                                 save_best_only=True,
                                 mode='min', verbose=1)]

    hist = gru.fit(X_train, y_train, epochs=epoch, verbose=1,
                   batch_size=batch_s,
                   shuffle=False, callbacks=callbacks,
                   validation_data=(trim_dataset(X_val, batch_s),
                                    trim_dataset(y_val, batch_s)))

    gru_best = load_model('best_model.h5')

    y_pred_t = gru_best.predict(trim_dataset(X_test_t, batch_s),
                            batch_size=batch_s)
    y_pred_t = y_pred_t.flatten()
    y_test_t = trim_dataset(y_test_t, batch_s)

    # Cannot use min_max_scaler.inverse_transform(y_pred) since this refers to
    # all the features (not only the closing price)
    y_pred = (y_pred_t * min_max_scaler.data_range_[y_col]) + min_max_scaler.data_min_[y_col]
    y_test = (y_test_t * min_max_scaler.data_range_[y_col]) + min_max_scaler.data_min_[y_col]

    mae = np.append(mae,
                    round(metrics.mean_absolute_error(y_test, y_pred), r))
    rmse = np.append(rmse,
                     round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)), r))
    mse = np.append(mse,
                    round(metrics.mean_squared_error(y_test, y_pred), r))
    da = np.append(da, round(direction_accuracy(y_test, y_pred), r))
    rt = np.append(rt, round(time.time() - start_time, r))

# Plots
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)

size = 150
fig = plt.figure(1, figsize=(7, 3.5), dpi=150)
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
axes2 = axes.twinx()
for i in range(0, max_thresh):
    axes.scatter(list(range(0, max_thresh))[i-1], mae[i-1], marker="o",
                 color='red', s=size, zorder=1)
axes.plot(range(0, max_thresh), mae, color='red',
          lw=2, zorder=1, label='Mean Absolute Error')
axes.plot(range(0, max_thresh), min(mae)*np.ones((max_thresh, 1)),
          color='black', lw=1, zorder=0)
axes.text(np.argmin(mae)+0.15, min(mae)+0.01, str(min(mae)), fontsize=16)
for i in range(0, max_thresh):
    axes2.scatter(list(range(0, max_thresh))[i-1], rmse[i-1], marker="o",
                  color='blue', s=size, zorder=1)
axes2.plot(range(0, max_thresh), rmse, color='blue',
           lw=2, zorder=0, label='Root Mean Squared Error')
axes2.text(np.argmin(rmse)-0.2, min(rmse)+0.04, str(min(rmse)), fontsize=16)
axes.set_xlabel('Correlation Threshold Considered', fontsize=16)
axes.set_ylabel('Mean Absolute Error', fontsize=16)
axes2.set_ylabel('Root Mean Squared Error', fontsize=16)
axes.set_title('GRU - Correlation Threshold Effect', fontsize=18)
axes.set_xticks(list(range(0, max_thresh)))
axes.set_xticklabels(list(np.around(np.arange(0, max_thresh/10, 0.1), 1)))
fig.legend(bbox_to_anchor=(1, 0.7),
           bbox_transform=axes.transAxes, fontsize=16)
axes.grid(True)
axes.autoscale()

size = 150
fig = plt.figure(2, figsize=(7, 3.5), dpi=150)
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
axes2 = axes.twinx()
for i in range(0, max_thresh):
    axes.scatter(list(range(0, max_thresh))[i-1], rt[i-1], marker="o",
                 color='red', s=size, zorder=1)
axes.plot(range(0, max_thresh), rt, color='red', lw=2,
          zorder=0, label='Running Time')
for i in range(0, max_thresh):
    axes2.scatter(list(range(0, max_thresh))[i-1], da[i-1], marker="o",
                  color='blue', s=size, zorder=1)
axes2.plot(range(0, max_thresh), da, color='blue',
           lw=2, zorder=0, label='Direction Accuracy')
axes.set_xlabel('Correlation Threshold considered', fontsize=16)
axes.set_ylabel('Running Time (s)', fontsize=16)
axes2.set_ylabel('Direction Accuracy', fontsize=16)
axes.set_title('Correlation Threshold effect', fontsize=18)
axes.set_xticks(list(range(0, max_thresh)))
axes.set_xticklabels(list(np.around(np.arange(0, max_thresh/10, 0.1), 1)))
fig.legend(bbox_to_anchor=(1, 1),
           bbox_transform=axes.transAxes, fontsize=16)
axes.grid(True)
axes.autoscale()

print('')
print("   ******** Evaluation Metrics ********    ")
print("Mean Absolute Error:")
print(mae)
print("Mean Squared Error:")
print(mse)
print("Root Mean Squared Error:")
print(rmse)
print('Direction Accuracy:', da)
print('Running Time:', rt)