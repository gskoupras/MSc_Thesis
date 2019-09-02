import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from numpy.random import seed
from tensorflow import set_random_seed

# To get the same results every time
seed(1)
set_random_seed(2)


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

## Divide features and output
#X = data.iloc[:, :-2]
## X = data.loc[:, ['TSDF']]
#y = data['Offers']
## y = data['Bids']

df_train, df_test = train_test_split(data, test_size=0.1,
                                     random_state=0, shuffle=False)

# scale the feature MinMax, build array
x = df_train.loc[:, :].values
min_max_scaler = MinMaxScaler()
x_train = min_max_scaler.fit_transform(x)
x_test = min_max_scaler.transform(df_test.loc[:, :])


# Function that creates the 3D array that LSTM needs
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


# Function that calculates the direction accuracy
# Accepts both row and column vectors
def direction_accuracy(real, pred):
    real=pd.DataFrame(real)
    pred=pd.DataFrame(pred, index=real.index)
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
        if (real_diff.iloc[i,0] > 0) and (pred_diff.iloc[i,0] > 0):
            true = true + 1
        elif (real_diff.iloc[i,0] == 0) and (pred_diff.iloc[i,0] == 0):
            true = true + 1
        elif (real_diff.iloc[i,0] < 0) and (pred_diff.iloc[i,0] < 0):
            true = true + 1
        else:
            false = false + 1
    acc = true / (true + false)
    return acc

# Calculate running time
start_time = time.time()

r = 2
time_st = 100  # Days to look back to predict the price in next day
y_col = data.columns.get_loc("Offers")  # Output column
batch_s = 2000  # Number of time steps checked before weight update
epoch = 10  # Number of epochs
unit = 100  # Number of neurons in the output of a hidden layer
r=2

# Create training set
X_train, y_train = build_timeseries(x_train, y_col, time_st)
X_train = trim_dataset(X_train, batch_s)
y_train = trim_dataset(y_train, batch_s)

# Create validation and test set
X_temp, y_temp = build_timeseries(x_test, y_col, time_st)
X_val, X_test_t = np.split(trim_dataset(X_temp, batch_s), 2)
y_val, y_test_t = np.split(trim_dataset(y_temp, batch_s), 2)

labels = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
mae = np.array([])
rmse = np.array([])
mse = np.array([])
da = np.array([])
rt = np.array([])

for i in labels:
    
    start_time = time.time()
    # Build the LSTM model
    lstm = Sequential()
    lstm.add(LSTM(units=unit, batch_input_shape=(batch_s, time_st,
             X_train.shape[2]), stateful=True,
             kernel_initializer='random_uniform'))
    lstm.add(Dropout(0.5))
    lstm.add(Dense(20, activation='relu'))
    lstm.add(Dense(1, activation='linear'))
    lstm.compile(loss='mean_squared_error', optimizer=i,
                 metrics=['mse', 'mae'])
    
    # Callbacks
    callbacks = [EarlyStopping(monitor='val_loss', patience=5),
                 ModelCheckpoint(filepath='best_model.h5',
                                 monitor='val_loss', save_best_only=True)]
    
    lstm.fit(X_train, y_train, epochs=epoch, verbose=1, batch_size=batch_s,
             shuffle=False, callbacks=callbacks,
             validation_data=(trim_dataset(X_val,
                                           batch_s), trim_dataset(y_val, batch_s)))
    
    y_pred_t = lstm.predict(trim_dataset(X_test_t, batch_s), batch_size=batch_s)
    y_pred_t = y_pred_t.flatten()
    y_test_t = trim_dataset(y_test_t, batch_s)
    
    # Cannot use min_max_scaler.inverse_transform(y_pred) since this refers to
    # all the features (not only the closing price)
    y_pred = (y_pred_t * min_max_scaler.data_range_[y_col]) + min_max_scaler.data_min_[y_col]
    y_test = (y_test_t * min_max_scaler.data_range_[y_col]) + min_max_scaler.data_min_[y_col]
    
    mae = np.append(mae, round(metrics.mean_absolute_error(y_test,
                                                           y_pred), r))
    rmse = np.append(rmse,
                     round(np.sqrt(metrics.mean_squared_error(y_test,
                                                              y_pred)), r))
    mse = np.append(mse, round(metrics.mean_squared_error(y_test,
                                                          y_pred), r))
    da = np.append(da, round(direction_accuracy(y_test, y_pred), r))
    rt = np.append(rt, round(time.time() - start_time, r))

size = 150
fig = plt.figure(1, figsize=(7, 3.5), dpi=150)
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
for i in range(0, len(labels)):
    axes.scatter(i, mae[i], marker="o", s=size, label=labels[i])
axes.set_xlabel('Basic Method')
axes.set_ylabel('Mean Absolute Error')
axes.set_title('Basic Methods Performance')
axes.set_xticks(list(range(0, len(labels))))
axes.set_xticklabels(list(range(1, len(labels)+1)))
axes.legend(loc='best')
axes.grid(True)
axes.autoscale()

fig = plt.figure(2, figsize=(7, 3.5), dpi=150)
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
for i in range(0, len(labels)):
    axes.scatter(i, rmse[i], marker="o", s=size, label=labels[i])
axes.set_xlabel('Basic Method')
axes.set_ylabel('Root Mean Squared Error')
axes.set_title('Basic Methods Performance')
axes.set_xticks(list(range(0, len(labels))))
axes.set_xticklabels(list(range(1, len(labels)+1)))
axes.legend(loc='best')
axes.grid(True)
axes.autoscale()

fig = plt.figure(3, figsize=(7, 3.5), dpi=150)
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
for i in range(0, len(labels)):
    axes.scatter(i, rt[i], marker="o", s=size, label=labels[i])
axes.set_xlabel('Basic Method')
axes.set_ylabel('Running Time (s)')
axes.set_title('Basic Methods Performance')
axes.set_xticks(list(range(0, len(labels))))
axes.set_xticklabels(list(range(1, len(labels)+1)))
axes.legend(loc='best')
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
print('Running time:', rt)

