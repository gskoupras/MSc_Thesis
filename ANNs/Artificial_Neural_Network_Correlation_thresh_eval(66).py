# =============================================================================
# Artificial Neural Networks_Correlation_thresh_eval.py
# =============================================================================
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from numpy.random import seed
from tensorflow import set_random_seed
from keras.layers.advanced_activations import LeakyReLU

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


r = 2
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

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=0,
                                                        shuffle=False)

    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test,
                                                    test_size=0.5,
                                                    random_state=0,
                                                    shuffle=False)

    # Standardization
    X_test = (X_test-X_train.mean())/X_train.std()
    X_val = (X_val-X_train.mean())/X_train.std()
    X_train = (X_train-X_train.mean())/X_train.std()

    # Calculate running time
    start_time = time.time()
    r = 2
    a = 0.1


    def build_regressor(optimizer='Nadam', n_neurons=20, n_layers=2):
        # Initialising the ANN
        nn = Sequential()
        # Adding the input layer and the first hidden layer
        nn.add(Dense(units=n_neurons,
                     kernel_initializer='uniform', bias_initializer='zeros',
                     activation='linear', input_dim=len(X.columns)))
        nn.add(LeakyReLU(alpha=a))
        for i in range(0, n_layers):
            nn.add(Dense(units=n_neurons,
                         kernel_initializer='uniform',
                         bias_initializer='zeros',
                         activation='linear'))
            nn.add(LeakyReLU(alpha=a))
        nn.add(Dense(units=1,
                     kernel_initializer='uniform',
                     bias_initializer='zeros',
                     activation='linear'))
        nn.compile(optimizer=optimizer,
                   loss='mae',
                   metrics=['mse', 'mae'])
        return nn
    
    
    # Callbacks
    callbacks = [EarlyStopping(monitor='val_loss', patience=20),
                 ModelCheckpoint(filepath='best_model.h5',
                                 monitor='val_loss',
                                 save_best_only=True,
                                 mode='min', verbose=1)]
    
    nn = KerasRegressor(build_fn=build_regressor)
    hist = nn.fit(X_train, y_train, callbacks=callbacks,
                  batch_size=10, epochs=100,
                  validation_data=(X_val, y_val))
    
    nn_best = load_model('best_model.h5')
    
    y_pred = nn_best.predict(X_test)
    y_pred2 = nn_best.predict(X_train)
    y_pred3 = nn_best.predict(X_val)

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
axes.text(np.argmin(mae)-0.18, min(mae)+0.2, str(min(mae)), fontsize=16)
for i in range(0, max_thresh):
    axes2.scatter(list(range(0, max_thresh))[i-1], rmse[i-1], marker="o",
                  color='blue', s=size, zorder=1)
axes2.plot(range(0, max_thresh), rmse, color='blue',
           lw=2, zorder=0, label='Root Mean Squared Error')
axes.set_xlabel('Correlation Threshold considered', fontsize=16)
axes.set_ylabel('Mean Absolute Error', fontsize=16)
axes2.set_ylabel('Root Mean Squared Error', fontsize=16)
axes.set_title('Correlation Threshold effect', fontsize=18)
axes.set_xticks(list(range(0, max_thresh)))
axes.set_xticklabels(list(np.around(np.arange(0, max_thresh/10, 0.1), 1)))
fig.legend(bbox_to_anchor=(0.95, 1),
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
