# =============================================================================
# Artificial_Neural_Network_grid_search_optimizer.py
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
data = pd.concat([X, offers, bids], axis=1, sort=True)

# Keep only data from 2018 (for simplicity)
data = data.loc[data.index > 2018000000, :]

# Handle missing data
data.fillna(data.median(), inplace=True)

# Predict 1h ahead instead of same time
data['Offers'] = data['Offers'].shift(-2)
data['Offers'].fillna(method='ffill', inplace=True)

# ********************* Feature Selection *********************

if 'Bids' in data.columns:
    data.drop('Bids', axis=1, inplace=True)


## Based on Correlation threshold
#def corr_thresh(df, thr):
#    """df: last column is the output"""
#    corr = df.corr()
#    features = list(corr['Offers'].iloc[:-1][abs(corr['Offers'].iloc[:-1]) < thr].index)
#    for j in df.columns[:-1]:
#        if j in features:
#            df.drop(j, axis=1, inplace=True)
#    return df
#
#
#data = corr_thresh(df=data, thr=0.3)


# Based on number of correlated features
def corr_num_of_feat(df, num):
    """df: last column is the output"""
    corr = df.corr()
    features = list(abs(corr['Offers'].iloc[:-1]).sort_values(ascending=False).index)
    features = features[num:]
    for i in df.columns[:-1]:
        if i in features:
            df.drop(i, axis=1, inplace=True)
    return df


data = corr_num_of_feat(df=data, num=15)

## Based on BE
#data = data.loc[:, ['Ren_R', 'Rene', 'TSDF', 'NIV',
#                    'Im_Pr', 'In_gen', 'DRM', 'Gr1', 'Ret', 'SMA20',
#                    'EMA10', 'RSI', 'SRSID', 'BB', 'BBw', 'Med', 'Offers']]

## Based on BE on MAE
#data = data.loc[:, ['Ren_R', 'DRM', 'LOLP', 'Gr1', 'Gr2', 'Prev', 'SMA20',
#                    'EMA10', 'RSI', 'BB', 'Spk', 'Med', 'Offers']]

## Based on BE on MAE (CV)
#data = data.loc[:, ['Ren_R', 'TSDF', 'Gr2', 'SMA20',
#                    'BB', 'BBw', 'Med', 'Offers']]

# All-in
X = data.iloc[:, :-1]
y = data['Offers']

# *************************************************************


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


# Callbacks
callbacks = [EarlyStopping(monitor='val_loss', patience=10),
             ModelCheckpoint(filepath='best_model.h5',
                             monitor='val_loss',
                             save_best_only=True,
                             mode='min', verbose=1)]

mae = np.array([])
rmse = np.array([])
mse = np.array([])
da = np.array([])
rt = np.array([])
a = 0.3
r = 2

params = {
    'optimizer': ['SGD', 'RMSprop', 'Adagrad',
                  'Adadelta', 'Adam', 'Adamax', 'Nadam']
    }

for i in params[list(params)[0]]:
    # Calculate running time
    start_time = time.time()

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

    # Build the ANN
    def build_regressor(n_neurons=15, n_layers=1, optimizer=i):
        # Initialising the ANN
        nn = Sequential()
        # Adding the input layer and the first hidden layer
        nn.add(Dense(units=n_neurons,
                     kernel_initializer='uniform',
                     bias_initializer='zeros',
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
                     round(np.sqrt(metrics.mean_squared_error(y_test,
                                                              y_pred)), r))
    mse = np.append(mse,
                    round(metrics.mean_squared_error(y_test, y_pred), r))
    da = np.append(da, round(direction_accuracy(y_test, y_pred), r))
    rt = np.append(rt, round(time.time() - start_time, r))
    print(i)


# Plots
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)

size = 150
fig = plt.figure(1, figsize=(7, 3.5), dpi=150)
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
for i in range(0, len(mae)):
    axes.scatter(i, mae[i], marker="o", s=size,
                 label=params[list(params)[0]][i])
mael = axes.scatter(0, mae[0],
                    marker="o",
                    color=plt.rcParams['axes.prop_cycle'].by_key()['color'][0],
                    s=size)
axes2 = axes.twinx()
for i in range(0, len(rmse)):
    axes2.scatter(i, rmse[i], marker="^", s=size)
rmsel = axes2.scatter(0, rmse[0],
                      marker="^",
                      color=plt.rcParams['axes.prop_cycle'].by_key()['color'][0],
                      s=size)
axes.set_xlabel('Optimizer', fontsize=16)
axes.set_ylabel('Mean Absolute Error', fontsize=16)
axes2.set_ylabel('Root Mean Squared Error', fontsize=16)
axes.set_title('Effect of Optimizer on '
               'Artificial Neural Network Performance', fontsize=18)
axes.set_xticks(list(range(0, len(mae))))
axes.set_xticklabels(list(range(1, len(mae)+1)))
leg1 = fig.legend(bbox_to_anchor=(1, 0.62),
                  bbox_transform=axes.transAxes, fontsize=16)
leg2 = fig.legend([mael, rmsel],
                  ['Mean Absolute Error', 'Root Mean Squared Error'],
                  bbox_to_anchor=(0.38, 0.7),
                  bbox_transform=axes.transAxes, fontsize=16)
axes.add_artist(leg1)
axes.grid(True)
axes.autoscale()

#fig = plt.figure(1, figsize=(7, 3.5), dpi=150)
#axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
#for i in range(0, len(params[list(params)[0]])):
#    axes.scatter(i, mae[i],
#                 marker="o", s=size, label=params[list(params)[0]][i])
#axes.set_xlabel('Method', fontsize=16)
#axes.set_ylabel('Mean Absolute Error', fontsize=16)
#axes.set_title('Effect of Optimizer Method on '
#               'Artificial Neural Network Performance', fontsize=18)
#axes.legend(loc='upper left', fontsize=16)
#axes.set_xticks(list(range(0, len(mae))))
#axes.set_xticklabels(list(range(1, len(mae)+1)))
#axes.grid(True)
#axes.autoscale()
#
#fig = plt.figure(2, figsize=(7, 3.5), dpi=150)
#axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
#for i in range(0, len(params[list(params)[0]])):
#    axes.scatter(i, rmse[i],
#                 marker="o", s=size, label=params[list(params)[0]][i])
#axes.set_xlabel('Method', fontsize=16)
#axes.set_ylabel('Root Mean Squared Error', fontsize=16)
#axes.set_title('Effect of Optimizer Method on '
#               'Artificial Neural Network Performance', fontsize=18)
#axes.legend(loc='upper left', fontsize=16)
#axes.set_xticks(list(range(0, len(rmse))))
#axes.set_xticklabels(list(range(1, len(rmse)+1)))
#axes.grid(True)
#axes.autoscale()

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
print('')
print('Best MAE: ')
print(mae.min())
print('For', list(params)[0], ':', params[list(params)[0]][np.argmin(mae)])
print('Best RMSE: ')
print(rmse.min())
print('For', list(params)[0], ':', params[list(params)[0]][np.argmin(rmse)])
