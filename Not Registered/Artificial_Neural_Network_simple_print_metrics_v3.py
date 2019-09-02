# =============================================================================
# Neural_Network.py
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
#data = corr_thresh(df=data, thr=0.2)


## Based on number of correlated features
#def corr_num_of_feat(df, num):
#    """df: last column is the output"""
#    corr = df.corr()
#    features = list(abs(corr['Offers'].iloc[:-1]).sort_values(ascending=False).index)
#    features = features[num:]
#    for i in df.columns[:-1]:
#        if i in features:
#            df.drop(i, axis=1, inplace=True)
#    return df
#
#
#data = corr_num_of_feat(df=data, num=5)

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

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.1,
                                                    random_state=0,
                                                    shuffle=False)

X_test, X_val, y_test, y_val = train_test_split(X_test, y_test,
                                                test_size=0.5,
                                                random_state=0,
                                                shuffle=False)

# Standardization
#y_train_old = y_train  # Keep the old y_train for the inverse transformation
X_test = (X_test-X_train.mean())/X_train.std()
X_val = (X_val-X_train.mean())/X_train.std()
X_train = (X_train-X_train.mean())/X_train.std()
#y_test = (y_test-y_train.mean())/y_train.std()
#y_val = (y_val-y_train.mean())/y_train.std()
#y_train = (y_train-y_train.mean())/y_train.std()

# Calculate running time
start_time = time.time()
r = 2


def build_regressor(optimizer='sgd', n_neurons=10, n_layers=3):
    # Initialising the ANN
    nn = Sequential()
    # Adding the input layer and the first hidden layer
    nn.add(Dense(units=n_neurons,
                 kernel_initializer='uniform',
                 activation='relu', input_dim=len(X.columns)))
    for i in range(0, n_layers):
        nn.add(Dense(units=n_neurons,
                     kernel_initializer='uniform',
                     activation='relu'))
    nn.add(Dense(units=1,
                 kernel_initializer='uniform',
                 activation='linear'))
    nn.compile(optimizer=optimizer,
               loss='mae',
               metrics=['mse', 'mae'])
    return nn


# Callbacks
callbacks = [EarlyStopping(monitor='val_loss', patience=10),
             ModelCheckpoint(filepath='best_model.h5',
                             monitor='val_loss', save_best_only=True)]

nn = KerasRegressor(build_fn=build_regressor, batch_size=10, epochs=100)
nn.fit(X_train, y_train, callbacks=callbacks, validation_data=(X_val, y_val))

y_pred = nn.predict(X_test)
y_pred2 = nn.predict(X_train)

#y_pred = (y_pred * y_train_old.std()) + y_train_old.mean()
#y_pred2 = (y_pred2 * y_train_old.std()) + y_train_old.mean()
#y_test = (y_test * y_train_old.std()) + y_train_old.mean()

# Plots
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)

fig = plt.figure(2, figsize=(7, 3.5), dpi=150)
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
axes.plot(range(len(y_test[700:900])), y_test[700:900], zorder=1, lw=3,
          color='red', label='Real Price')
axes.plot(y_pred[700:900], zorder=2, lw=3,
          color='blue', label='Predicted Price')
axes.plot(y_pred[700:900]-y_test[700:900].values, zorder=0, lw=3,
          color='green', label='Residual Error')
axes.set_title('Artificial Neural Network (Real vs Predicted values)',
               fontsize=18)
axes.set_xlabel('Day and SP', fontsize=16)
axes.set_ylabel('Offer price and Residual Error', fontsize=16)
axes.legend(loc='best', fontsize=16)
axes.grid(True)
axes.autoscale()

rt = round(time.time() - start_time, r)
da = round(direction_accuracy(y_test, y_pred), r)

print('')
print("   ******** Evaluation Metrics for an ANN ********    ")
print("Mean Absolute Error (test set and training set):")
print(round(metrics.mean_absolute_error(y_test, y_pred), r),
      round(metrics.mean_absolute_error(y_train, y_pred2), r))
print("Mean Squared Error (test set and training set):")
print(round(metrics.mean_squared_error(y_test, y_pred), r),
      round(metrics.mean_squared_error(y_train, y_pred2), r))
print("Root Mean Squared Error (test set and training set):")
print(round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)), r),
      round(np.sqrt(metrics.mean_squared_error(y_train, y_pred2)), r))
print('Direction Accuracy:', da)
print('Running time:', rt)

