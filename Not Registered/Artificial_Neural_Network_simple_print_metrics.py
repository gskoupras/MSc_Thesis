import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.model_selection import train_test_split
from sklearn import metrics
import statsmodels.formula.api as sm
from sklearn.preprocessing import StandardScaler
import keras
from keras.models import Sequential
from keras.layers import Dense

## SGD
#keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
## ADAM
#keras.optimizers.Adam(lr=0.001, beta_1=0.9,
#                      beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

# To see options of metrics (or anything else:)
# keras.metrics. (and press Tab)

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

# Divide features and output
X = data.iloc[:, :-2]
# X = data.loc[:, ['TSDF']]
y = data['Offers']
# y = data['Bids']

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.1,
                                                    random_state=0,
                                                    shuffle=False)

# Standardization
X_test = (X_test-X_train.mean())/X_train.std()
X_train = (X_train-X_train.mean())/X_train.std()

# Calculate running time
start_time = time.time()
r = 2

# un = round((len(X.columns)+1)/2)
un = 6  # only 6 & 7 give a decent model

# Initialising the ANN
nn = Sequential()

# Adding the input layer and the first hidden layer
nn.add(Dense(units=un,
             kernel_initializer='uniform',
             activation='relu', input_dim=len(X.columns)))

# Adding more hidden layers
nn.add(Dense(units=un,
             kernel_initializer='uniform',
             activation='relu'))

nn.add(Dense(units=un,
             kernel_initializer='uniform',
             activation='relu'))

nn.add(Dense(units=un,
             kernel_initializer='uniform',
             activation='relu'))

nn.add(Dense(units=un,
             kernel_initializer='uniform',
             activation='relu'))

nn.add(Dense(units=un,
             kernel_initializer='uniform',
             activation='relu'))

# Adding the output layer
nn.add(Dense(units=1,
             kernel_initializer='uniform',
             activation='relu'))

# Check the structure of the model
# nn.summary()

# Compiling the ANN
nn.compile(optimizer='sgd',
           loss='mae',
           metrics=['mse', 'mae'])

# Fitting the ANN to the Training set
nn.fit(X_train, y_train, batch_size=20, epochs=30)

y_pred = nn.predict(X_test)
y_pred2 = nn.predict(X_train)

fig = plt.figure(2, figsize=(7, 3.5), dpi=150)
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
axes.plot(range(len(y_test)), y_test, color='red', label='Real')
axes.plot(y_pred, color='blue', label='Predicted')
axes.set_title('Artificial Neural Network (Real vs Predicted values)')
axes.set_xlabel('Day and SP')
axes.set_ylabel('Offer price')
axes.legend()
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

