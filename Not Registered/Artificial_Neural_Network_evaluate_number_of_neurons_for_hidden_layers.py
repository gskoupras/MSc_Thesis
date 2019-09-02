import pandas as pd
import numpy as np
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

neu = list(range(2, 16))
mae = np.array([])
rmse = np.array([])

for i in neu:

    print('')
    print('For {} neurons: '.format(i))
    print('')

    # Initialising the ANN
    nn = Sequential()

    # Adding the input layer and the first hidden layer
    nn.add(Dense(units=i,
                 kernel_initializer='uniform',
                 activation='relu', input_dim=len(X.columns)))

    # Adding more hidden layers
    nn.add(Dense(units=i,
                 kernel_initializer='uniform',
                 activation='relu'))

    nn.add(Dense(units=i,
                 kernel_initializer='uniform',
                 activation='relu'))

    nn.add(Dense(units=i,
                 kernel_initializer='uniform',
                 activation='relu'))

    nn.add(Dense(units=i,
                 kernel_initializer='uniform',
                 activation='relu'))

    nn.add(Dense(units=i,
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
    nn.fit(X_train, y_train, batch_size=100, epochs=20)

    y_pred = nn.predict(X_test)
    mae = np.append(mae, metrics.mean_absolute_error(y_test, y_pred))
    rmse = np.append(rmse,
                     np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

size = 50
fig = plt.figure(1, figsize=(7, 3.5), dpi=150)
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
axes.plot(mae,
          '-', lw=1.5, color='red', label='Mean Absolute Error')
axes.scatter(mae.argmin(),
             min(mae), marker="o", color='green',
             s=size, label='Best', zorder=3)
axes.set_xlabel('Number of neurons on the hidden layer')
axes.set_ylabel('Metrics')
axes.set_title('Effect of number of neurons on the hidden layer on '
               'Artificial Neural Network Performance')
axes.set_xticks(list(range(0, len(rmse), 2)))
axes.set_xticklabels(list(range(neu[0], len(rmse)+neu[0], 2)))
axes.legend(loc='best')
axes.grid(True)
axes.autoscale()
# fig.savefig('Effect of number of neurons on the hidden layer on '
#               'Artificial Neural Network Performance,
#            bbox_inches='tight', dpi=800)

fig = plt.figure(2, figsize=(7, 3.5), dpi=150)
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
axes.plot(rmse,
          '-', lw=1.5, color='blue', label='Root Mean Squared Error')
axes.scatter(rmse.argmin(),
             min(rmse), marker="o", color='green',
             s=size, label='Best', zorder=3)
axes.set_xlabel('Number of neurons on the hidden layer')
axes.set_ylabel('Metrics')
axes.set_title('Effect of number of neurons on the hidden layer on '
               'Artificial Neural Network Performance')
axes.set_xticks(list(range(0, len(rmse), 2)))
axes.set_xticklabels(list(range(neu[0], len(rmse)+neu[0], 2)))
axes.legend(loc='best')
axes.grid(True)
axes.autoscale()
# fig.savefig('Effect of number of neurons on the hidden layer on '
#               'Artificial Neural Network Performance',
#            bbox_inches='tight', dpi=800)
