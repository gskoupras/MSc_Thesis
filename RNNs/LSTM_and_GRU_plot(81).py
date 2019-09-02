# =============================================================================
# LSTM_and_GRU_plot.py
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

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


y_test_ = pd.read_csv('LSTM_y_test.csv', header=None)
y_pred_ = pd.read_csv('LSTM_y_pred.csv', header=None)
y_pred2_ = pd.read_csv('GRU_y_pred.csv', header=None)

r = 2
print('')
print("   ******** Evaluation Metrics for the LSTM model ********    ")
print("Mean Absolute Error (test set):")
print(round(metrics.mean_absolute_error(y_test_, y_pred_), r))
print("Mean Squared Error (test set):")
print(round(metrics.mean_squared_error(y_test_, y_pred_), r))
print("Root Mean Squared Error (test set):")
print(round(np.sqrt(metrics.mean_squared_error(y_test_, y_pred_)), r))
print('Direction Accuracy:', direction_accuracy(y_test_, y_pred_))

print('')
print("   ******** Evaluation Metrics for the GRU model ********    ")
print("Mean Absolute Error (test set):")
print(round(metrics.mean_absolute_error(y_test_, y_pred2_), r))
print("Mean Squared Error (test set):")
print(round(metrics.mean_squared_error(y_test_, y_pred2_), r))
print("Root Mean Squared Error (test set):")
print(round(np.sqrt(metrics.mean_squared_error(y_test_, y_pred2_)), r))
print('Direction Accuracy:', direction_accuracy(y_test_, y_pred2_))

# Plots
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)

fig = plt.figure(1, figsize=(7, 3.5), dpi=150)
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
axes2 = axes.twinx()
axes.plot(y_test_[700:900], zorder=1, lw=3,
          color='red', label='Real Price')
axes.plot(y_pred_[700:900], zorder=2, lw=3,
          color='blue', label='LSTM')
axes.plot(y_pred2_[700:900], zorder=2, lw=3,
          color='green', label='GRU')
axes2.plot(y_pred_[700:900]-y_test_[700:900],
           zorder=0, lw=3, color='#a0b52b', label='LSTM Error')
axes2.plot(y_pred2_[700:900]-y_test_[700:900],
           zorder=0, lw=3, color='#2bc2bf', label='GRU Error')
axes.set_title('LSTM vs GRU',
               fontsize=18)
axes.set_xlabel('Day and SP', fontsize=16)
axes.set_ylabel('Offer price', fontsize=16)
axes2.set_ylabel('Residual Error', fontsize=16)
fig.legend(bbox_to_anchor=(1, 0.62),
           bbox_transform=axes.transAxes, fontsize=16)
#axes.legend(loc='best', fontsize=16)
axes.grid(True)
axes.autoscale()
