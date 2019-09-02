# =============================================================================
# Results.py
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

results = pd.read_csv('Results.csv', parse_dates=True,
                      index_col=0, error_bad_lines=False)

results = results.drop(['DTR'])

mae = results.iloc[:, 0]
rmse = results.iloc[:, 1]

# Plots
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)

size = 300
fig = plt.figure(1, figsize=(7, 3.5), dpi=150)
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
for i in range(0, len(results.index)):
    axes.scatter(i, mae[i], marker="o", s=size, label=results.index[i])
    if i == 3:
        axes.text(i-0.28, mae[i]-0.4, str(round(mae[i], 2)),
                  fontsize=16)
    else:
        axes.text(i-0.28, mae[i]+0.2, str(round(mae[i], 2)),
                  fontsize=16)
axes.plot(list(range(0, len(results.index))),
          min(mae)*np.ones((len(mae), 1)),
          color='black', lw=1, zorder=0)
axes.set_xlabel('Model', fontsize=16)
axes.set_ylabel('Mean Absolute Error', fontsize=16)
axes.set_title('MAE of all models', fontsize=18)
axes.set_xticks(range(0, len(results.index)))
axes.set_xticklabels(range(1, len(results.index)+1))
axes.legend(loc='upper right', fontsize=13)
axes.grid(True)
axes.autoscale()

fig = plt.figure(2, figsize=(7, 3.5), dpi=150)
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
for i in range(0, len(results.index)):
    axes.scatter(i, rmse[i], marker="o", s=size, label=results.index[i])
    if i == 0:
        axes.text(i-0.28, rmse[i]-0.28, str(round(rmse[i], 2)),
                  fontsize=16)
    else:
        axes.text(i-0.28, rmse[i]+0.12, str(round(rmse[i], 2)),
                  fontsize=16)
axes.plot(list(range(0, len(results.index))),
          min(rmse)*np.ones((len(rmse), 1)),
          color='black', lw=1, zorder=0)
axes.set_xlabel('Model', fontsize=16)
axes.set_ylabel('Root Mean Square Error', fontsize=16)
axes.set_title('RMSE of All Models', fontsize=18)
axes.set_xticks(range(0, len(results.index)))
axes.set_xticklabels(range(1, len(results.index)+1))
axes.legend(loc='upper right', fontsize=13)
axes.grid(True)
axes.autoscale()

#size = 230
#fig = plt.figure(1, figsize=(7, 3.5), dpi=150)
#axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
#for i in range(0, len(results.index)):
#    axes.scatter(i, mae[i], marker="o", s=size, label=results.index[i])
#mael = axes.scatter(7, mae[7],
#                    marker="o",
#                    color=plt.rcParams['axes.prop_cycle'].by_key()['color'][7],
#                    s=size)
#axes2 = axes.twinx()
#for i in range(0, len(results.index)):
#    axes2.scatter(i, rmse[i], marker="^", s=size)
#rmsel = axes2.scatter(7, rmse[7],
#                      marker="^",
#                      color=plt.rcParams['axes.prop_cycle'].by_key()['color'][7],
#                      s=size)
#axes.set_xlabel('Basic Method', fontsize=16)
#axes.set_ylabel('Mean Absolute Error', fontsize=16)
#axes2.set_ylabel('Root Mean Squared Error', fontsize=16)
#axes.set_title('Basic Methods Performance', fontsize=16)
#axes.set_xticks(list(range(0, len(results.index))))
#axes.set_xticklabels(list(range(1, len(results.index)+1)))
#leg1 = fig.legend(bbox_to_anchor=(0.282, 1.01),
#                  bbox_transform=axes.transAxes, fontsize=12)
#leg2 = fig.legend([mael, rmsel],
#                  ['Mean Absolute Error', 'Root Mean Squared Error'],
#                  bbox_to_anchor=(1.005, 1.01),
#                  bbox_transform=axes.transAxes, fontsize=12)
#axes.add_artist(leg1)
#axes.grid(True)
#axes.autoscale()
