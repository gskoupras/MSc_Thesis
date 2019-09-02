# =============================================================================
# Results_bar_chart.py
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

results = pd.read_csv('Results.csv', parse_dates=True,
                      index_col=0, error_bad_lines=False)

#results = results.drop(['DTR'])

mae = results.iloc[:, 0]
rmse = results.iloc[:, 1]

labels = results.index

x = np.arange(len(labels))  # the label locations
width = 0.45  # the width of the bars

# Plots
plt.rc('xtick', labelsize=18)
plt.rc('ytick', labelsize=18)

fig, ax = plt.subplots()
ax2 = ax.twinx()
rects1 = ax.bar(x - width/2, mae, width,
                label='MAE', color='#e62929', zorder=10)
rects2 = ax2.bar(x + width/2, rmse, width, label='RMSE', color='#3d4de0')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('MAE', fontsize=18)
ax2.set_ylabel('RMSE', fontsize=18)
ax.set_xlabel('Model', fontsize=18)
ax.set_title('Performance of all models', fontsize=20)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=16)
#ax.legend(loc='upper right', fontsize=13)
leg2 = fig.legend([rects1, rects2],
                  ['MAE', 'RMSE'],
                  bbox_to_anchor=(0.125, 1.01),
                  bbox_transform=ax.transAxes, fontsize=18)
ax.grid(True)
ax.autoscale()


def autolabel(rects, axi):
    """Attach a text label above each bar in *rects*, displaying its height."""
    if axi == 1:
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2 - 0.02, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=14, color='#e62929')
    else:
        for rect in rects:
            height = rect.get_height()
            ax2.annotate('{}'.format(height),
                         xy=(rect.get_x() + rect.get_width() / 2 + 0.02, height),
                         xytext=(0, 3),  # 3 points vertical offset
                         textcoords="offset points",
                         ha='center', va='bottom',
                         fontsize=14, color='#3d4de0')


autolabel(rects1, 1)
autolabel(rects2, 2)

fig.tight_layout()

plt.show()
