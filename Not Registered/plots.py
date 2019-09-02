import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import statsmodels.formula.api as sm

offers = pd.read_csv('Offers.csv', parse_dates=True, index_col=0)
bids = pd.read_csv('Bids.csv', parse_dates=True, index_col=0)


offers.isnull().sum()
print(bids.loc[2016110339])
print(bids.loc[2016110340])
print(len(bids))
bids = bids[bids > -250]
offers = offers[offers < 2000]
print(len(bids))
print(bids.loc[2016110339])
print(bids.loc[2016110340])
#offers.dropna(inplace=True)
offers.interpolate(method='linear', inplace=True)
bids.dropna(inplace=True)

plt.figure(1)
plt.plot(range(len(offers)), offers['Offers'])
plt.ylabel('Offer Price (GBP/MWh)')
plt.title('Highest Offers Accepted')
plt.xlabel('Date and Settlement Period')
plt.legend(loc='best')
plt.grid(True)
plt.autoscale()
plt.tight_layout()
plt.savefig("Offers.png")
plt.show()

plt.figure(2)
plt.plot(range(len(bids)), bids['Bids'])
plt.ylabel('Bid Price (GBP/MWh)')
plt.title('Lowest Bids Accepted')
plt.xlabel('Date and Settlement Period')
plt.legend(loc='best')
plt.grid(True)
plt.autoscale()
plt.tight_layout()
plt.savefig("Bids.png")
plt.show()
