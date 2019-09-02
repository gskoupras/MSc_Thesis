# =============================================================================
# produce_output_data.py
# =============================================================================

import pickle
import pandas as pd
import numpy as np

# Open the first pickle object (first month) and put it on a variable (data)
pickle_off = open("1.pkl", "rb")
data = pickle.load(pickle_off)

# Open the rest 35 pickle objects and concatenate in data
for i in range(2, 37):
    pickle_off = open("{}.pkl".format(i), "rb")
    data = pd.concat([data, pickle.load(pickle_off)])

# Create numpy array for the highest offer and bid
offers = np.array([])
bids = np.array([])

# Extract the highest offer price and lowest bid price
for i in range(len(data)):
    offers = np.append(offers,
                       max(data.iloc[i][1]['offerPrice'].astype(float)))
    bids = np.append(bids, min(data.iloc[i][0]['bidPrice'].astype(float)))

# Transform to DataFrame
offers = pd.DataFrame(offers, index=data.index, columns=['Offers'])
bids = pd.DataFrame(bids, index=data.index, columns=['Bids'])

# Save to csv file
offers.to_csv('Offers.csv')
bids.to_csv('Bids.csv')
