# Import necessary libraries.
import os
import numpy as np
import pandas as pd
import pickle
from collections import defaultdict
import matplotlib.pyplot as plt

plt.style.use('ggplot')

# Define paths, functions and variables
mydir = r"D:\project1"
sp = {"sep":"\n\n", "end":"\n\n"}

# A. Bussiness Understanding                 
#       1. Formulate research/analysis question(s)
            ## Question I
                ### where are the Airbnb rooms in New York
            ## Question II
                ### where is the cheapest, most affordable Airbnb rooms in New York
            ## Question III
                ### what is the most popular and affordable Airbnb rooms in each New York borough
            ## Question IV
                ### what are the major determinants of prices of rooms in New York Airbnb 

# B. Data Understanding
#     1. Seek for relevant datasets
#           Our dataset comes for publicly available  Airbnb and New York State Department of Health form their websites
#     2. Download relevant datasets
#           Download in the datasets from Airbnb and unitedstateszipcodes.org websites
#   Airbnb websit: "http://data.insideairbnb.com/united-states/ny/new-york-city/2019-12-04/data/listings.csv.gz"
#   New York State Department of Health:  "https://www.health.ny.gov/statistics/cancer/registry/appendix/neighborhoods.htm"

airbnb = "http://data.insideairbnb.com/united-states/ny/new-york-city/2019-12-04/data/listings.csv.gz"
listings = pd.read_csv(airbnb)

# Save data to local disk 
# save as pickle file
pd.to_pickle(Dlistings, os.path.join(airbnb, "airbnb_ny.pkl"))



# C. Data preparation
#     1. Clean datasets
#     2. Features extraction and engineering
#     3. Exploratory data analysis
#     4. Data visualization

pandas_pickle = pd.read_pickle(os.path.join(mydir, "listings.pkl"))

print(pandas_pickle.columns)
