# Import necessary libraries.
import os
import numpy as np
import pandas as pd
import pickle
from collections import defaultdict

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
#           Our dataset comes for publicly available  Airbnb and unitedstateszipcodes.org form their websites
#     2. Download relevant datasets
#           Download in the datasets from Airbnb and unitedstateszipcodes.org websites
airbnb = "http://data.insideairbnb.com/united-states/ny/new-york-city/2019-12-04/data/listings.csv.gz"
uszp = "https://www.unitedstateszipcodes.org/zip_code_database.csv?download_auth=3b54c0c5134f0b49e7512b2140749642"
listings = pd.read_csv(airbnb)
zipcodes = pd.read_csv(uszp)

# C. Data preparation
#     1. Clean datasets
#     2. Features extraction and engineering
#     3. Exploratory data analysis
#     4. Data visualization

pandas_pickle = pd.read_pickle(os.path.join(mydir, "listings.pkl"))

print(pandas_pickle.columns)
