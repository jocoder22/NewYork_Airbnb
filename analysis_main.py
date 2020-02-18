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

# airbnb = "http://data.insideairbnb.com/united-states/ny/new-york-city/2019-12-04/data/listings.csv.gz"
# listing_ny = pd.read_csv(airbnb)

# # # Save data to local disk 
# # # save as pickle file
# pd.to_pickle(listing_ny, os.path.join(mydir, "airbnb_ny.pkl"))



# C. Data preparation
#     1. Clean the datasets
#     2. Features extraction and engineering
#     3. Exploratory data analysis
#     4. Data visualization

ny_data = pd.read_pickle(os.path.join(mydir, "airbnb_ny.pkl"))
# print(ny_data.columns, ny_data.shape)


# select features to for data analysis
feature_list = ['id', 'latitude', 'longitude',
       'is_location_exact', 'property_type', 'room_type', 'accommodates',
       'bathrooms', 'bedrooms', 'beds', 'bed_type', 'amenities',
       'price', 'security_deposit',
       'cleaning_fee', 'guests_included', 'extra_people', 'minimum_nights',
       'maximum_nights', 'minimum_minimum_nights', 'maximum_minimum_nights',
       'minimum_maximum_nights', 'maximum_maximum_nights', 'zipcode',
       'minimum_nights_avg_ntm', 'maximum_nights_avg_ntm', 'calendar_updated',
       'has_availability', 'availability_30', 'availability_60',
       'availability_90', 'availability_365', 'calendar_last_scraped',
       'number_of_reviews', 'number_of_reviews_ltm', 'first_review',
       'last_review', 'review_scores_rating', 'review_scores_accuracy',
       'review_scores_cleanliness', 'review_scores_checkin',
       'review_scores_communication', 'review_scores_location',
       'review_scores_value', 'requires_license',
        'instant_bookable', 'is_business_travel_ready',
       'cancellation_policy', 'require_guest_profile_picture',
       'require_guest_phone_verification', 'calculated_host_listings_count',
       'calculated_host_listings_count_entire_homes',
       'calculated_host_listings_count_private_rooms',
       'calculated_host_listings_count_shared_rooms', 'reviews_per_month']


# get dtypes list
working_data = ny_data.loc[:, feature_list]
features_dtypes = "float64 int64 object".split()

cat_features, int_features, float_features = [], [], []

cat_list = [float_features, int_features,  cat_features]

for i, v in enumerate(features_dtypes):
    colnames = working_data.select_dtypes(include=[v]).columns
    cat_list[i].append(colnames)
    print(f'There {len(colnames)} {v} features')
    

# print(working_data.head(), end="\n\n")

# print(float_features, int_features,  cat_features, sep="\n\n")

# print(ny_data.info(), working_data.info(), features_dtypes)
