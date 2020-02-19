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
    

# clean zipcode by extracting digits, drop NaN and change it to integer
working_data["zipcode"] = working_data['zipcode'].str.extract(r'(\d+)', expand=False)
working_data.dropna(subset=["zipcode"], inplace=True)
working_data["zipcode"] = working_data["zipcode"] .astype(int) 

# create borough based on zipcode
Nassau = [11001, 11559]
Westchester = [10550, 10705]
Bronx = [10453, 10457, 10460, 10458, 10467, 10468, 10451, 10452, 10456,
	10454, 10455, 10459, 10474, 10463, 10471,
	10466, 10469, 10470, 10475, 10461, 10462,10464, 10465, 10472, 10473]
Brooklyn = [11212, 11213, 11216, 11233, 11238, 11209, 11214, 11228,
	11204, 11218, 11219, 11230, 11234, 11236, 11239,
	11223, 11224, 11229, 11235, 11201, 11205, 11215, 11217, 11231,
	11203, 11210, 11225, 11226, 11207, 11208, 11211, 11222,
	11220, 11232, 11206, 11221, 11237, 11249, 11243]
Manhattan =	[10026, 10027, 10030, 10037, 10039, 10001, 10011, 10018, 10019, 10020, 10036,
	10029, 10035, 10010, 10016, 10017, 10022,
	10012, 10013, 10014, 10004, 10005, 10006, 10007, 10038, 10280,
	10002, 10003, 10009, 10021, 10028, 10044, 10065, 10075, 10128,
	10023, 10024, 10025, 10031, 10032, 10033, 10034, 10040,
    10162, 10069, 10282, 10129, 10281, 10174, 10270]
Queens = [11361, 11362, 11363, 11364, 11354, 11355, 11356, 11357, 11358, 11359, 11360,
	11365, 11366, 11367, 11412, 11423, 11432, 11433, 11434, 11435, 11436,
	11101, 11102, 11103, 11104, 11105, 11106, 11374, 11375, 11379, 11385,
	11691, 11692, 11693, 11694, 11695, 11697, 11004, 11005, 11411, 11413, 11422, 11426, 11427, 11428, 11429,
	11414, 11415, 11416, 11417, 11418, 11419, 11420, 11421,11368, 11369, 11370, 11372, 11373, 11377, 11378, 11109]
Staten_Island = [10302, 10303, 10310, 10306, 10307, 10308, 10309, 10312, 10301, 10304, 10305, 10314]


print(sorted(Queens), set(Queens).intersection(Staten_Island, Manhattan, Bronx, Brooklyn), sep='\n\n')
print(working_data.loc[:,["zipcode"]].head())  

mak = [Queens, Staten_Island, Manhattan, Bronx, Brooklyn]
def ranger(x):
    maxx = max(x)
    minn = min(x)
    print(f"maximium {maxx}, minimium {minn}")
    
for elem in mak:
    ranger(elem)
    
# maximium 11697, minimium 11004
# maximium 10314, minimium 10301
# maximium 10282, minimium 10001
# maximium 10475, minimium 10451
# maximium 11249, minimium 11201