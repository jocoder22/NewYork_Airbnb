# Import necessary libraries.
import os
import numpy as np
import pandas as pd
import pickle
from collections import OrderedDict
import matplotlib.pyplot as plt

# plt.style.use('ggplot')
# plt.style.use(['classic'])
# plt.style.use(['dark_background'])

def print2(*args):
    for arg in args:
        print(arg, sep="\n\n", end="\n\n")

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
#   The data from New York State Department of Health is stored as csv file: nyczipcode.csv


"""
airbnb = "http://data.insideairbnb.com/united-states/ny/new-york-city/2019-12-04/data/listings.csv.gz"
listing_ny = pd.read_csv(airbnb)
nyc_zip = pd.read_csv(os.path.join(mydir, "nyczipcode.csv"))

"""

# C. Data preparation
#     1. Clean the datasets
#     2. Features extraction and engineering
#     3. Exploratory data analysis
#     4. Data visualization



features_list = ['id', 'room_type', 'price','accommodates', 'bathrooms', 'bedrooms', 'beds', 'bed_type',  'minimum_nights', 
             'number_of_reviews', 'reviews_per_month', 'calculated_host_listings_count', 'availability_365', 
             'number_of_reviews', 'review_scores_value', 'reviews_per_month', 'instant_bookable', 'cancellation_policy', 
             'require_guest_profile_picture', 'require_guest_phone_verification', 'security_deposit', 'cleaning_fee',
             'guests_included', 'extra_people', 'minimum_nights', 'maximum_nights', 'zipcode', 'availability_30', 
             'availability_60', 'availability_90', 'availability_365']

remove_dollar = ['security_deposit','cleaning_fee', 'extra_people', 'price']
for_dummy =  ['instant_bookable', 'require_guest_profile_picture', 'require_guest_phone_verification', 'bed_type'] 


def dict_former(zipdata):
    """The dict_former function forms an ordered dictionary from a DataFrame

    Args: None
        dataset (DataFrame): the DataFrame to turn into dictionary

    Returns: 
        dict: dictionary   

    """
    
    _newdict = zipdata.to_dict(into=OrderedDict)
    
    newdict = {str(_newdict['localzip'][i]):_newdict['Name'][i] for i in range(len(_newdict['Name']))}

    return newdict
    


def data_clearner(data, data2, features, rmdollar, dummy):
    """The data_clearner function will return a clean DataFrame after removing, replacing and
        and cleaning the DataFrame to  a suitable form for further analysis

    Args: 
        data (DataFrame): the DataFrame for data_wrangling
        data2 (DataFrame): DataFrame for creating search dictionary
        features (list): list for features to select from the DataFrame
        rmdollar (list): list of string features with dollar signs
        dummy (list): list of categorical features to turn to dummy variables before feeding into machine learning
        
    Returns: 
        DataFrame: The DataFrame for analysis

    """
    
    # select only the required feaatures
    dataset = data[features]
    
    # extract only the five digits zipcodes 
    dataset["zipcode"] = dataset['zipcode'].str.extract(r'(\d+)', expand=False)
   
    # create lookup dictionary
    searchdict = dict_former(data2)
    
    # Create boroughs from zipcode
    dataset["Boroughs"] = dataset.zipcode.replace(searchdict)
    dataset = dataset.loc[dataset['Boroughs'].isin(searchdict.values())]
    
    # remove dollar signs and turn columns to float
    for col in rmdollar:
        dataset[col] = dataset[col].replace('[\$,]','', regex=True).astype(float)
        
    # form dummies for categorical features
    for ele in dummy:
        dataset = pd.get_dummies(dataset, columns=[ele], prefix=ele.split("_")[-1])
        
        
    return dataset

"""

listing_ny = data_clearner(listing_ny, nyc_zip, features_list, remove_dollar, for_dummy)

# # # Save data to local disk 
# # # save as pickle file
pd.to_pickle(listing_ny, os.path.join(mydir, "airbnb_ny.pkl"))

"""

# read dataset 
working_data = pd.read_pickle(os.path.join(mydir, "airbnb_ny.pkl"))

"""
# Get listing percentage for each New York Borough
ddf = working_data["Boroughs"].value_counts(normalize=True) * 100



# Plot the Airbnb listing in New York
plt.bar( ddf.index, ddf.values,  edgecolor="#2b2b28")
plt.xlabel("New York Boroughs")
plt.ylabel("Percentage of Listings")
plt.title("Airbnb Listing in New York")
plt.tight_layout()
plt.show()


# get the room types percentages
roomtypes = working_data["room_type"].value_counts(normalize=True) * 100

# # plot the room types
plt.bar(roomtypes.index, roomtypes.values, edgecolor="#2b2b28")
plt.xlabel("Room Type")
plt.ylabel("Percentage of Total")
plt.title("Airbnb Listing in New York")
plt.tight_layout()
plt.show()



# get the counts of room_types per bourough
hh = pd.crosstab(working_data["Boroughs"], working_data["room_type"], normalize="index", margins = True).fillna(0) * 100

# Plot the distribution of listings room_types within the boroughs
hh.plot.bar(stacked=True, cmap='tab20c', figsize=(10,7), edgecolor="#2b2b28")
plt.xticks(rotation=0)
plt.ylabel("Percent")
plt.title("Airbnb Listing in New York")
plt.tight_layout()
plt.show()


# find average price of listing in each borough
ave_price = working_data.groupby("Boroughs", as_index=False).agg({'price': 'mean'})

# # plot the Average price of listing in each Borough
plt.bar(ave_price.Boroughs, ave_price.price, edgecolor="#2b2b28")
plt.xlabel("New York Boroughs")
plt.ylabel("Average Price")
plt.title("Airbnb Listing in New York")
plt.tight_layout()
plt.show()



# Average price per room type in each Borough
nprice_room = working_data.groupby(["Boroughs", "room_type"], as_index=False).agg({'price': 'mean'})
price_room = nprice_room.pivot(index = "Boroughs",
                                 columns = "room_type",
                                 values = "price")
price_room.plot.bar(rot=0, cmap='tab20c', edgecolor="#2b2b28")
plt.xlabel("New York Boroughs")
plt.ylabel("Average Price")
plt.title("Airbnb Listing in New York")
plt.tight_layout()
plt.show()



"""

# Prepare for Supervised machine learning
print2(working_data.shape)
learningdata = working_data.dropna()
print2(learningdata.shape)

target = learningdata.pop("price")
print2(learningdata.shape)
"""

ppp = ['number_of_reviews', 
       'review_scores_value', 'reviews_per_month']

print(len(features2))
cat_features, int_features, float_features = [], [], []
features_dtypes = "float64 int64 object".split()
cat_list = [float_features, int_features,  cat_features]

for i, v in enumerate(features_dtypes):
    colnames = listing_ny[features2].select_dtypes(include=[v]).columns.tolist()
    cat_list[i].extend(colnames)
    print(f'There {len(colnames)} {v} features')
    
for item in features2:
    bb = len(listing_ny[item].unique())
    print(f'{item} has {bb} groups')
    print(listing_ny[[item]].head(), end='\n\n')
    
newlist = ['room_type', 
           'instant_bookable',  'cancellation_policy', 
           'require_guest_profile_picture', 'require_guest_phone_verification'
           ]
print(cat_features)

for item in newlist:
    # bb = len(listing_ny[item].unique())
    print(listing_ny[item].unique(), end='\n\n')
    # print(f'{item} has {bb} groups')


string_to = ['security_deposit','cleaning_fee', 'extra_people']
tf =  ['instant_bookable', 'require_guest_profile_picture', 'require_guest_phone_verification', 'bed_type']

print2(listing_ny[string_to].head())

def extractdigits(dataset, col):
    dataset[col] = dataset[col].replace('[\$,]','', regex=True).astype(float)
    
def createdummy(data, col):
    for ele in col:
        data = pd.get_dummies(data, columns=[ele], prefix=ele.split("_")[-1])
        # data.join(df_dummy)
    return data
    
    
for ele in string_to:
    extractdigits(listing_ny, ele) 
    

newdata = createdummy(listing_ny[features2], tf) 
      
print2(listing_ny[string_to].head())

print2(newdata.iloc[:,-7:].head())

list44 = ['property_type', 'room_type', 'accommodates',
       'bathrooms', 'bedrooms', 'beds', 'bed_type' ]


for item in list44:
    # bb = len(listing_ny[item].unique())
    print(listing_ny[item].unique(), end='\n\n')
    # print(f'{item} has {bb} groups')
    
print2(len(newdata.columns))

# col2 = ["Name", "localzip"]
# bb = pd.DataFrame({ "b":"Bronx", "a":Bronxa}, colums=col2)
## Supervised learning

# # # plot the Average price of listing in each Borough
# plt.bar(ave_price.Boroughs, ave_price.price)
# plt.xlabel("New York Boroughs")
# plt.ylabel("Average Price")
# plt.title("Airbnb Listing in New York")
# plt.show()


print(sorted(Queens), set(Queens).intersection(Staten_Island, Manhattan, Bronx, Brooklyn), sep='\n\n')
print(working_data.loc[:,["Boroughs"]].head())

print(working_data.groupby("Boroughs").count()["id"].sort_values(ascending=False)) 

look = working_data.loc[:, ['property_type', 'room_type', 'accommodates',
       'bathrooms', 'bedrooms', 'beds', 'bed_type', 'amenities']]
print(look.head())
# https://chrisalbon.com/python/data_wrangling/pandas_apply_operations_to_groups/
# df['preTestScore'].groupby([df['regiment'], df['company']]).mean()
demean = lambda n: n / sum(n)
print(working_data["room_type"].value_counts(normalize=True) * 100, **sp) 
pp = working_data.groupby(["Boroughs", "room_type"]).size().reset_index()



ggg = working_data.groupby(["Boroughs", "room_type"])
pp.columns = ['Boroughs', 'room_type', 'number']
cohortTable = pp.pivot(index = "Boroughs",
                                 columns = "room_type",
                                 values = "number")
cohortTable.fillna(0, inplace=True)
cohortTable["Total"] = cohortTable.sum(axis=1)
cohortTable = cohortTable.div(cohortTable["Total"], axis=0)  * 100
print(cohortTable, **sp)






cmap22 = [
            'viridis', 'plasma', 'inferno', 'magma',

            'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn',
         
            'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
            'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
            'hot', 'afmhot', 'gist_heat', 'copper',
    
            'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
        
            'Pastel1', 'Pastel2', 'Paired', 'Accent',
            'Dark2', 'Set1', 'Set2', 'Set3',
            'tab10', 'tab20', 'tab20b', 'tab20c',
       
            'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
            'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg', 'hsv',
            'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar']
            
cmap22 = [
            
    
            'PRGn', 
  'tab20c',
       
            ]

for color in cmap22:
    # plt.title('This is for cmap: ' + color)
    # plt.scatter(xplot, yplot, c=np.cos(xplot), cmap=i,
    #             edgecolors='none',
    #             s=np.power(xplot, 4))
    price_room.plot.bar(rot=0, cmap=color,  figsize=(50,27))
    plt.title('This is for cmap: ' + color)
    plt.xticks(rotation=0)
    plt.pause(8)
    # plt.clf()
    
    plt.close()



newcd = pd.read_csv(os.path.join(mydir, "nyczipcode.csv"))
print2(newcd.head())

from collections import OrderedDict, defaultdict
def set_value(row_number, assigned_value): 
    return assigned_value[row_number] 

pp = newcd.to_dict(into=OrderedDict)

gg = {str(pp['localzip'][i]):pp['Name'][i] for i in range(len(pp['Name']))}
gg2 = {pp['localzip'][i]:pp['Name'][i] for i in range(len(pp['Name']))}
print2(gg)

# working_data["mm"] = working_data["zipcode"].apply(set_value, args=(gg,))
working_data["mm"] = working_data["zipcode"].replace(gg2)

print2(working_data.iloc[:,-3:])

"""
