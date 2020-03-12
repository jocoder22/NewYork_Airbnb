# Import necessary libraries.
import os
import numpy as np
import pandas as pd
import pickle
from collections import OrderedDict
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# plt.style.use('ggplot')
# plt.style.use(['classic'])
# plt.style.use(['dark_background'])


def print2(*args):
    for arg in args:
        print(arg, sep="\n\n", end="\n\n")


# Define paths, functions and variables
mydir = r"D:\NewYork_Airbnb"
dir2 = r"C:\Users\okigb\Desktop"
     
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
#           Our dataset comes for publicly available  Airbnb and New York State Department of Health websites
#     2. Download relevant datasets
#           Download datasets from Airbnb and unitedstateszipcodes.org websites
#   Airbnb data from website: "http://data.insideairbnb.com/united-states/ny/new-york-city/2019-12-04/data/listings.csv.gz"
# http://data.insideairbnb.com/united-states/ny/new-york-city/2020-02-12/data/listings.csv.gz
#   New York State Department of Health:  "https://www.health.ny.gov/statistics/cancer/registry/appendix/neighborhoods.htm"
#   The data from New York State Department of Health website was stored as csv file: nyczipcode.csv


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


features_list = ['id',  'host_id', 'host_since', 'host_is_superhost', 'neighbourhood_group_cleansed',
             'host_has_profile_pic', 'host_identity_verified', 'accommodates', 'bathrooms', 
            'latitude', 'longitude', 'is_location_exact',  'room_type', 'maximum_nights','availability_30',
            'bedrooms', 'beds', 'bed_type',  'amenities',  'price',  'security_deposit', 
             'cleaning_fee', 'guests_included', 'extra_people', 'minimum_nights',  
             'availability_60', 'availability_90', 'availability_365', 'number_of_reviews', 
            'instant_bookable',  'require_guest_profile_picture', 'require_guest_phone_verification']           

remove_dollar = ['security_deposit','cleaning_fee', 'extra_people', 'price']


cat = ['host_is_superhost', 'neighbourhood_group_cleansed',
       'host_has_profile_pic', 'host_identity_verified', 'is_location_exact',
       'room_type', 'bed_type',  'instant_bookable',
       'require_guest_profile_picture', 'require_guest_phone_verification']

for_dummy =  ['instant_bookable', 'require_guest_profile_picture', 'require_guest_phone_verification', 'bed_type'] 


def data_clearner(data, features, rmdollar, catfeatures):
    """The data_clearner function will return a clean DataFrame after removing, replacing and
        and cleaning the DataFrame to  a suitable form for further analysis

    Args: 
        data (DataFrame): the DataFrame for data_wrangling
        features (list): list for features to select from the DataFrame
        rmdollar (list): list of string features with dollar signs
        
    Returns: 
        DataFrame: The DataFrame for analysis

    """
    
    # select only the required feaatures
    dataset = data[features]

    
    # remove dollar signs and turn columns to float
    for col in rmdollar:
        dataset[col] = dataset[col].replace('[\$,]','', regex=True).astype(float)
        
    # # features to turn to categorical features
    for ele in catfeatures:
        dataset[ele] = dataset[ele].astype("category", inplace=True)
        
        
    return dataset

"""

listing_ny = data_clearner(listing_ny, nyc_zip, features_list, remove_dollar, for_dummy)

# # # Save data to local disk 
# # # save as pickle file
pd.to_pickle(listing_ny, os.path.join(mydir, "airbnb_ny.pkl"))

"""

# read dataset 
datta = pd.read_pickle(os.path.join(mydir, "df.pkl"))
working_data = data_clearner(datta, features_list, remove_dollar, cat)
print2(working_data.shape)


# Get listing percentage for each New York Borough
ddf = working_data["neighbourhood_group_cleansed"].value_counts(normalize=True) * 100
ddf2 = working_data["neighbourhood_group_cleansed"].value_counts() 
print2("listing each Borough Percentages :", ddf, " listing each Borough Raw counts" , ddf2 )
"""




"""
# Plot the Airbnb listing in New York
plt.bar( ddf.index, ddf.values,  edgecolor="#2b2b28")
plt.xlabel("New York City Borough")
plt.ylabel("Percentage of Listings")
plt.title("  New York City Airbnb Listing ")
plt.tight_layout()
plt.show()


# get the room types percentages
roomtypes = working_data["room_type"].value_counts(normalize=True) * 100
roomtypes2 = working_data["room_type"].value_counts()
print2("Roomtypes Percentages :", roomtypes, "Raw counts" , roomtypes2 )
"""



"""
# # plot the room types
plt.bar(roomtypes.index, roomtypes.values, edgecolor="#2b2b28")
plt.xlabel("Room Type")
plt.ylabel("Percentage of Total")
plt.title("  New York City Airbnb Listing ")
plt.tight_layout()
plt.show()



# get the counts of room_types per bourough
hh = pd.crosstab(working_data["neighbourhood_group_cleansed"], working_data["room_type"], normalize="index", margins = True).fillna(0) * 100
hh2 = pd.crosstab(working_data["neighbourhood_group_cleansed"], working_data["room_type"],  margins = True).fillna(0)
hh3 = pd.crosstab(working_data["neighbourhood_group_cleansed"], working_data["room_type"]).fillna(0) 
print2("room_types per bourough Percentages :", hh, " room_types per bourough Raw counts" , hh2,
        " room_types per bourough Raw counts", hh3 )

"""



"""

# Plot the distribution of listings room_types within the boroughs
hh.plot.bar(stacked=True, cmap='tab20c', figsize=(10,7), edgecolor="#2b2b28")
plt.xticks(rotation=0)
plt.xlabel("New York City Borough")
plt.ylabel("Percent")
plt.title("  New York City Airbnb Listing ")
plt.tight_layout()
plt.show()


# find average price of listing in each borough
ave_price = working_data.groupby("neighbourhood_group_cleansed", as_index=False).agg({'price': 'mean'})
print2("average price of listing in each borough :", ave_price)
"""




"""

# # plot the Average price of listing in each Borough
plt.bar(ave_price.neighbourhood_group_cleansed, ave_price.price, edgecolor="#2b2b28")
plt.xlabel("New York City Borough")
plt.ylabel("Average Price")
plt.title("  New York City Airbnb Listing ")
plt.tight_layout()
plt.show()





"""
nprice_room = working_data.groupby(["Boroughs", "room_type"], as_index=False).agg({'price': 'mean'})
price_room = nprice_room.pivot(index = "Boroughs",
                                 columns = "room_type",
                                 values = "price")
price_room.plot.bar(rot=0, cmap='tab20c', edgecolor="#2b2b28")
plt.xlabel("New York City Borough")
plt.ylabel("Average Price")
plt.title("Airbnb Listing in New York")
plt.tight_layout()
plt.show()
"""

# Average price per room type in each Borough
nprice_room = working_data.groupby(["neighbourhood_group_cleansed", "room_type"], as_index=False).agg({'price': 'mean'})
price_room = nprice_room.pivot(index = ['neighbourhood_group_cleansed', 'room_type'],
                                 columns = "room_type",
                                 values = "price")
price_room.plot.bar(rot=0, cmap='tab20c', edgecolor="#2b2b28")
plt.xlabel("New York City Borough")
plt.ylabel("Average Price")
plt.title("Airbnb Listing in New York")
plt.tight_layout()
plt.show()

print2("Average price per room type in each Borough :", nprice_room, 
       "Average price per room type in each Borough: Pivot" , price_room )

print2(nprice_room, nprice_room.index)

"""



"""


# Prepare for Supervised machine learning
print2(working_data.shape)
learningdata = working_data.dropna()
print2(learningdata.shape)


# Cells that are in green show positive correlation, 
# while cells that are in red show negative correlation
# a, b = 0, 6
# while b < learningdata.shape[1]:
#     sns.heatmap(learningdata.iloc[:, a:b].corr(), square=True, cmap='RdYlGn')
#     a = b
#     b += 6
#     plt.pause(3) 
#     plt.close()

# scale the dataset

"""
scaler = StandardScaler()
  
learningdata = pd.get_dummies(learningdata, prefix="DD")
target = learningdata.pop("price")
print2(learningdata.shape)

learningdata = scaler.fit_transform(learningdata)
# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    learningdata, target, test_size=0.25, random_state=42)

# Create the regressor: reg_all
reg_all = LinearRegression()

# Fit the regressor to the training data
reg_all.fit(X_train, y_train)

# Predict on the test data: y_pred
y_pred = reg_all.predict(X_test)

# Compute and print R^2 and RMSE
print("R^2: {}".format(reg_all.score(X_test, y_test)))  
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error: {}".format(rmse))

#Predict using your model
y_test_preds = reg_all.predict(X_test)
y_train_preds = reg_all.predict(X_train)


#Score using your model
test_score = r2_score(y_test, y_test_preds)
train_score = r2_score(y_train, y_train_preds)

print2(test_score, train_score)


# Create a linear regression object: reg
reg = LinearRegression()

# Compute 5-fold cross-validation scores: cv_scores
cv_scores = cross_val_score(reg, learningdata, target, cv=5)

# Print the 5-fold cross-validation scores
print(cv_scores)

print("Average 5-Fold CV Score: {}".format(cv_scores.mean()))


"""



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
