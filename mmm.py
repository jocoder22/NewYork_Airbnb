# Import necessary libraries.
import os
import numpy as np
import pandas as pd
import pickle
from collections import OrderedDict
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


# plt.style.use('ggplot')
# plt.style.use(['classic'])
# plt.style.use(['dark_background'])


def print2(*args):
    for arg in args:
        print(arg, sep="\n\n", end="\n\n")


# Define paths, functions and variables
mydir = r"D:\NewYork_Airbnb"
dir2 = r"C:\Users\HP\Desktop"
     
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
             'host_has_profile_pic', 'host_identity_verified', 'accommodates', 'bathrooms', 'number_of_reviews', 
            'latitude', 'longitude', 'is_location_exact',  'room_type', 'maximum_nights','availability_30',
            'bedrooms', 'beds', 'bed_type', 'price',  'security_deposit', 'cleaning_fee', 'guests_included',
            'extra_people', 'minimum_nights', 'availability_60', 'availability_90', 'availability_365',   
            'instant_bookable',  'require_guest_profile_picture', 'require_guest_phone_verification']           

remove_dollar = ['security_deposit','cleaning_fee', 'extra_people', 'price']


cat = ['host_is_superhost', 'neighbourhood_group_cleansed',
       'host_has_profile_pic', 'host_identity_verified', 'is_location_exact',
       'room_type', 'bed_type',  'instant_bookable',
       'require_guest_profile_picture', 'require_guest_phone_verification']

for_dummy =  ['instant_bookable', 'require_guest_profile_picture', 'require_guest_phone_verification', 'bed_type'] 


features3 = ['host_id', 'host_since', 'host_is_superhost', 
             'host_has_profile_pic', 'host_identity_verified', 'neighbourhood_group_cleansed', 
             'is_location_exact',  'room_type', 'accommodates', 'bathrooms', 
            'bedrooms', 'beds', 'bed_type']

pp = [ 'host_id', 'host_since', 'host_is_superhost']

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
    datasett = data[features]

    # drop duplicates
    dataset = datasett.drop_duplicates(subset = features3, keep = "first")

    # remove dollar signs and turn columns to float
    for col in rmdollar:
        dataset.loc[:, col] = dataset.loc[:, col].replace('[\$,]','', regex=True).astype(float)
        
    # # features to turn to categorical features
    for ele in catfeatures:
        dataset.loc[:, ele] = dataset.loc[:, ele].astype("category", inplace=True)
        
        
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
# (45078, 32)
# pd.to_pickle(working_data, os.path.join(mydir, "airbnb_ny.pkl"))


# Get listing percentage for each New York Borough
ddf = working_data["neighbourhood_group_cleansed"].value_counts(normalize=True) * 100
ddf2 = working_data["neighbourhood_group_cleansed"].value_counts() 
# print2("listing each Borough Percentages :", ddf, " listing each Borough Raw counts" , ddf2 )


"""

listing each Borough Percentages :

Manhattan        42.559563
Brooklyn         42.049337
Queens           12.041351
Bronx             2.555570
Staten Island     0.794179
Name: neighbourhood_group_cleansed, dtype: float64

 listing each Borough Raw counts

Manhattan        19185
Brooklyn         18955
Queens            5428
Bronx             1152
Staten Island      358
Name: neighbourhood_group_cleansed, dtype: int64

"""

# Plot the Airbnb listing in New York
plt.bar( ddf.index, ddf.values,  edgecolor="#2b2b28")
plt.xlabel("New York City Borough")
plt.ylabel("Percentage of Listings")
plt.title("  New York City Airbnb Listing ")
plt.tight_layout()
plt.savefig(os.path.join(dir2, "Listings.png"))
plt.show()


# get the room types percentages
roomtypes = working_data["room_type"].value_counts(normalize=True) * 100
roomtypes2 = working_data["room_type"].value_counts()
# print2("Roomtypes Percentages :", roomtypes, "Raw counts" , roomtypes2 )


# get the room types percentages
roomtypes = working_data["room_type"].value_counts(normalize=True) * 100
roomtypes2 = working_data["room_type"].value_counts()
# print2("Roomtypes Percentages :", roomtypes, "Raw counts" , roomtypes2)

"""


Roomtypes Percentages :

Entire home/apt    53.653667
Private room       43.901682
Shared room         2.036470
Hotel room          0.408181
Name: room_type, dtype: float64

Raw counts

Entire home/apt    24186
Private room       19790
Shared room          918
Hotel room           184
Name: room_type, dtype: int64


"""
# # plot the room types
plt.bar(roomtypes.index, roomtypes.values, edgecolor="#2b2b28")
plt.xlabel("Room Type")
plt.ylabel("Percentage of Total")
plt.title("  New York City Airbnb Listing ")
plt.tight_layout()
plt.savefig(os.path.join(dir2, "RoomTypes.png"))
plt.show()



# get the counts of room_types per bourough
hh = pd.crosstab(working_data["neighbourhood_group_cleansed"], working_data["room_type"], normalize="index", margins = True).fillna(0) * 100
hh2 = pd.crosstab(working_data["neighbourhood_group_cleansed"], working_data["room_type"],  margins = True).fillna(0)
hh3 = pd.crosstab(working_data["neighbourhood_group_cleansed"], working_data["room_type"]).fillna(0) 
# print2("room_types per bourough Percentages :", hh, " room_types per bourough Raw counts" , hh2,
#         " room_types per bourough Raw counts", hh3 )

"""


room_types per bourough Percentages :

room_type                     Entire home/apt  Hotel room  Private room  Shared room
neighbourhood_group_cleansed
Bronx                               37.847222    0.000000     58.246528     3.906250
Brooklyn                            51.015563    0.068583     47.280401     1.635452
Manhattan                           60.734949    0.828772     36.340891     2.095387
Queens                              41.341194    0.221076     55.563744     2.873987
Staten Island                       51.396648    0.000000     47.206704     1.396648
All                                 53.653667    0.408181     43.901682     2.036470

 room_types per bourough Raw counts

room_type                     Entire home/apt  Hotel room  Private room  Shared room    All
neighbourhood_group_cleansed
Bronx                                     436           0           671           45   1152
Brooklyn                                 9670          13          8962          310  18955
Manhattan                               11652         159          6972          402  19185
Queens                                   2244          12          3016          156   5428
Staten Island                             184           0           169            5    358
All                                     24186         184         19790          918  45078

 room_types per bourough Raw counts

room_type                     Entire home/apt  Hotel room  Private room  Shared room
neighbourhood_group_cleansed
Bronx                                     436           0           671           45
Brooklyn                                 9670          13          8962          310
Manhattan                               11652         159          6972          402
Queens                                   2244          12          3016          156
Staten Island                             184           0           169            5


"""

# Plot the distribution of listings room_types within the boroughs
hh.plot.bar(stacked=True, cmap='tab20c', figsize=(10,7), edgecolor="#2b2b28")
plt.xticks(rotation=0)
plt.xlabel("New York City Borough")
plt.ylabel("Percent")
plt.title("  New York City Airbnb Listing ")
plt.tight_layout()
plt.savefig(os.path.join(dir2, "BoroughRoomType.png"))
plt.show()


# find average price of listing in each borough
ave_price = working_data.groupby("neighbourhood_group_cleansed", as_index=False).agg({'price': 'mean'})
# print2("average price of listing in each borough :", ave_price)


"""


average price of listing in each borough :

  neighbourhood_group_cleansed       price
0                        Bronx   92.307292
1                     Brooklyn  128.416249
2                    Manhattan  208.563722
3                       Queens  102.608696
4                Staten Island  109.069832


"""

# # plot the Average price of listing in each Borough
plt.bar(ave_price.neighbourhood_group_cleansed, ave_price.price, edgecolor="#2b2b28")
plt.xlabel("New York City Borough")
plt.ylabel("Average Price")
plt.title("  New York City Airbnb Listing ")
plt.tight_layout()
plt.savefig(os.path.join(dir2, "Prices.png"))
plt.show()



# Average price per room type in each Borough
nprice_room = working_data.groupby(["neighbourhood_group_cleansed", "room_type"], as_index=False, observed=True).agg({'price': 'mean'})
price_room = nprice_room.pivot(index = 'neighbourhood_group_cleansed',
                                 columns = "room_type",
                                 values = "price")
price_room.plot.bar(rot=0, cmap='tab20c', edgecolor="#2b2b28")
plt.xlabel("New York City Borough")
plt.ylabel("Average Price")
plt.title("Airbnb Listing in New York")
plt.tight_layout()
plt.savefig(os.path.join(dir2, "BoRoomPrice.png"))
plt.show()

print2("Average price per room type in each Borough :", nprice_room, 
       "Average price per room type in each Borough: Pivot" , price_room )


"""


Average price per room type in each Borough :

   neighbourhood_group_cleansed        room_type       price
0                     Manhattan  Entire home/apt  251.344576
1                     Manhattan     Private room  140.018503
2                     Manhattan      Shared room  115.079602
3                     Manhattan       Hotel room  315.452830
4                      Brooklyn  Entire home/apt  177.941675
5                      Brooklyn     Private room   76.305735
6                      Brooklyn      Shared room   87.238710
7                      Brooklyn       Hotel room  195.230769
8                        Queens  Entire home/apt  145.081551
9                        Queens     Private room   71.163462
10                       Queens      Shared room   96.724359
11                       Queens       Hotel room  139.916667
12                Staten Island  Entire home/apt  150.842391
13                Staten Island     Private room   65.923077
14                Staten Island      Shared room   30.200000
15                        Bronx  Entire home/apt  131.529817
16                        Bronx     Private room   66.782414
17                        Bronx      Shared room   92.888889

Average price per room type in each Borough: Pivot

room_type                     Entire home/apt  Hotel room  Private room  Shared room
neighbourhood_group_cleansed
Bronx                              131.529817         NaN     66.782414    92.888889
Brooklyn                           177.941675  195.230769     76.305735    87.238710
Manhattan                          251.344576  315.452830    140.018503   115.079602
Queens                             145.081551  139.916667     71.163462    96.724359
Staten Island                      150.842391         NaN     65.923077    30.200000



"""






# Prepare for Supervised machine learning
print2(working_data.shape)
learningdata = working_data.dropna()
print2(learningdata.shape)


def preanalysis(data1):
    """The 'preanalysis' function replaces or drop missing, NAN or Na and
        get dummies from categorical features.
 
    Args: 
        'data1' (DataFrame): the DataFrame for data_wrangling

 
    Returns: 
        DataFrame: The DataFrame for analysis
 
    """
    # drop nans
    df = data1.dropna()

    # Extra the number of days
    ddate = datetime(2020, 2, 12)

    df.loc[:, "date"] = pd.to_datetime(df['host_since'], format='%Y-%m-%d')
    df.loc[:, "days"] = df["date"].apply(lambda x: (ddate - x).days)

    # Drop the datetime columns
    df.drop(['host_since', 'date', "id", "host_id"], axis=1, inplace=True)



    # get dummies
    # dff = pd.get_dummies(df, drop_first=True, prefix_sep="_")

    return df



def dfform(data1):
    v = pd.DataFrame(data1, columns=["coefficients"])
    vv = pd.DataFrame(X_test.columns, columns=["features"])
    df2 = pd.concat([vv, v], axis=1, join="inner")
    df3 = df2.sort_values(by="coefficients")

    return df3


working_data2 = preanalysis(working_data)
print2(len(working_data2.columns))

working_data2.to_csv(os.path.join(mydir, 'clean.csv'), index=False, compression='gzip')
# data = pd.read_csv('textdata.csv', compression='gzip')
# car = pd.read_csv('car.csv', compression='gzip')

"""

for item in cat:
    bb = len(working_data[item].unique())
    print(f'{item} has {bb} groups')
    # print(working_data[[item]].head(), end='\n\n')
    print(working_data[item].value_counts(), end='\n\n')

print2(working_data2.info())



from sklearn.ensemble import RandomForestRegressor 
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline


print2("price" in list(working_data2.columns))
target = working_data2.pop("price")

X_train, X_test, y_train, y_test = train_test_split(working_data2, target, test_size=0.25)

print2("price" in list(working_data2.columns))

pl = Pipeline([
            ("scaler", StandardScaler()),
            # ("dtree", DecisionTreeRegressor())
            # ("reg1",  LinearRegression())
            ("reg2", RandomForestRegressor(n_estimators = 100,
                           n_jobs = -1,
                           oob_score = True,
                           bootstrap = True,
                           random_state = 42))
    
        
])

pl.fit(X_train, y_train)
pred = pl.predict(X_test)

print("mean_absolute_error : ", mean_absolute_error(y_test, pred))
print("mean_squared_error: ", mean_squared_error(y_test, pred))
print("Root mean_squared_error: ", np.sqrt(mean_squared_error(y_test, pred)))
print("R_squared : ", pl.score(X_test, y_test))




import eli5
from eli5.permutation_importance import get_score_importances

rf = RandomForestRegressor(n_estimators = 100,
                           n_jobs = -1,
                           oob_score = True,
                           bootstrap = True,
                           random_state = 42)







rf = pl[-1]
print('R^2 Training Score: {:.2f} \nOOB Score: {:.2f} \nR^2 Validation Score: {:.2f}'.format(rf.score(X_train, y_train), 
                                                                                             rf.oob_score_,
                                                                                             rf.score(X_test, y_test)))


# plt.bar(X_test.columns, pl[-1].coef_)

data7 = dfform(pl[-1].feature_importances_)

plt.barh(data7.features, data7.coefficients)
plt.xticks(rotation=45)
# plt.axhline(y=0.0, color='black', linestyle='-')
plt.show()



from sklearn.metrics import r2_score
from rfpimp import permutation_importances

def r2(pl, X_train, y_train):
    return r2_score(y_train, pl.predict(X_train))

perm_= permutation_importances(pl, X_train, y_train, r2)

pp = pd.DataFrame(perm_.reset_index())
ppp = pp.sort_values(by="Importance")
plt.barh(ppp.Feature, ppp.Importance)
plt.xticks(rotation=45)
# plt.axhline(y=0.0, color='black', linestyle='-')
plt.show()

print2(perm_, ppp.head())




# base_score, score_decreases = get_score_importances(rf, X_train, y_train)
# feature_importances = np.mean(score_decreases, axis=0)
# print2("#"*90, feature_importances)


import eli5
from eli5.sklearn import PermutationImportance

rf.fit(X_train, y_train)
perm = PermutationImportance(rf, cv = None, refit = False, n_iter = 50).fit(X_train, y_train)
perm_imp_eli5 = dfform(perm.feature_importances_)
print2(perm_imp_eli5.head(), perm_imp_eli5)




"""





"""

                                      features  coefficients
22  neighbourhood_group_cleansed_Staten Island      0.000003
31                      bed_type_Pull-out Sofa      0.000004
30                              bed_type_Futon      0.000005
23                      host_has_profile_pic_t      0.000006
32                           bed_type_Real Bed      0.000215
35          require_guest_phone_verification_t      0.000287
34             require_guest_profile_picture_t      0.000332
29                              bed_type_Couch      0.000410
21         neighbourhood_group_cleansed_Queens      0.000819
26                        room_type_Hotel room      0.000827
18                         host_is_superhost_t      0.001015
33                          instant_bookable_t      0.001871
24                    host_identity_verified_t      0.002330
25                         is_location_exact_t      0.002806
19       neighbourhood_group_cleansed_Brooklyn      0.003001
11                             guests_included      0.012350
8                                         beds      0.017263
12                                extra_people      0.017477
6                              availability_30      0.017951
14                             availability_60      0.020871
2                            number_of_reviews      0.027677
15                             availability_90      0.031557
7                                     bedrooms      0.040290
20      neighbourhood_group_cleansed_Manhattan      0.043741
9                             security_deposit      0.045480
5                               maximum_nights      0.052783
16                            availability_365      0.058787
28                       room_type_Shared room      0.059069
3                                     latitude      0.070106
1                                    bathrooms      0.099160
0                                 accommodates      0.111321
13                              minimum_nights      0.155781
27                      room_type_Private room      0.285804
10                                cleaning_fee      0.329109
4                                    longitude      0.374817
17                                        days      0.42590

"""
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
