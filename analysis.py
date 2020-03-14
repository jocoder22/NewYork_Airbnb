import os
import numpy as np
import pandas as pd
import pickle
import re
from collections import defaultdict

mydir = r"D:\project1"

sp = {"sep":"\n\n", "end":"\n\n"}

def des(*datas):
    for data in datas:
        name = [x for x in globals() if globals()[x] is data][0]
        print(f"#########################  Printing for {name} ######################\n"*5)
        print(data.head(), data.info(), data.shape, data.describe(), data.columns, sep="\n\n", end="\n\n")
        
def mmshape(*datat):
    for data in datat:
        name = [x for x in globals() if globals()[x] is data][0]
        print(f"{name} has dimension : {data.shape}", sep="\n\n")
        
def print2(*args):
    for arg in args:
        print(arg, end="\n\n")
        
# ww = "http://data.insideairbnb.com/united-states/ny/new-york-city/2019-12-04/visualisations/listings.csv"
# ww2 = "http://data.insideairbnb.com/united-states/ny/new-york-city/2019-12-04/visualisations/reviews.csv"
# nhood = "http://data.insideairbnb.com/united-states/ny/new-york-city/2019-12-04/visualisations/neighbourhoods.csv"
# Dc = "http://data.insideairbnb.com/united-states/ny/new-york-city/2019-12-04/data/calendar.csv.gz"
# Dr = "http://data.insideairbnb.com/united-states/ny/new-york-city/2019-12-04/data/reviews.csv.gz"
# Dl = "http://data.insideairbnb.com/united-states/ny/new-york-city/2019-12-04/data/listings.csv.gz"


# Reviews = pd.read_csv(ww2)
# Neighborhood = pd.read_csv(nhood)

# Dcalender = pd.read_csv(Dc)
# Dreview = pd.read_csv(Dl)



# des(Listings, Reviews, Neighborhood, Dlistings, Dcalender, Dreview)
"""
ww = "http://data.insideairbnb.com/united-states/ny/new-york-city/2019-12-04/visualisations/listings.csv"
Dl = "http://data.insideairbnb.com/united-states/ny/new-york-city/2019-12-04/data/listings.csv.gz"
Listings = pd.read_csv(ww)
Dlistings = pd.read_csv(Dl)
# des(Listings, Dlistings)
# mmshape(Listings,  Dlistings)
print(Dlistings.info())

# dtt = "float int64 object".split()
dtt = "float64 int64 object".split()
datt = "DRF DRI DRO".split()

lls = []
for i, v in enumerate(datt):
    v = Dlistings.select_dtypes(include=dtt[i])
    # print(v.info(), sep="\n\n")
    print(v.notnull().all().sum(), v.shape[1])
    for t, vt  in enumerate(v.notnull().all().values):
        if vt == True:
            lls.append(v.notnull().all().index[t])
    


print("\n\n")
dfm = Dlistings[lls]
print(lls, len(lls), Dlistings.shape, dfm.shape, **sp)
print(dfm[["price"]].head())
ggg = dfm['price']
dfm2 = dfm['price'].str.replace(r'\D+', '').astype(int) / 100
dfm22 = dfm['price'].str.extract(r'(\d+)', expand=False).astype(int) 
dfm3 = dfm['price'].replace('[\$,]','', regex=True).astype(float)
# dfm['price'] = dfm['price'].map(lambda x: x.lstrip('$,').rstrip('aAbBcC,')).astype(float)
print(dfm.price.head(), dfm2.mean(), dfm22.mean(), dfm3.mean(), **sp)
print(dfm2[:100], dfm22[:100], dfm3[:100], **sp)
print((dfm2 == dfm3).sum())
indx = []
for i,v in enumerate(dfm2):
    if v != dfm22[i]:
        # print(v, dfm3[i])
        indx.append(i)
    
# print(indx)

print(dfm22[77], dfm2[77],  ggg[77])


print(Dlistings.columns)


# Using pandas methods
# save pickle
pd.to_pickle(Dlistings, os.path.join(mydir, "Dlistings.pkl"))
pd.to_pickle(Listings, os.path.join(mydir, "Listings.pkl"))
"""
# import pickle file
pandas_pickle = pd.read_pickle(os.path.join(mydir, "Dlistings.pkl"))
print(pandas_pickle.head(), pandas_pickle.shape, **sp)
nycode = pd.read_pickle(os.path.join(mydir, "nyzipcode.pkl"))
nycode["zip"] = nycode["zip"].astype(object)
pandas_pickle.dropna(subset=['zipcode'], inplace=True)
pandas_pickle["zipcode"] = pandas_pickle["zipcode"].astype(object)



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

def stringform(a):
    return [str(i) for i in a]

newdict = {"Bronx": stringform(Bronx), "Brooklyn": stringform(Brooklyn), "Manhattan": stringform(Manhattan),
           "Queens": stringform(Queens), "Staten_Island": stringform(Staten_Island),
           "Westchester": stringform(Westchester), "Nassau": stringform(Nassau)}





nydict = defaultdict(list)
for i, row in nycode.loc[:, ["zip", "county"]].iterrows():
# for i, row in nycode.loc[:, ["zip", "primary_city"]].iterrows():
    nydict[row.values[1]].append(str(row.values[0]))  

newlist = ['id', 'name', 'city', 'latitude', 'longitude',
       'is_location_exact', 'property_type', 'room_type', 'accommodates',
       'bathrooms', 'bedrooms', 'beds', 'bed_type', 'amenities',
       'price', 'security_deposit', "state", "street", "neighborhood",
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

list2 = ['id', 'name', 'city', 'latitude', 'longitude',
       'is_location_exact', 'property_type', 'room_type', 
       'bathrooms', 'bedrooms', 'price',  "street", "neighborhood",
       'cleaning_fee', 'guests_included',  'zipcode', 'Newcity','Newcity2']

dtt = "float64 int64 object".split()
datt = "DRF DRI DRO".split()

Dlistings = pandas_pickle.loc[:, newlist]


print(Dlistings.dropna( axis=0).shape, **sp)

nyc = ['NY','ny', 'Ny', 'New York']
cat = ["room_type", "cancellation_policy"]
mm = ['is_location_exact', 'room_type', 'bed_type', 'calendar_last_scraped', 'instant_bookable', 'city',
      'cancellation_policy', 'require_guest_profile_picture', 'require_guest_phone_verification']
print(Dlistings[Dlistings["state"] == "NY"].shape, len(Dlistings.cancellation_policy.unique()), **sp)

for memb in mm:
    print(Dlistings[memb].unique())
    
Dlistings["price2"] = Dlistings['price'].replace('[\$,]','', regex=True).astype(float)

for memb in mm:
    print(Dlistings.groupby(memb)["price2"].mean().sort_values(ascending=True))
    
def assing(a):
    # for i in a:
    for key in nydict:
        if a in nydict[key]:
            # print(f"{i} is the zipcode in {key}\n")
            return key
        
def assing2(a):
    # for i in a:
    for key in newdict:
        if a in newdict[key]:
            # print(f"{i} is the zipcode in {key}\n")
            return key
        else:
            continue

pandas_pickle["zipcode"] = pandas_pickle['zipcode'].str.extract(r'(\d+)', expand=False)           
print2(pandas_pickle.zipcode.head(), pandas_pickle.loc[:, list2].info(), nycode.info(), nycode.head())
pandas_pickle["Newcity"] = pandas_pickle.apply(lambda row : assing(row["zipcode"]),  axis = 1)
pandas_pickle["Newcity2"] = pandas_pickle.apply(lambda row : assing2(row["zipcode"]),  axis = 1)
pandas_pickle["Newcity2"] = np.where(pandas_pickle["Newcity2"].isnull(), pandas_pickle["Newcity"], pandas_pickle["Newcity2"])
pandas_pickle["price2"] = pandas_pickle['price'].replace('[\$,]','', regex=True).astype(float)

# print2(pandas_pickle.Newcity.unique(), nycode.head())
# print2(pandas_pickle.groupby("Newcity")["price2"].mean().sort_values(ascending=True))
# print2(pandas_pickle.loc[:, list2].info(), nycode.info(), nycode.head())

pp = pandas_pickle[pandas_pickle.Newcity.isnull()].zipcode

newcode = pp.str.split(' - ')

print2(newcode)

newbronx = [str(i) for i in Bronx]
print2(pandas_pickle.Newcity.unique())
print(str([11364, 11354, 11355, 11356, 11357]))
print2(pp, newbronx)
print2(pandas_pickle["Newcity2"].isnull().sum())
print2(pandas_pickle.groupby("Newcity")["price2"].mean().sort_values(ascending=True).round(2))
for k, v in newdict.items():
    print(f'{k} has length of {len(v)}')
"""
Bronx	Central Bronx	10453, 10457, 10460
Bronx Park and Fordham	10458, 10467, 10468
High Bridge and Morrisania	10451, 10452, 10456
Hunts Point and Mott Haven	10454, 10455, 10459, 10474
Kingsbridge and Riverdale	10463, 10471
Northeast Bronx	10466, 10469, 10470, 10475
Southeast Bronx	10461, 10462,10464, 10465, 10472, 10473
Brooklyn	Central Brooklyn	11212, 11213, 11216, 11233, 11238
Southwest Brooklyn	11209, 11214, 11228
Borough Park	11204, 11218, 11219, 11230
Canarsie and Flatlands	11234, 11236, 11239
Southern Brooklyn	11223, 11224, 11229, 11235
Northwest Brooklyn	11201, 11205, 11215, 11217, 11231
Flatbush	11203, 11210, 11225, 11226
East New York and New Lots	11207, 11208
Greenpoint	11211, 11222
Sunset Park	11220, 11232
Bushwick and Williamsburg	11206, 11221, 11237
Manhattan	Central Harlem	10026, 10027, 10030, 10037, 10039
Chelsea and Clinton	10001, 10011, 10018, 10019, 10020, 10036
East Harlem	10029, 10035
Gramercy Park and Murray Hill	10010, 10016, 10017, 10022
Greenwich Village and Soho	10012, 10013, 10014
Lower Manhattan	10004, 10005, 10006, 10007, 10038, 10280
Lower East Side	10002, 10003, 10009
Upper East Side	10021, 10028, 10044, 10065, 10075, 10128
Upper West Side	10023, 10024, 10025
Inwood and Washington Heights	10031, 10032, 10033, 10034, 10040
Queens	Northeast Queens	11361, 11362, 11363, 11364
North Queens	11354, 11355, 11356, 11357, 11358, 11359, 11360
Central Queens	11365, 11366, 11367
Jamaica	11412, 11423, 11432, 11433, 11434, 11435, 11436
Northwest Queens	11101, 11102, 11103, 11104, 11105, 11106
West Central Queens	11374, 11375, 11379, 11385
Rockaways	11691, 11692, 11693, 11694, 11695, 11697
Southeast Queens	11004, 11005, 11411, 11413, 11422, 11426, 11427, 11428, 11429
Southwest Queens	11414, 11415, 11416, 11417, 11418, 11419, 11420, 11421
West Queens	11368, 11369, 11370, 11372, 11373, 11377, 11378
Staten Island	Port Richmond	10302, 10303, 10310
South Shore	10306, 10307, 10308, 10309, 10312
Stapleton and St. George	10301, 10304, 10305
Mid-Island	10314
"""
print(pandas_pickle[pandas_pickle["Newcity2"] == "Nassau County"]["zipcode"].unique())  
    





"""

rom sklearn.base import clone 

def drop_col_feat_imp(model, X_train, y_train, random_state = 42):
    
    # clone the model to have the exact same specification as the one initially trained
    model_clone = clone(model)
    # set random_state for comparability
    model_clone.random_state = random_state
    # training and scoring the benchmark model
    model_clone.fit(X_train, y_train)
    benchmark_score = model_clone.score(X_train, y_train)
    # list for storing feature importances
    importances = []
    
    # iterating over all columns and storing feature importance (difference between benchmark and new model)
    for col in X_train.columns:
        model_clone = clone(model)
        model_clone.random_state = random_state
        model_clone.fit(X_train.drop(col, axis = 1), y_train)
        drop_col_score = model_clone.score(X_train.drop(col, axis = 1), y_train)
        importances.append(benchmark_score - drop_col_score)
    
    importances_df = imp_df(X_train.columns, importances)
    return importances_df


"""
    
    
