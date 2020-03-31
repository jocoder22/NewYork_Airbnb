#!/usr/bin/env python
# coding: utf-8

# # Analysis PLan 
#  
# Writing a Data Scientist Blog Post
# 
#       using
#                  
# Cross-Industry Standard Process of Data Mining - (CRISP-DM)
# 
# ## A. Bussiness Understanding                 
#      1. Formulate research/analysis question(s)
# ## B. Data Understanding
#      1. Seek for relevant datasets
#      2. Download relevant datasets
# ## C. Data preparation
#     1. Data Exploration
#     2. Cleaning data
#     3. Features extraction and engineering
#     4. Exploratory data analysis
# ## D. Analysis and Modelling
#     1. Answers to research/analysis question(s)
#     2. Supervised learning: Predicting prices
# ## E. Deployment
#     1. Summary report
#     2. Conclusion(s)
#     
# 
# 

# ## A. Bussiness Understanding                 
# ###       1. Formulate research/analysis question(s)
# 
# * Question I
#    - Where are the Airbnb rooms in New York City?
#       -  Analyzing the distribution of Airbnb listing in the 5 boroughs of NYC.
# * Question II
#   - Where is the most affordable best Airbnb rooms in New York using average price in each borough.
#     -  Analyzing the distribution of Airbnb average prices in the 5 boroughs of NYC.
# * Question III
#   - What is the most popular and affordable Airbnb rooms in each New York borough
#     - Finding the average prices of Airbnb rooms according to room types in each NYC borough.
# * Question IV
#   - What are the major determinants of prices of rooms in New York Airbnb 
#     - Finding factors that positively and negatively impact the price of Airbnb rooms in NYC using linear regression analysis.

# ## B. Data Understanding
# ###     1. Seek for relevant datasets
# * Our dataset comes for publicly available  Airbnb and New York State Department of Health websites
# ###     2. Download relevant datasets
#     - Download datasets from Airbnb data using the [url](http://data.insideairbnb.com/united-states/ny/new-york-city/2020-02-12/data/listings.csv.gz)

# In[3]:


# Import necessary packages.
import numpy as np
import pandas as pd
 
from datetime import datetime
from math import log
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import StackingRegressor, GradientBoostingRegressor, VotingRegressor, RandomForestRegressor 
from sklearn.linear_model import LinearRegression, TheilSenRegressor, RANSACRegressor, HuberRegressor, RidgeCV, LassoCV
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor 

from xgboost import XGBRegressor
from eli5.sklearn import PermutationImportance

sns.set(style="ticks", color_codes=True)
plt.rc('figure', figsize=[12,8], dpi=100)

def print2(*args):
    for arg in args:
        print(arg, sep="\n\n", end="\n\n")


airbnb = "http://data.insideairbnb.com/united-states/ny/new-york-city/2020-02-12/data/listings.csv.gz"
data = pd.read_csv(airbnb)


# ## C. Data preparation
# ###    1. Data Exploration
# 



print2(data.shape, data.info(), data.columns, data.dtypes)



# explore in depth each feature, and select features of interest
print2(data.head())


# explore selected features
selected_features = ['id',  'host_id', 'host_since', 'host_is_superhost', 'neighbourhood_group_cleansed',
             'host_has_profile_pic', 'host_identity_verified', 'accommodates', 'bathrooms', 'number_of_reviews', 
            'latitude', 'longitude', 'is_location_exact',  'room_type', 'maximum_nights','availability_30',
            'bedrooms', 'beds', 'bed_type', 'price',  'security_deposit', 'cleaning_fee', 'guests_included',
            'extra_people', 'minimum_nights', 'availability_60', 'availability_90', 'availability_365',   
            'instant_bookable',  'require_guest_profile_picture', 'require_guest_phone_verification']   

selected = data[selected_features]
selected.info()


# explore in depth each selected feature
selected.head(10)




# features with dollar signs
dollarfeatures = ['security_deposit','cleaning_fee', 'extra_people', 'price']
selected[dollarfeatures].head()



# object features for categorical dtypes
object_cat = ['host_is_superhost', 'neighbourhood_group_cleansed',
       'host_has_profile_pic', 'host_identity_verified', 'is_location_exact',
       'room_type', 'bed_type',  'instant_bookable',
       'require_guest_profile_picture', 'require_guest_phone_verification']

selected[object_cat].head()




# explore other numerical selected features
selected.select_dtypes(include=["int64", "float64"]).head()




# explore object dtype selected features
# host_since is acutally a datetime dtype
# confirmed security_deposit,cleaning_fee, extra_people, price are actually numerical dtype with dollar signs
selected.select_dtypes(include=["object"]).head()


# ## C. Data preparation
# ###    2. Cleaning data
# ### 3. Features extraction and engineering
# * define the data_cleaner function




def data_cleaner(data, features, rmdollar, catfeatures, duplicate):
    """The data_cleaner function will return a clean DataFrame after removing, replacing and
        and cleaning the DataFrame to  a suitable form for further analysis

    Args: 
        data (DataFrame): the DataFrame for data wrangling
        features (list): list for features of interest to select from the DataFrame
        rmdollar (list): list features with dollar signs to remove dollar sign and turn to numeric dtype
        catfeatures (list): list of features to change dtype to category
        duplicates (list): list of features to for duplicate removal
        
    Returns: 
        DataFrame: The DataFrame for analysis

    """
    
    # select only the features of interest to select from the DataFrame
    datasett = data[features]

    # drop duplicates
    dataset = datasett.drop_duplicates(subset = duplicate, keep = "first")

    # remove dollar signs and turn columns to float
    for col in rmdollar:
        dataset.loc[:, col] = dataset.loc[:, col].replace('[\$,]','', regex=True).astype(float)
        
    # # features to turn to categorical features
    for ele in catfeatures:
        dataset.loc[:, ele] = dataset.loc[:, ele].astype("category", inplace=True)
        
    # Extra the number of days on NYC Airbnb from host_since
    ddate = datetime(2020, 2, 12)

    dataset.loc[:, "date"] = pd.to_datetime(dataset.loc[:, 'host_since'], format='%Y-%m-%d')
    dataset.loc[:, "host_days_on_Airbnb"] = dataset.loc[:, "date"].apply(lambda x: (ddate - x).days)
    dataset.loc[:, "Borough"] = dataset.loc[:, "neighbourhood_group_cleansed"]

    # Drop the datetime columns
    _dataset = dataset.drop(['host_since', 'date', "id", "host_id", "neighbourhood_group_cleansed"], axis=1)
        
        
    return _dataset



dup_list = ['host_id', 'host_since', 'host_is_superhost', 'host_has_profile_pic', 'is_location_exact', 
            'host_identity_verified', 'neighbourhood_group_cleansed', 'accommodates', 'bathrooms', 
              'bedrooms', 'beds', 'bed_type',  'room_type']
            



# perform data cleaning
cleandata = data_cleaner(data, selected_features, dollarfeatures, object_cat, dup_list)


cleandata.info()




# explore in depth cleandata
cleandata.head(10)


# ## C. Data preparation
# ###    4. Exploratory data analysis



cleandata.shape


cleandata.describe()


# visualise pair wise relationship
sns.pairplot(cleandata);



# ecplore pairwise on selected features
selected_pairwise = ['security_deposit', 'price','cleaning_fee', 'guests_included','availability_90', 'availability_365'] 

sns.pairplot(cleandata[selected_pairwise]);


# explore correlations among features
# strong positive correlation between availability_60 and availability_90
cleandata_correlation = cleandata.corr()
sns.heatmap(cleandata_correlation,  square=True,cmap="YlGnBu")




# drop avaliability 30days and 60days
cleandata = cleandata.drop(columns=['availability_60','availability_30'])


# explore the distribution of prices
# price distribution is right skewed
sns.distplot(cleandata.price,rug=True)



# notable records with price equal to zero
# price equal to zero is unattainable, so select rows with prices greater than zero
# explore log price to correct skewness for linear regression

mask = cleandata.price > 0
cleandata['logprice'] = np.log(cleandata[mask].price).round(4)
sns.distplot(cleandata['logprice'], hist=False, rug=True)




num_features = ['security_deposit','cleaning_fee', 'extra_people', 'logprice', 'longitude', 'latitude','is_location_exact']
sns.pairplot(cleandata[num_features], hue="is_location_exact",  palette="husl");


# # D. Analysis and Modelling
#     1. Answers to research/analysis question(s)  
#     
# ###        A. Formulate research/analysis question(s)
# 
#         Question I:
#            - Where are the Airbnb rooms in New York City?
#               -  Analyzing the distribution of Airbnb listing in the 5 boroughs of NYC.



# # Get listing percentage for each New York Borough
# Analyzing the distribution of Airbnb listing in the 5 boroughs of NYC.
ddf = cleandata["Borough"].value_counts(normalize=True) * 100
ddf2 = cleandata["Borough"].value_counts() 
print("\n\n\n\nlisting each Borough Percentages :", ddf, "\n\n\n\nlisting each Borough Raw counts" , ddf2, sep="\n" )



# Plot the Airbnb listing in New York
plt.bar( ddf.index, ddf.values,  edgecolor="#2b2b28")
plt.xlabel("New York City Borough")
plt.ylabel("Percentage of Listings")
plt.title("  New York City Airbnb Listing ")
plt.tight_layout()
plt.show()



# # get the room types percentages
roomtypes = cleandata["room_type"].value_counts(normalize=True) * 100
roomtypes2 = cleandata["room_type"].value_counts()
print("\n\n\n\nRoomtypes Percentages :", roomtypes, "\n\n\n\nRaw counts" , roomtypes2 , sep="\n")




# # # plot the room types
plt.bar(roomtypes.index, roomtypes.values, edgecolor="#2b2b28")
plt.xlabel("Room Type")
plt.ylabel("Percentage of Total")
plt.title("  New York City Airbnb Listing ")
plt.tight_layout()
plt.show()






# ### A. Formulate research/analysis question(s)
# 
#        Question II:
# 
#         - Where is the most affordable best Airbnb rooms in New York using average price in each borough.
#             - Analyzing the distribution of Airbnb average prices in the 5 boroughs of NYC.


# # get the counts of room_types per bourough
hh = pd.crosstab(cleandata["Borough"], cleandata["room_type"], normalize="index", margins = True).fillna(0) * 100
hh2 = pd.crosstab(cleandata["Borough"], cleandata["room_type"],  margins = True).fillna(0)
hht = pd.crosstab(cleandata["Borough"], cleandata["room_type"], normalize="all").fillna(0) * 100
print("\n\n\n\nroom_types per bourough Percentages :", hh, " \n\n\n\nroom_types per bourough Raw counts" , hh2,
        " \n\n\n\nroom_types per bourough Raw counts", hht , sep="\n")


# # Plot the distribution of listings room_types within the boroughs
hh.plot.bar(stacked=True, cmap='tab20c', figsize=(10,7), edgecolor="#2b2b28")
plt.xticks(rotation=0)
plt.xlabel("New York City Borough")
plt.ylabel("Percent")
plt.title("  New York City Airbnb Listing ")
plt.tight_layout()
plt.show()




# # Plot the distribution of listings room_types across the boroughs
hht.plot.bar(stacked=True, cmap='tab20c', figsize=(10,7), edgecolor="#2b2b28")
plt.xticks(rotation=0)
plt.xlabel("New York City Borough")
plt.ylabel("Percent")
plt.title("  New York City Airbnb Listing ")
plt.tight_layout()
plt.show()





# ### A. Formulate research/analysis question(s)
# 
#        Question III:
# 
#         - What is the most popular and affordable Airbnb rooms in each New York borough
#             - Finding the average prices of Airbnb rooms according to room types in each NYC borough.
#             
# 

# # find average price of listing in each borough
ave_price = cleandata.groupby("Borough", as_index=False).agg({'price': 'mean'})
print("average price of listing in each borough :", ave_price, sep="\n")




# # # plot the Average price of listing in each Borough
plt.bar(ave_price.Borough, ave_price.price, edgecolor="#2b2b28")
plt.xlabel("New York City Borough")
plt.ylabel("Average Price")
plt.title("  New York City Airbnb Listing ")
plt.tight_layout()
plt.show()



# # Average price per room type in each Borough
nprice_room = cleandata.groupby(["Borough", "room_type"], as_index=False, observed=True).agg({'price': 'mean'})
price_room = nprice_room.pivot(index = 'Borough', columns = "room_type", values = "price")
print("\n\n\nAverage price per room type in each Borough :", nprice_room, 
       "\n\n\nAverage price per room type in each Borough: Pivot" , price_room , sep="\n")



# plot the average price within each borough
price_room.plot.bar(rot=0, cmap='tab20c', edgecolor="#2b2b28")
plt.xlabel("New York City Borough")
plt.ylabel("Average Price")
plt.title("Airbnb Listing in New York")
plt.tight_layout()
plt.show()


# ### A. Formulate research/analysis question(s)
# 
#        Question IV:
# 
#         - What are the major determinants of prices of rooms in New York Airbnb 
#             - Finding factors that positively and negatively impact the price of Airbnb rooms in NYC using linear regression analysis.
#             
# 


# prepare dataset for analysis
cleandata.info()


def preanalysis(datat):
    """The 'preanalysis' function replaces or drop missing, NAN or Na and
        get dummies from categorical features.
 
    Args: 
        'data1' (DataFrame): the DataFrame for data wrangling

 
    Returns: 
        DataFrame: The DataFrame for analysis
 
    """
    # replace nans with zero
    data = datat.copy()
    data[['security_deposit','cleaning_fee']] = data[['security_deposit','cleaning_fee']].fillna(value=0)
    data = data.dropna()

    dff = pd.get_dummies(data, prefix_sep="_")

    return dff



# prepare data for analysis
analysis_data = preanalysis(cleandata) 
print(analysis_data.info(), analysis_data.shape, sep="\n\n")



# split data for analysis and validation
analysis_data.dropna()
target = analysis_data.pop("logprice")
analysis_data.drop(columns=["price"], inplace=True)
X_train, X_test, y_train, y_test = train_test_split(analysis_data, target, test_size=0.2)



# explore the split data
print(X_train.head(), X_test.head(), y_train.head(), y_test.head(), sep="\n", end="\n\n")
print(X_train.shape,  y_train.shape, X_test.shape, y_test.shape, sep="\n")



# construct the data analysis pipeline
pl = Pipeline([
              ("scaler", StandardScaler()),
              ("radomForest", RandomForestRegressor(n_estimators = 200, n_jobs = -1,
                           oob_score = True, bootstrap = True))
            ])


# fit and evaluate the model
pl.fit(X_train, y_train)
pred = pl.predict(X_test)

print("mean_absolute_error : ", mean_absolute_error(y_test, pred))
print("mean_squared_error: ", mean_squared_error(y_test, pred))
print("Root mean_squared_error: ", np.sqrt(mean_squared_error(y_test, pred)))
print("R_squared : ", pl.score(X_test, y_test))
print("R2_squared : ", r2_score(pred, y_test))


# fit and evaluate complex models

estimatorstack = [
    ('Random Forest', RandomForestRegressor(random_state=342)),
    ('Lasso', LassoCV())
        ]


stacking_regressor = StackingRegressor(
    estimators=estimatorstack, final_estimator=RidgeCV())


estimators = [('OLS', LinearRegression()),
              ('Theil-Sen', TheilSenRegressor(random_state=682)),
              ('RANSAC', RANSACRegressor(random_state=9052)),
              ('HuberRegressor', HuberRegressor()),
              ("decisionTree", DecisionTreeRegressor()),
              ("radomForest", RandomForestRegressor(n_estimators = 200,
                           n_jobs = -1,
                           oob_score = True,
                           bootstrap = True,
                           random_state = 3452)), 
            ('Stacked Regressors', stacking_regressor),
            ("MLregs", MLPRegressor(hidden_layer_sizes=(100,100),
                                    tol=1e-2, max_iter=5000, random_state=670))
            ]

for name, estimator in estimators:
    model = make_pipeline(StandardScaler(), estimator)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(name," mean_absolute_error : ", mean_absolute_error(y_test, pred))
    print(name," mean_squared_error: ", mean_squared_error(y_test, pred))
    print(name," Root mean_squared_error: ", np.sqrt(mean_squared_error(y_test, pred)))
    print(name," R2_squared : ", r2_score(pred, y_test))
    print(name," R_squared : ", model.score(X_test, y_test), end="\n\n")
    


#     
# # Output
# 
#     OLS  mean_absolute_error :  0.13988029516539568
#     OLS  mean_squared_error:  0.03905320294442518
#     OLS  Root mean_squared_error:  0.1976188324639764
#     OLS  R2_squared :  0.28264443166503916
#     OLS  R_squared :  0.5779350193670276
# 
#     Theil-Sen  mean_absolute_error :  0.31642120613682606
#     Theil-Sen  mean_squared_error:  4.610522612751241
#     Theil-Sen  Root mean_squared_error:  2.147212754421704
#     Theil-Sen  R2_squared :  0.007562444329575646
#     Theil-Sen  R_squared :  -48.82792679074012
# 
#     RANSAC  mean_absolute_error :  0.36688845730633046
#     RANSAC  mean_squared_error:  162.8150648265272
#     RANSAC  Root mean_squared_error:  12.759900658959975
#     RANSAC  R2_squared :  -7.565071070692042e-06
#     RANSAC  R_squared :  -1758.6133479897806
# 
#     HuberRegressor  mean_absolute_error :  0.13893407537365612
#     HuberRegressor  mean_squared_error:  0.03946914126437717
#     HuberRegressor  Root mean_squared_error:  0.19866842040036753
#     HuberRegressor  R2_squared :  0.2860200025705454
#     HuberRegressor  R_squared :  0.5734397927090535
# 
#     decisionTree  mean_absolute_error :  0.17522363582111822
#     decisionTree  mean_squared_error:  0.060854076940186234
#     decisionTree  Root mean_squared_error:  0.24668619122315347
#     decisionTree  R2_squared :  0.35691306341919005
#     decisionTree  R_squared :  0.34232347493373655
# 
#     radomForest  mean_absolute_error :  0.12083424502413624
#     radomForest  mean_squared_error:  0.03006908430964762
#     radomForest  Root mean_squared_error:  0.17340439530083318
#     radomForest  R2_squared :  0.5311448560329223
#     radomForest  R_squared :  0.6750303040479735
# 
#     Stacked Regressors  mean_absolute_error :  0.12090605028442415
#     Stacked Regressors  mean_squared_error:  0.030230604673628945
#     Stacked Regressors  Root mean_squared_error:  0.17386950472589766
#     Stacked Regressors  R2_squared :  0.52810017116366
#     Stacked Regressors  R_squared :  0.6732846830961499
# 
#     MLregs  mean_absolute_error :  0.1337693352256908
#     MLregs  mean_squared_error:  0.035579289925210866
#     MLregs  Root mean_squared_error:  0.1886247330686272
#     MLregs  R2_squared :  0.49413721052259063
#     MLregs  R_squared :  0.6154791110324878
# 


# fit more models
# Training classifiers
# RandomForestRegressor outperformed even the complex models
gbr = GradientBoostingRegressor(n_estimators=200)
rfr = RandomForestRegressor(n_estimators=200)
lreg = LinearRegression()
vreg = VotingRegressor([('lr', lreg), ('gb', gbr), ('rf', rfr)])

estimators2 = [
    ("GradientBoostingRegressor", gbr),
    ("RandomForestRegressor", rfr),
    ("LinearRegression", lreg),
    ("VotingRegressor", vreg)
]

for name, estimator in estimators2:
    model = make_pipeline(StandardScaler(), estimator)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(name," mean_absolute_error : ", mean_absolute_error(y_test, pred))
    print(name," mean_squared_error: ", mean_squared_error(y_test, pred))
    print(name," Root mean_squared_error: ", np.sqrt(mean_squared_error(y_test, pred)))
    print(name," R2_squared : ", r2_score(pred, y_test))
    print(name," R_squared : ", model.score(X_test, y_test), end="\n\n")



# Gridsearch for model, and hyperparameter tuning

pipe = Pipeline([
              ("scaler", StandardScaler()),
              ("model", RandomForestRegressor())
            ])

grid_ = [{"model": [XGBRegressor()],
            "model__colsample_bytree": [0.8, 0.9],
            "model__max_depth": [4, 6],
            "model__n_estimators": [90, 100]},
        {"model": [GradientBoostingRegressor()],
        'model__learning_rate': [0.1, 0.01], 
        'model__max_features': [1.0, 0.5], 
        'model__n_estimators': [100, 150]},
        {"model": [RandomForestRegressor()],
            "model__ccp_alpha":  np.arange(0.0, 0.4, 0.05),
            "model__max_depth": [4, 6],
            "model__n_estimators": [90, 100, 150]

}]

gridsearcher = GridSearchCV(estimator=pipe, param_grid=grid_,
                       cv=5, verbose=1, n_jobs=-1)


best_model = gridsearcher.fit(X_train, y_train)

# # View best model
best_model.best_estimator_.get_params()['model']



# let get the feature importance
def dfform(lstt):
    """The dfform function form pandas dataframe from list
 
    Args: 
        lstt (list, series): the list or series to used in forming dataFrame

 
    Returns: 
        DataFrame: The DataFrame for analysis
 
    """
    df = pd.DataFrame(list(zip(X_test.columns, lstt)), columns=["features", "coefficients"])
    df3 = df.sort_values(by="coefficients", ascending=False).reset_index(drop=True)
#     df3.reset_index(drop=True,inplace=True)

    return df3

features_weight = dfform(pl[-1].feature_importances_)
features_weight.head()



# visualize the top 10 important feature affecting prices
top10 = features_weight[:10].sort_values(by="coefficients")
plt.barh(top10.features, top10.coefficients)
plt.xticks(rotation=45)
plt.axvline(x=0.05, color='red', linestyle='-')
plt.gcf().subplots_adjust(left=0.15)
plt.show()



# construct the data analysis pipeline
xgbpipe = Pipeline([
              ("scaler", StandardScaler()),
              ("XGBRegressor", best_model.best_estimator_.get_params()['model'])
            ])

xgbpipe.fit(X_train, y_train)


# visualize the top 10 important feature affecting prices using eli5 permutation
perm = PermutationImportance(xgbpipe).fit(X_test, y_test)

_imp_eli5 = dfform(perm.feature_importances_)

eli5_top10 = _imp_eli5.head(10).sort_values(by="coefficients")
plt.barh(eli5_top10.features, eli5_top10.coefficients)
plt.axvline(x=0.05, color='red', linestyle='-')
plt.xlabel("Importance Weight")
plt.title("Airbnb Listing in New York")
plt.gcf().subplots_adjust(left=0.15)
plt.savefig("plot.jpeg")
plt.show()

