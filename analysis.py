import os
import numpy as np
import pandas as pd
import pickle
import re


mydir = r"D:\project1"

def des(*datas):
    for data in datas:
        name = [x for x in globals() if globals()[x] is data][0]
        print(f"#########################  Printing for {name} ######################\n"*5)
        print(data.head(), data.info(), data.shape, data.describe(), data.columns, sep="\n\n", end="\n\n")
        
def mmshape(*datat):
    for data in datat:
        name = [x for x in globals() if globals()[x] is data][0]
        print(f"{name} has dimension : {data.shape}", sep="\n\n")
        
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
    
sp = {"sep":"\n\n", "end":"\n\n"}

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
print(pandas_pickle.head())