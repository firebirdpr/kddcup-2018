# -*- coding: utf-8 -*-
"""
Created on Wed May  9 11:36:33 2018

@author: ponni
"""
import matplotlib.pyplot as plt
import pandas as pd #this is how I usually import pandas
import sys #only needed to determine Python version number
import matplotlib #only needed to determine Matplotlib version number
import numpy as np

# Enable inline plotting
%matplotlib inline

import re
from sklearn.linear_model import LinearRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from sklearn.metrics import r2_score
from sklearn.linear_model import Lasso
from sklearn import linear_model
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.ensemble import GradientBoostingRegressor

enc = OneHotEncoder()
le = preprocessing.LabelEncoder()
#set working directory
import os
os.chdir("J:/UWPCE/DataScience_SPR/kddcup2018-master/data")
os.getcwd()

#read file

ld_aq = pd.read_csv("ld_aq_2017-01-01-0-2018-05-18-23.csv")
ld_aq = pd.DataFrame(ld_aq)
ld_aq1.dtypes
type(ld_aq["time"][0])

ld_grid = pd.read_csv("ld_grid_2017-01-01-0-2018-05-18-23.csv")
ld_grid.head(5)
ld_grid = pd.DataFrame(ld_grid)
ld_grid2 = ld_grid

ld_met = pd.read_csv("ld_meteorology_2017-01-01-0-2018-05-18-23.csv")
ld_met.head(5)

ld_forecast = pd.read_csv("ldforecast_2018-05-31-00.csv")
ld_forecast = pd.DataFrame(ld_forecast)
col_names = list(ld_station)

ldX2 = pd.read_csv('ldX2.csv')
ldy = pd.read_csv('ldy.csv')
ld_station2 = pd.read_csv('ld_station2.csv')

#ld_station.rename(columns={'Unnamed: 0':'station'}, inplace=True )
ld_station2 = ld_station.loc[ld_station['need_prediction']== True,:]
ld3 = ld_station2.merge(ld_grid2,how='inner', left_on='grid', right_on= 'station_id')
ld3 = ld3.drop(['historical_data','Latitude','Longitude','SiteType','SiteName'], axis =1)
ld3 = ld3.drop(['grid','id'], axis =1)
ld3 = ld3.drop(['station_id'], axis =1)
ld3 = ld3.drop(['api_data','need_prediction'], axis =1)
ld_aq = ld_aq.drop(['CO_Concentration','O3_Concentration','SO2_Concentration'],axis =1)

ld_aqo3 = ld_aq.drop(['CO_Concentration','SO2_Concentration'],axis =1)

ld3 = ld3.sort_values(by = 'station_id_x')
ld_aq = ld_aq.sort_values(by = 'station_id')
ld_aqo3 = ld_aqo3.sort_values(by = 'station_id')


ld4 = ld3.merge(ld_aq, left_on = ['station_id_x','time'], right_on = ['station_id','time'], how = 'left')

ldO3 = ld3.merge(ld_aqo3, left_on = ['station_id_x','time'], right_on = ['station_id','time'], how = 'left')
#ld_BL0 = ld_aq.loc[ld_aq['station_id'] == 'BL0',:]
#ld3_BL0 = ld3.loc[ld3['station']== 'BL0', :]

#ld5 = ld3_BL0.merge(ld_BL0, left_on = 'time', right_on ='time', how = 'inner')

    

x = ld_aq["time"][0].split(" ")

def day_time(x):
    y = x.split(" ")
    day = int(y[0][8:10])
    hour = int(y[1][0:2])
    return day, hour

def day(x):
    y = x.split(" ")
    d = int(y[0][8:10])
    return d

def hour(x):
    y = x.split(" ")
    h = int(y[1][0:2])
    return h

ld4["day"] = ld4["time"].apply(day)
ldO3["day"] = ldO3["time"].apply(day)

ld4["hour"] = ld4["time"].apply(hour)
ldO3["hour"] = ldO3["time"].apply(hour)

cols_ld = list(ld4)

ldX = ld4.iloc[:,[0,3,4,5,6,7,14,15]]
ldX_no2 = ld4.iloc[:,[0,3,4,5,6,7,13,14,15]]

ldy = ld4.iloc[:,[11,12]]


#lr = LinearRegression()

#lr.fit(X_train, y_train)

#X_temp = le.transform(X["station_id"])

#X["station_id"] = enc.fit(X_temp)

ldX = ldX.drop('station_id_x', 1).join(
    pd.get_dummies(
        pd.DataFrame(ldX.station_id_x.tolist()).stack()
    ).astype(int).sum(level=0)
)
    
ldX2 = ldX.drop('weather', 1).join(
    pd.get_dummies(
        pd.DataFrame(ldX.weather.tolist()).stack()
    ).astype(int).sum(level=0)
)
    
ldX_no2 = ldX_no2.drop('station_id_x', 1).join(
    pd.get_dummies(
        pd.DataFrame(ldX_no2.station_id_x.tolist()).stack()
    ).astype(int).sum(level=0)
)
    
ldX_no2 = ldX_no2.drop('weather', 1).join(
    pd.get_dummies(
        pd.DataFrame(ldX_no2.weather.tolist()).stack()
    ).astype(int).sum(level=0)
)
    
#get column names
cols = list(X)
cols1 = cols
X = X.loc[:,cols1]

#replace na vqlues with mean

ldX2 = ldX2.fillna(ldX2.mean())
ldX_no2 = ldX_no2.fillna(ldX_no2.mean())
ldy = ldy.fillna(ldy.mean())
pd.isna(ldX2).sum().sum()
#split into train and test datasets

ldX_drop = ldX2.drop(['temperature','pressure','hour','CLEAR_DAY', 'CLEAR_NIGHT'], axis =1)
ldX_drop = ldX2.drop(['CLOUDY', 'PARTLY_CLOUDY_DAY', 'PARTLY_CLOUDY_NIGHT','RAIN','WIND'], axis =1)


ldX2 = pd.read_csv('ldX2.csv')
ldy = pd.read_csv('ldy.csv')
ldX_train,ldX_test, ldy_train, ldy_test = train_test_split(ldX2, ldy, test_size=0.2, random_state=42)

ldX_train = ldX_train.reset_index()
ldX_train = ldX_train.drop(['index'],axis =1)

ldy_train = ldy_train.reset_index()
ldy_train = ldy_train.drop(['index'],axis =1)

ldX_test = ldX_test.reset_index()
ldX_test = ldX_test.drop(['index'],axis =1)

ldy_test = ldy_test.reset_index()
ldy_test = ldy_test.drop(['index'],axis =1)

#scale data


#lasso model
alpha = 0.1
lasso = Lasso(alpha=alpha)

lasso.fit(X_train, y_train)

y_pred_lasso = lasso.predict(X_test)
r2_score_lasso = r2_score(y_test, y_pred_lasso)
print(lasso)
print("r^2 on test data : %f" % r2_score_lasso)

lasso.coef_[1]
len(lasso.coef_[0])
len(list(X_test))
#plt.scatter(y_test, y_pred_lasso)

mlp = MLPRegressor()
mlp.fit(X_train,y_train)

predictedPM25 = mlp.predict(X_test)
r2_mlp_pm25 = r2_score(y_test,predictedPM25)
print(r2_mlp_pm25)

ldX_train = ldX_train.drop(['index'],axis =1)
ldy_train = ldy_train.drop(['index'],axis =1 )
#Multioutput
max_depth = 30
ld_multirf = MultiOutputRegressor(RandomForestRegressor(max_depth = max_depth,random_state=0))
ld_multirf1 = RandomForestRegressor(max_depth = max_depth,random_state=0)
ld_multirf1.fit(ldX_train, ldy_train)

ld_col1=list(ld_multirf1.feature_importances_)
ld_multirf1.get_params

ld_col = list(ldX2)
mlp = MLPRegressor(hidden_layer_sizes=(1000))
ld_multimlp = MultiOutputRegressor(mlp)
ld_multimlp.fit(ldX_train, ldy_train)

ld_ada = AdaBoostRegressor(DecisionTreeRegressor(max_depth=50),
                          n_estimators=300, random_state=0)

ld_multiada = MultiOutputRegressor(ld_ada)


ld_multiada.fit(ldX_train, ldy_train)

ld_GBR = MultiOutputRegressor(GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
    max_depth=30, random_state=0, loss='ls')).fit(ldX_train, ldy_train)

param_grid = {"max_depth": [3, None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap" :[True,False]}

grid_search = GridSearchCV(ld_multirf1, param_grid=param_grid)
grid_search.fit(ldX_train, ldy_train)
ldy_grid_search = grid_search.predict(ldX_test)


scaler = MinMaxScaler(feature_range=(0, 1))
b = scaler.fit_transform(ldy2_test)
inv_ldyhat = scaler.inverse_transform(ldy_grid_search)

inv_ldyhat = scalef(ldy2_test,inv_ldyhat)
#grid_search1 = GridSearchCV(ld_multirf1, param_grid=param_grid)
#grid_search1.fit(ldX_train, ldy_train.iloc[:,1])
3ld_multirf1.get_params().keys()


regr_multimlp.get_params
ld_multirf1.get_params

# Predict on new data
ldy_multirf = ld_multirf.predict(ldX_test)
ldy_multimlp = ld_multimlp.predict(ldX_test)
ldy_multiada = ld_multiada.predict(ldX_test)
ldy_grid_search = grid_search.predict(ldX_test)
grid_search.score(ldX_test, ldy_test)
grid_search1.score(ldX_test, ldy_test.iloc[:,1])

ldysc_multirf = ld_multirf.predict(ldX_testsc)
ldysc_multimlp = ld_multimlp.predict(ldX_testsc)

scl = MinMaxScaler()
a = scl.fit_transform(ldy_testsc)

ldX_test = ldX_test.drop(['index'],axis =1)
ldy_test = ldy_test.drop(['index'], axis =1)
ldyrf_pred = scl.inverse_transform(ldysc_multirf)
ldyml_pred = scl.inverse_transform(ldysc_multimlp)
rf_mae =mean_absolute_error(ldy_test, ldy_multirf, multioutput='raw_values')
rf_r2 = r2_score(ldy_test, ldy_multirf, multioutput='raw_values')
ml_mae = mean_absolute_error(ldy_test, ldy_multimlp, multioutput='raw_values')
ada_mae = mean_absolute_error(ldy_test, ldy_multiada, multioutput='raw_values')
gbr_mae = mean_absolute_error(ldy_test, ld_GBR.predict(ldX_test), multioutput='raw_values')
ml_r2 = r2_score(ldy_test, ldy_multimlp, multioutput='raw_values')
ml_r2 = r2_score(ldy_test, ldyml_pred, multioutput='raw_values')
ada_r2 = r2_score(ldy_test, ldy_multiada, multioutput='raw_values')
gbr_r2 = r2_score(ldy_test, ld_GBR.predict(ldX_test), multioutput='raw_values')
# Plot the results
plt.figure()
s = 50
a = 0.4
plt.scatter(ldy_test.iloc[:, 0], ldy_test.iloc[:, 1], edgecolor='k',
            c="red", s=s, marker="s", alpha=a, label="Data")
plt.scatter(ldy_multirf[:, 0], ldy_multirf[:, 1], edgecolor='k',
            c="cornflowerblue", s=s, alpha=a,
            label="Multi RF score=%.2f" % ld_multirf.score(ldX_test, ldy_test))
plt.scatter(ldy_grid_search[:, 0], ldy_grid_search[:, 1], edgecolor='k',
            c="green", s=s, marker="^", alpha=a,
            label="GS score=%.2f" % grid_search.score(ldX_test, ldy_test))
plt.xlim([1, 100])
plt.ylim([1, 100])
plt.xlabel("target 1")
plt.ylabel("target 2")
plt.title("Comparing random forests and adaboost estimator")
plt.legend()
plt.show()

#The code below is for submitting the file in the right format

ld_aq1 = pd.read_csv('J:/UWPCE/DataScience_SPR/kddcup2018-master/data/ld_airquality_2018-04-04-0_2018-04-04-23.csv')

ld_aq1 = pd.DataFrame(ld_aq1)

ld_forecast = pd.concat([ld_aq1, ld_aq1], axis =0, ignore_index = True)

ld_forecast["day"][0:457] = int(21)
ld_forecast["day"][457:912] = int(22)

ld_forecast = ld_forecast.drop('station_id', 1).join(
    pd.get_dummies(
        pd.DataFrame(ld_forecast.station_id.tolist()).stack()
    ).astype(int).sum(level=0)
)
    
ld_forecast = ld_forecast.drop(['id','time','PM25_Concentration','CO_Concentration',
                        'O3_Concentration','SO2_Concentration'],axis =1)

ld_forecast.fillna(ld_forecast.mean())


X_forecast = pd.DataFrame(ld_aq)

ld_stations = ['BL0','BX1','BX9','CD1','CD9','CT2','CT3','GN0','GN3','GR4','GR9',
 'HV1',
 'KF1',
 'LW2',
 'MY7',
 'RB7',
 'ST5',
 'TD5',
 'TH4']

len(ld_stations)
x = 19*47




ld_forecast = ld_forecast.loc[:,cols]

sum(pd.isna(ld_forecast))

ld_forecast["PM10_Concentration"] = ld_forecast["PM10_Concentration"].fillna(ld_forecast["PM10_Concentration"].mean())

ld_forecast["NO2_Concentration"] = ld_forecast["NO2_Concentration"].fillna(ld_forecast["NO2_Concentration"].mean())

testld_forecast = lasso.predict(ld_forecast)
testld_forecast

testld_forecast = pd.DataFrame(testld_forecast)

ldf1 = pd.concat([ld_forecast,testld_forecast], axis =1)



cols2 = list(ldf1)

ldf1 = ldf1.rename(columns = {0 : 'PM25_Concentration'})
list(ldf1)

ldf1_a = ldf1.loc[:, ['BL0','PM25_Concentration']]

list(ldf1_a)

ldf1_a = ldf1_a.loc[lambda ldf1_a: ldf1_a.BL0 ==1]
ldf1_a

new_x2 = pd.DataFrame()

cols5 = []

for i in cols4:
    cols5.append(i.strip)

for i in cols5:
    x = ldf1.loc[:, ['i','PM25_Concentration', 'PM10_Concentration']]
    new_x = x.loc[lambda x: x.i ==1]
    new_x2 = pd.concat([new_x2,new_x], axis =1)


def pull(x,y):
    x = ldf1.loc[:, [y,'PM25_Concentration', 'PM10_Concentration']]
    x = x.loc[lambda x: x.loc[:,y] ==1]
    x = x.rename(columns = {'BL0' : 'test_id'})
    return x

BL0 =pull('bl0', 'BL0')

count = 0
def pull2(x,y):
    x = ldf1.loc[:, [y,'PM25_Concentration', 'PM10_Concentration']]
    x = x.loc[lambda x: x.loc[:,y] ==1]
    x = x.rename(columns = {y : 'test_id'})
    count = 0
    for i in x.index:
        x.loc[i,'test_id'] = y +'_aq'+'#'+ str(count)
        count = count + 1
    return x

BL0 = pull2('bl0','BL0')
BX1 = pull2('bx1','BX1')

df_list = pd.DataFrame()
cols4 = list(ldf1)[0:19]
for j in cols4:
        a = pull2(x,j)
        df_list = pd.concat([df_list,a], axis = 0)
        
def df_merge():
    cols4 = list(ldf1)[0:19]
    df_list = pd.DataFrame()
    for j in cols4:
        a = pull2(x,j)
        df_list = pd.concat([df_list,a], axis = 0)
    return df_list

a = df_merge()
#hour_list = list(range(0,48))
pd.options.mode.chained_assignment = None  # default='warn'

count = 0
newDF = pd.DataFrame()
test = pd.DataFrame(index = ind, columns = ['test_id'])
for i in test.index:
    test.loc[i] = 'BL0'+'_aq'+'#'+ str(count)
    count = count + 1
newDF = pd.concat([ldf1_a,test], axis =1)

cols3 = ['test_id', 'PM25_Concentration']

newDF = newDF.loc[:,cols3]

ldX2_csv = ldX2.to_csv(path_or_buf='J:/UWPCE/DataScience_SPR/kddcup2018-master/data/ldX2.csv', index = False)
ldy_csv = ldy.to_csv(path_or_buf='J:/UWPCE/DataScience_SPR/kddcup2018-master/data/ldy.csv', index = False)

ld_station2_csv = ld_station2.to_csv(path_or_buf='J:/UWPCE/DataScience_SPR/kddcup2018-master/data/ld_station2.csv', index = False)


import os
os.getcwd()
