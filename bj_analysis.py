# -*- coding: utf-8 -*-
"""
Created on Wed May 16 13:07:06 2018

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
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
import pickle

import tensorflow as tf
enc = OneHotEncoder()
le = preprocessing.LabelEncoder()
#set working directory
import os
os.chdir("J:/UWPCE/DataScience_SPR/kddcup2018-master/data")

bj_met = pd.read_csv("bj_meteorology_2017-01-01-0-2018-05-18-23.csv")
bj_met = pd.DataFrame(bj_met)
bj_met.dtypes
bj_met.shape
bj_met[1:10]

bj_grid = pd.read_csv("bj_grid_2017-01-01-0-2018-05-18-23.csv")
bj_grid = pd.DataFrame(bj_grid)

bj_aq = pd.read_csv("bj_aq_2017-01-01-0-2018-05-18-23.csv")
bj_aq = pd.DataFrame(bj_aq)
bj_aq[1:10]
bj_grid[1:10]
bj_met

bj_forecast = pd.read_csv("bjforecast_2018-05-31-00.csv")
bj_forecast = pd.DataFrame(bj_forecast)

bjX2 = pd.read_csv('bjX2.csv')
bjX2 = pd.DataFrame(bjX2)

bjy = pd.read_csv('bjy.csv')
bjy = pd.DataFrame(bjy)

#bj_met2 =  pd.read_csv("J:/UWPCE/DataScience_SPR/kddcup2018-master/data/web_data/beijing_17_18_meo.csv")
#bj_met2 = pd.DataFrame(bj_met2)
#bj_met2[1:10]

#bj_lat = pd.read_excel('Beijing_grid_weather_station.xlsx')
#bj_lat = pd.DataFrame(bj_lat)
#bj_lat[1:10]

#list(bj_grid)
#x = bj_grid.loc[lambda bj_grid: bj_grid.loc[:,'station_id'] =='beijing_grid_000']


bj_stat2grid = bj_stations.merge(bj_grid,how='inner', left_on='grid', right_on= 'station_id')


bj_stat2grid = bj_stat2grid.sort_values(by = 'station_id_x')
bj_aq = bj_aq.sort_values(by = 'station_id')


bj_aqw = bj_stat2grid.merge(bj_aq, left_on = ['station_id_x','time'], right_on = ['station_id','time'], how = 'left')

bj_aqw = bj_aqw.rename(columns = {'station_id_x': 'station_id'})

def day(x):
    y = x.split(" ")
    d = int(y[0][8:10])
    return d

def hour(x):
    y = x.split(" ")
    h = int(y[1][0:2])
    return h

bj_aqw["day"] = bj_aqw["time"].apply(day)

bj_aqw["hour"] = bj_aqw["time"].apply(hour)

cols_bj = list(bj_aqw)
X_bj = bj_aqw.iloc[:,[0,7,8,9,10,11,12,21,22]]

y_bj = bj_aqw.iloc[:,[15,16,19]]

X_bj = X_bj.drop('station_id', 1).join(
    pd.get_dummies(
        pd.DataFrame(X_bj.station_id.tolist()).stack()
    ).astype(int).sum(level=0)
)
    
X2_bj = X_bj.drop('weather', 1).join(
    pd.get_dummies(
        pd.DataFrame(X_bj.weather.tolist()).stack()
    ).astype(int).sum(level=0)
)
    
X2_bj = X2_bj.fillna(X2_bj.mean())
y_bj = y_bj.fillna(y_bj.mean())
pd.isna(X2_bj).sum().sum()
#split into train and test datasets

bjX2_trim = bjX2.iloc[:,0:42]
bjX2_trim['WIND'] = bjX2['WIND']
bjX_train,bjX_test, bjy_train, bjy_test = train_test_split(bjX2_trim, bjy, test_size=0.2, random_state=42)

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
plt.scatter(y_test, y_pred_lasso)

mlp = MLPRegressor()
mlp.fit(X_train,y_train)

predictedPM25 = mlp.predict(X_test)
r2_mlp_pm25 = r2_score(y_test,predictedPM25)
print(r2_mlp_pm25)

#Multioutput
max_depth = 30
bj_multirf = RandomForestRegressor(max_depth=max_depth,random_state=0)


bj_multirf.fit(bjX_train, bjy_train)

bj_col1 = list(bj_multirf.feature_importances_)
bj_col = list(bjX2_train)
bj_ada = AdaBoostRegressor(DecisionTreeRegressor(max_depth=30),
                          n_estimators=300, random_state=0)

bj_multiada = MultiOutputRegressor(bj_ada)


bj_multiada.fit(bjX_train, bjy_train)

bj_GBR =MultiOutputRegressor(GradientBoostingRegressor(n_estimators=50, learning_rate=0.1,
    max_depth=10, random_state=0, loss='ls')).fit(bjX_train, bjy_train) 

fn = 'bj_GBR.sav'
pickle.dump(bj_GBR, open(fn, 'wb'))

regr_multimlp = MultiOutputRegressor(mlp)
regr_multimlp.fit(X_train, y_train)

regr_multimlp.get_params
regr_multirf.get_params

# Predict on new data
bjy_multirf = bj_multirf.predict(bjX_test)
y_multimlp = regr_multimlp.predict(X_test)
bjy_multiada = bj_multiada.predict(bjX_test)

# Plot the results
plt.figure()
s = 50
a = 0.4
plt.scatter(y_test.iloc[:, 0], y_test.iloc[:, 1], edgecolor='k',
            c="red", s=s, marker="s", alpha=a, label="Data")
plt.scatter(y_multirf[:, 0], y_multirf[:, 1], edgecolor='k',
            c="cornflowerblue", s=s, alpha=a,
            label="Multi RF score=%.2f" % regr_multirf.score(X_test, y_test))
plt.scatter(y_multimlp[:, 0], y_multimlp[:, 1], edgecolor='k',
            c="green", s=s, marker="^", alpha=a,
            label="MLP score=%.2f" % regr_multimlp.score(X_test, y_test))
plt.xlim([-10, 400])
plt.ylim([-10, 400])
plt.xlabel("target 1")
plt.ylabel("target 2")
plt.title("Comparing random forests and the multi-output mlp estimator")
plt.legend()
plt.show()

bjX2_csv = X2_bj.to_csv(path_or_buf='J:/UWPCE/DataScience_SPR/kddcup2018-master/data/bjX2.csv', index = False)
bjy_csv = y_bj.to_csv(path_or_buf='J:/UWPCE/DataScience_SPR/kddcup2018-master/data/bjy.csv', index = False)

bj_stations_csv = bj_stations.to_csv(path_or_buf='J:/UWPCE/DataScience_SPR/kddcup2018-master/data/bj_stations.csv', index=False)
