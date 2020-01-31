#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 13:49:06 2019

@author: root
"""

#%%
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(random_state = 42)
from pprint import pprint
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
import pandas as pd

#%%
#%% Data
train = pd.read_csv(r"/Users/thomasgow/Documents/IAA/Machine Learning/Competition/MLProject_train.csv", engine='python')
valid = pd.read_csv(r"/Users/thomasgow/Documents/IAA/Machine Learning/Competition/MLProject_valid.csv")

#%%
train = train.drop(columns=['target1'])
valid = valid.drop(columns=['target1'])

#%%
train.dropna(inplace=True)
valid.dropna(inplace=True)
sample = train.sample(n = 10000)

X_sample = sample.drop(columns=['target2'])
X_train = train.drop(columns=['target2'])
X_test = valid.drop(columns=['target2'])
y_sample = sample['target2']
y_train = train['target2']
y_test = valid['target2']


#%%
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, random_state=42)
rf.fit(X_sample, y_sample)

y_pred = rf.predict(X_test)

#%%
for name, score in zip(X_sample.columns, rf.feature_importances_):
    print(name, score)

#%%
# Calculate the accuracy
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)
