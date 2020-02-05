#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 18:47:35 2019

@author: thomasgow
"""

#%%
import pandas as pd
import lifelines
import matplotlib.pyplot as plt


#%%
data = pd.read_sas(r"/Users/thomasgow/Documents/IAA/Survival Analysis/SA Data/hurricane.sas7bdat")

#%%
# Python code to create the above Kaplan Meier curve
time = data['hour']
event= data['survive']


## create a kmf object
kmf = lifelines.KaplanMeierFitter() 

## Fit the data into the model
kmf.fit(time, event==0,label='Kaplan Meier Estimate')

## Create an estimate
kmf.plot(ci_show=False) ## ci_show is meant for Confidence interval, since our data set is too tiny, thus i am not showing it.



#%% Plotting stratified reasons
ax = plt.subplot(111)

kmf = lifelines.KaplanMeierFitter()

for name, grouped_df in data.groupby('reason'):
    kmf.fit(grouped_df["hour"], grouped_df["survive"]==0, label=name)
    kmf.plot(ax=ax, ci_show=False)
    
    
    



#%%
    ## Import libraries
from lifelines.statistics import multivariate_logrank_test
from matplotlib.offsetbox import AnchoredText

## Set colour dictionary for consistent colour coding of KM curves
colours = {'Yes':'g', 'No':'r'}
        
## Set up subplot grid
fig, axes = plt.subplots(nrows = 6, ncols = 3, 
                         sharex = False, sharey = False,
                         figsize=(20, 35))

## Plot KM curve for each categorical variable
def categorical_km_curves(feature, t='hour', event='survive', df=data, ax=None):
    for cat in sorted(data[feature].unique(), reverse=True):
        idx = data[feature] == cat
        kmf = lifelines.KaplanMeierFitter()
        kmf.fit(data[idx][t], event_observed=data[idx][event]==0, label=cat)
        kmf.plot(ax=ax, label=cat, ci_show=False, c=colours[cat])

col_list = data[["backup", "bridgecrane", "elevation", "gear", "servo", "slope", "trashrack"]]

for cat, ax in zip(col_list, axes.flatten()):
    categorical_km_curves(feature=cat, t='hour', event=data['survive'], df = data, ax=ax)
    ax.legend(loc='lower left', prop=dict(size=18))
    ax.set_title(cat, fontsize=18)
    p = multivariate_logrank_test(data['time'], data[cat], data['survive'])
    ax.add_artist(AnchoredText(p.p_value, loc=4, frameon=False))
    ax.set_xlabel('Time')
    ax.set_ylabel('Survival probability')
    
fig.subplots_adjust(top=0.92)
