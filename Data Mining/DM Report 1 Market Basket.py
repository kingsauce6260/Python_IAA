#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 15:57:09 2019

@author: thomasgow
"""
#%%
import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import mlxtend as ml

#%%
restaurant = pd.read_csv(r"/Users/thomasgow/Documents/IAA/Data Mining/Data/Data/restaurantData.csv")
restaurant.head(8)

#%%
sns.countplot(x = 'order', data = restaurant, order = restaurant['order'].value_counts().iloc[:10].index)
sns.countplot(x = 'order', data = restaurant, order = restaurant['order'].value_counts().iloc[:10].index)
plt.xticks(rotation=90)
plt.show()

df = restaurant.groupby(['orderNumber','order']).size().reset_index(name='count')
basket = (df.groupby(['orderNumber', 'order'])['count']
          .sum().unstack().reset_index().fillna(0)
          .set_index('orderNumber'))
#The encoding function
def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1
basket_sets = basket.applymap(encode_units)



frequent_itemsets = apriori(basket_sets, min_support=0.01, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift")
rules.sort_values('confidence', ascending = False, inplace = True)
rules.head(10)

rules.to_csv('rules.csv')

restaurant_out = pd.read_csv("/Users/thomasgow/Documents/IAA/Data Mining/Data/meats.csv")
rules.to_csv('/Users/thomasgow/Documents/IAA/Data Mining/Data/restaraunt.csv')