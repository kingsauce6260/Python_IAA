# -*- coding: utf-8 -*-
"""

Title: Dark Sky API

Author: Thomas Gow

"""

import pandas as pd
import sys
sys.path.insert( 1, '/Users/thomasgow/IAAPython/DarkySkyAPI' )

from forecastiopy import *

#Importing day of week so forecast adjusts to the day
import datetime
datetime.datetime.today()
today=datetime.datetime.today().weekday()

api_key = '1d3d7504309291be4ca968c86f48d5be'

#Creating dict for locations
loc_dict = {'City': ['Anchorage, Alaska', 'Buenos Aires, Argentia',
            'São José dos Campos, Brazil', 'San José, Costa Rica',
            'Nanaimo, Canada','Ningbo, China','Giza, Egypt',
            'Mannheim, Germany', 'Hyderabad, India','Tehran, Iran',
            'Bishkek, Kyrgyzstan','Riga, Latvia','Quetta, Pakistan',
            'Warsaw, Poland','Dhahran, Saudia Arabia','Madrid, Spain',
            'Oldham, United Kingdom'],
            'lat':[61.2181,-34.6037,-23.2237,9.9281,49.1659,29.8683,
                   30.0131,49.4875,17.3850,35.6892,42.8746,56.9496,
                   30.1798,52.2297,26.2361,40.4168,53.5409],
            'long':[-149.9003,-58.3816,-45.9009,-84.0907,-123.9401,
                    121.5440,31.2089,8.4660,78.4867,51.3890,74.5698,
                    24.1052,66.9750,21.0122,50.0393,-3.7038,-2.1114],}
df= pd.DataFrame(loc_dict, columns=['City', 'lat','long', 'Min_1'])
#City,Min 1,Max 1, ... Min 5,Max 5,Min Avg,Max Avg

#Using two foreloops creating min and max for each day
for label, row in df.iterrows():
    weather=ForecastIO.ForecastIO( api_key, units='si', latitude=row['lat'], longitude=row['long'] )
    daily = FIODaily.FIODaily(weather)
    for day in range(2,7): #Note to create date workaround by putting if statement and change this to an i
        val=daily.get(day)
        if day==2:
            df.loc[label,'Min_1']= val['temperatureMin']
            df.loc[label,'Max_1']= val['temperatureMax']
        if day==3:
            df.loc[label,'Min_2']= val['temperatureMin']
            df.loc[label,'Max_2']= val['temperatureMax']
        if day==4:
            df.loc[label,'Min_3']= val['temperatureMin']
            df.loc[label,'Max_3']= val['temperatureMax']
        if day==5:
            df.loc[label,'Min_4']= val['temperatureMin']
            df.loc[label,'Max_4']= val['temperatureMax']
        if day==6:
            df.loc[label,'Min_5']= val['temperatureMin']
            df.loc[label,'Max_5']= val['temperatureMax']

        
#Selecting column for averages for both min and max
df.mean(axis = 1, skipna = True) 
col_Min = df.loc[: , ["Min_1", "Min_2", "Min_3", "Min_4", "Min_5"]]
col_Max = df.loc[: , ["Max_1", "Max_2", "Max_3", "Max_4", "Max_5"]]

#Calculating and adding min and max averages to their own columns in df
df['Min_Avg'] = col_Min.mean(axis=1)
df['Max_Avg'] = col_Max.mean(axis=1)
df
final_df=df.drop(['lat','long'], axis=1)
final_df['Min_Avg']=final_df.Min_Avg.round(2)
final_df['Max_Avg']=final_df.Max_Avg.round(2)

final_df.to_csv (r'/Users/thomasgow/IAAPython/DarkySkyAPI/export_darkskyAPI_Gow.csv',
           index = None, header=True) #Don't forget to add '.csv' at the end of the path
