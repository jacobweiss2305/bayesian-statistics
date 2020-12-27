# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 08:58:29 2018

@author: jweiss
"""

import pymc3 as pm
from pymc3 import summary
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

data = pd.read_excel('C:\Users\jweiss\Desktop\modified_sleep.xlsx')
column_names = list(data)
#create histograms of orginal data


#WARNING: Takes a few hours to run with sample 100000
#impute random variable for missing data per column
data_frames_imputation = []
for i in column_names:
    with pm.Model() as model:
        mu = pm.Normal('mu', mu=data[i].mean(), sd=data[i].std())
        sd = pm.HalfNormal('sd', sd=1)
        n = pm.Normal('n', mu=mu, sd=sd, observed=data[i])
        start = pm.find_MAP()
        step = pm.Metropolis(start=start)
        trace = pm.sample(100000, step, start=start, progressbar=True)   
    pm.traceplot(trace)
    data_frames_imputation.append(summary(trace)['mean'])

#replacing missing values from Bayesian model
impute_values = [list(i)[1:-1] for i in data_frames_imputation]
new_dataframe = []
for z in range(len(column_names)):  
    orginal_data = list(data[column_names[z]])
    replace_index = [n for n,i in enumerate(orginal_data) if math.isnan(i) == True]
    for index in range(len(replace_index)):
        orginal_data[replace_index[index]] = impute_values[z][index]    
    new_dataframe.append(orginal_data)
new_data = pd.DataFrame(new_dataframe).T

#lag calories burned and drop calories burned
new_data.columns = column_names
new_data['lagCaloriesBurned'] = new_data['Calories Burned'].shift(-1)
new_data = new_data[np.isfinite(new_data['lagCaloriesBurned'])].drop(['Calories Burned'], axis=1)
new_data['Sleep Quality'] = new_data['Minutes Asleep']/new_data['Time in Bed']

writer = pd.ExcelWriter('C:/Users/jweiss/Desktop/new_data.xlsx', engine='xlsxwriter')
new_data.to_excel(writer)
writer.save()

