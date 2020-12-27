# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 21:58:49 2018

@author: jweiss
"""
import time
import datetime
from bs4 import BeautifulSoup
from urllib import urlopen
import re
import pymc3 as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Yahoo Price Scraper 
def priceScraper(tickerList, start, end):
    columns = ['timetrade','open','High','Low','Close','Volume','adjClose']
    url_list = [str('https://finance.yahoo.com/quote/'+i+'/history?period1='
                + str(int(time.mktime(datetime.datetime.strptime(start, "%m/%d/%Y").timetuple())))
                +'&period2='+str(int(time.mktime(datetime.datetime.strptime(end, "%m/%d/%Y").timetuple())))
                +'&interval=1d&filter=history&frequency=1d' ) for i in tickerList]
    BeautifulSoup_List = [str(BeautifulSoup(i, 'html.parser')) for i in [urlopen(i) for i in url_list]]
    raw_data = [i[i.find("HistoricalPriceStore")+len("HistoricalPriceStore"):i.rfind("isPending")][13:len(i)-3] for i in BeautifulSoup_List]
    pandas_list = [pd.DataFrame([[i.split(',') for i in [re.sub('[^0-9\.\}\,]+','',i).split('},') 
                    for i in raw_data][j] if len(i) > 1] for j in range(len([re.sub('[^0-9\.\}\,]+','',i).split('},') 
                    for i in raw_data]))][k]) for k in range(len(raw_data))]
    for i in range(len(pandas_list)):
        pandas_list[i].columns = columns
        pandas_list[i] = pandas_list[i].dropna()
        pandas_list[i] = pandas_list[i][['adjClose']]
        pandas_list[i] = pandas_list[i].apply(pd.to_numeric)
    return pandas_list

#Enter Ticker and Dates
data = priceScraper(['AAPL'], '01/01/2016', '08/01/2018')[0][::-1]
returns = (data-data.shift(1))/data.shift(1)

#Plot Returns
returns.plot(figsize=(10, 6))
plt.ylabel('daily returns in %');

#Stochastic Volatillity Model
with pm.Model() as single_stock_model:
    nu = pm.Exponential('nu', 1/10., testval=5.)
    sigma = pm.Exponential('sigma', 1/0.02, testval=.1)
    s = pm.GaussianRandomWalk('s', sd=sigma, shape=len(returns))
    volatility_process = pm.Deterministic('volatility_process', pm.math.exp(-2*s)**0.5)
    r = pm.StudentT('r', nu=nu, sd=volatility_process, observed=returns)

#No-U-Turn Sampling
with single_stock_model:
    trace = pm.sample(2000)
pm.traceplot(trace, varnames=['nu', 'sigma']);

#Slice Sampling algo
with single_stock_model:
    step = pm.Slice()
    trace = pm.sample(2000, step=step)
pm.traceplot(trace, varnames=['nu', 'sigma']);

#Posterior Analysis
fig, ax = plt.subplots(figsize=(15, 8))
returns.plot(ax=ax)
ax.plot(returns.index, 1/np.exp(trace['s',::5].T), 'C3', alpha=.03);
ax.set(title='volatility_process', xlabel='time', ylabel='volatility');
ax.legend(['Stock', 'stochastic volatility process']);












