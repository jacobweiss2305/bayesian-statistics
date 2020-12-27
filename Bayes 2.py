# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 06:50:30 2018

@author: jweiss
"""

import pymc3 as pm
import theano.tensor as tt

# Recall count_data is the variable that holds our txt counts
with pm.Model() as model:
    alpha = 1.0/count_data.mean()  
    lambda_1 = pm.Exponential("lambda_1", alpha)
    lambda_2 = pm.Exponential("lambda_2", alpha)
    
    tau = pm.DiscreteUniform("tau", lower=0, upper=n_count_data - 1)
    
    
with model:
    idx = np.arange(n_count_data) # Index
    lambda_ = pm.math.switch(tau > idx, lambda_1, lambda_2)    
    
    
with model:
    observation = pm.Poisson("obs", lambda_, observed=count_data)    
    
    
with model:
    step = pm.Metropolis()
    trace = pm.sample(10000, tune=5000,step=step)    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    