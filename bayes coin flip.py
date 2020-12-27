# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 21:17:06 2018

@author: jweiss
"""

#Bayesian Approach to 10 coin flips

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate


plt.figure()
p = np.arange(0,1,.001)
pl0t = lambda p: 11* (1-p)**10
plt.plot(p,pl0t(p), label = 'density')

#mean
area = scipy.integrate.quad(pl0t,0,1) #posterior
print(area)

#median
is05 = scipy.integrate.quad(pl0t,0,.06)











