# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 15:26:23 2018

@author: jweiss
"""

#metropolis algo

import numpy as np
import scipy.stats as stats
from scipy.stats import gamma
import seaborn as sns
import matplotlib.pyplot as plt

import pymc3 as pm
from pymc3 import df_summary

x = np.random.normal(0,10,50)

with pm.Model() as model:
    #prior, only unknown parameter is mu
    sd = pm.Gamma('sd', mu = .5, sd =1)
    #unknown model
    observations = pm.Normal('obs', mu= 0, sd = 1/sd, observed = -2)
    

with model:
    trace = pm.sample(1000, tune = 500, njobs = 4)

pm.traceplot(trace)
pass    
    

    
    
a = 2
b = 3

z1 = 11
N1 = 14
z2 = 7
N2 = 14


import numpy as np
import scipy.stats as stats
from scipy.stats import gamma
import seaborn as sns
import matplotlib.pyplot as plt

import pymc3 as pm
from pymc3 import df_summary

def bern(theta, z, N):
    """Bernoulli likelihood with N trials and z successes."""
    return np.clip(theta**z * (1-theta)**(N-z), 0, 1)


def bern2(theta1, theta2, z1, z2, N1, N2):
    """Bernoulli likelihood with N trials and z successes."""
    return bern(theta1, z1, N1) * bern(theta2, z2, N2)


def make_thetas(xmin, xmax, n):
    xs = np.linspace(xmin, xmax, n)
    widths =(xs[1:] - xs[:-1])/2.0
    thetas = xs[:-1]+ widths
    return thetas

from mpl_toolkits.mplot3d import Axes3D

def make_plots(X, Y, prior, likelihood, posterior, projection=None):
    fig, ax = plt.subplots(1,3, subplot_kw=dict(projection=projection, aspect='equal'), figsize=(12,3))
    if projection == '3d':
        ax[0].plot_surface(X, Y, prior, alpha=0.3, cmap=plt.cm.jet)
        ax[1].plot_surface(X, Y, likelihood, alpha=0.3, cmap=plt.cm.jet)
        ax[2].plot_surface(X, Y, posterior, alpha=0.3, cmap=plt.cm.jet)
        for ax_ in ax: ax_._axis3don = False
    else:
        ax[0].contour(X, Y, prior, cmap=plt.cm.jet)
        ax[1].contour(X, Y, likelihood, cmap=plt.cm.jet)
        ax[2].contour(X, Y, posterior, cmap=plt.cm.jet)
    ax[0].set_title('Prior')
    ax[1].set_title('Likelihood')
    ax[2].set_title('Posteior')
    plt.tight_layout()

thetas1 = make_thetas(0, 1, 101)
thetas2 = make_thetas(0, 1, 101)
X, Y = np.meshgrid(thetas1, thetas2)

a = 3.4
b = 5.2

z1 = 11
N1 = 54
z2 = 7
N2 = 54


prior = stats.beta(a, b).pdf(X) * stats.beta(a, b).pdf(Y)
likelihood = bern2(X, Y, z1, z2, N1, N2)
posterior = stats.beta(a + z1, b + N1 - z1).pdf(X) * stats.beta(a + z2, b + N2 - z2).pdf(Y)
make_plots(X, Y, prior, likelihood, posterior)
make_plots(X, Y, prior, likelihood, posterior, projection='3d')

a = 3.4
b = 5.2
n = 54
k = 2

f(k|n,a,b) = comb(n,k) * stats.beta(k+a, n-k+b) / stats.beta(a,b)



