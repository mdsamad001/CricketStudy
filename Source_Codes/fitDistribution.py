#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 10:44:09 2019

@author: mdsamad
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
import Source_Codes.Conti_CDF as contiCDF

# Return the distribution with lowest fitting error
def BestDistribution (df):
    
    
    y = df.fInn
    
    h, bn = np.histogram(y, bins=10)
    bn = (bn + np.roll(bn, -1))[:-1] / 2.0
    
    dist_names = ['alpha', 'expon','logistic','exponnorm',
               'rayleigh', 'norm', 'invgauss','gamma']

    allSSE = []
    for dist_name in dist_names:
        
        dist = getattr(ss, dist_name)
        param = dist.fit(y)
      
        yfit = dist.pdf(bn,*param[:-2], loc=param[-2], scale=param[-1])
    
    
        sse = np.sum(np.power(yfit - h/len(y), 2.0))
    
        allSSE.append(sse)

    indx = allSSE.index(min(allSSE))
    
    print ('Best Distribution is: ',dist_names[indx])
    
    # returning the normal distribution, otherwise use 'indx' in place
    # of 5
    return dist_names[2]



# Plot all distributions, their fitting error along with the actual distribution
def fitModelAna (df):
    
    y = df.fInn 
    x = np.arange(400)
    
    # Plot histogram of the raw data    
    plt.hist(y, normed=1,bins=10,alpha=0.3)
    
    # Histogram values and bin ranges 
    h, bn = np.histogram(y, bins=10)
    
    
    # X-value for each bin is the mid value of the bin range
    bn = (bn + np.roll(bn, -1))[:-1] / 2.0
    
    
    
    # A set of well known distribution functions 
    
    dist_names = ['alpha', 'nbinom','logistic','exponnorm',
                   'rayleigh', 'norm', 'invgauss','gamma']
    
    allSSE = []
    
    
    for dist_name in dist_names:
        
        if dist_name != 'nbinom':
            
            dist = getattr(ss, dist_name)
            
            # Fit model with raw data
            param = dist.fit(y)
            pdf_fitted = dist.pdf(x, *param[:-2], loc=param[-2], scale=param[-1])#* size
            
            # Plot the fitted distribution on raw data
            plt.plot(pdf_fitted, label=dist_name)
            plt.xlim(0,400)
            
            # Distribution values for the histogram bin mid-values to be 
            # able to compare fitted values with the histogram values
            
            yfit = dist.pdf(bn,*param[:-2], loc=param[-2], scale=param[-1])
            
            
            # Calculate sum squared values between fitted probability and 
            # actual probability obtained from histogram (y/sample_size)
            sse = np.sum(np.power(yfit - h/len(y), 2.0))
            
            allSSE.append(sse)
            
            print(dist_name, 'Sum squared error',  sse)
        
        else:
        
            pdf_fitted = []
            n, p, q = contiCDF.get_bnomi_param (y)
            
           # xx = np.arange(ss.nbinom.ppf(0.01, n, p), ss.nbinom.ppf(0.99, n, p))
            
            #s = np.random.negative_binomial(n, p, 1000000)
            
          #  for i in range(150, 320):
           #         probability = sum(s==i) / 1000000
            #        pdf_fitted.append(probability)

    
            #pdf_fitted = ss.binom.pmf (x, n, p)
            
        #    print(pdf_fitted)
            
            print (n, p, q)
            
            # Plot the fitted distribution on raw data
            plt.plot(range(0,400), ss.binom.pmf (range(0,400), n, p, q), label=dist_name)
            plt.xlim(0,400)
            
            # Distribution values for the histogram bin mid-values to be 
            # able to compare fitted values with the histogram values
            
            yfit = ss.binom.pmf (bn, n, p)
            
            # Calculate sum squared values between fitted probability and 
            # actual probability obtained from histogram (y/sample_size)
            sse = np.sum(np.power(yfit - h/len(y), 2.0))
            
            allSSE.append(sse)
            
            print(dist_name, 'Sum squared error',  sse)
 
    
        
    
    indx = allSSE.index(min(allSSE))
    
    print ('Best fit is', dist_names[indx])
    
    
    plt.legend(loc='upper right')
    plt.xlabel ('Score in runs')
    plt.ylabel ('Probability')
    
    plt.show()
    
