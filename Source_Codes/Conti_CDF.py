#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 10:40:32 2019

@author: mdsamad
"""

import numpy as np
import scipy.stats as ss
import scipy.optimize as so

import matplotlib.pyplot as plt



# Cumulative density function p(x)
def pval(df,x_val,dist):
    
    mu = df.mean()
    sigma = df.std()
    
    p = dist.cdf(x_val, mu, sigma)

    return p

# Inverse CDF to obtain X-val for a given probability
def InvPval(df,p_val,dist):
    
    mu = df.mean()
    sigma = df.std()
    
    x_val = dist.ppf(p_val, mu, sigma)

    return x_val


def PlotCDF(df, label,dist):
    
    mu = df.mean()
    sigma = df.std()
    
    x = np.linspace(mu - 5*sigma, mu + 5*sigma, 100)
    
    plt.plot(x, 1- dist.cdf(x, mu, sigma), label = label)
    
    plt.xlim(0, 400)
    plt.xlabel('Score in runs')
    plt.ylabel ('Probability')
    plt.legend ()
    
    
    
    
def PLOT_ALL_CDF (df, filename, dist):
    
    
    # Data of Win by runs (first bat team win)
    df1= df.loc[df.Type=='runs']
    
    # Data of Win by wickets (first bowl team win)
    df2= df.loc[df.Type=='wickets']
    
    fig = plt.figure()
    fig.suptitle(filename[:-5], fontsize=10)

    PlotCDF(df1.fInn, 'Bat First win', dist)
    PlotCDF(df1.sInn, 'Bat Second lose', dist)
    PlotCDF (df2.sInn, 'Bat Second win', dist)
    PlotCDF(df2.fInn, 'Bat First lose', dist)  
    
    fig.savefig('Results/Figure/Continuous/'+filename[:-5]+ '.jpg')   # save the figure to file
    plt.close(fig) 
    
    
def AnalysisData (df, runs, dist):
    
                
    # Data of Bat-first-win cases, type = won by runs
    df1= df.loc[df.Type =='runs']
    
    # Probability of scoring > "runs" given that bat-first-win case
  
    # f(x) = 1 - phi (x)
    # P (S > Xf | Bat-first-win)
        
    pBatFW =  1- pval(df1.fInn,runs,dist)

   
    # Data of Bat-second-win 
    df2= df.loc[df.Type=='wickets']
    
    # number of matches bat-second-win versus bat-first-win
    ratio = sum(df.Type == 'runs')/sum(df.Type == 'wickets')

    
    # Revised score for second innings batting that ensures equal
    # likelihood of "scoring" and "winning" in both innings
    rev_scr = np.round (InvPval (df2.sInn, 1- pBatFW*ratio, dist))
    
    
    ################### Extra Analysis #################
    #########################################
    
    # P (S > Xf | B-first) - regarless of win or loss
    # pBatF = 1- pval(df.fInn,runs,dist)
    
    # probability of scoring "runs" run in the first innings
    # regardless of match results
    
    pp = pval(df.fInn,runs, dist)
    
    # print the runs in the second innings that is equally probable as the
    # run scored in the first innings
    
    print('Revised score for equal scoring probability: ', np.round (InvPval (df.sInn, pp ,dist)) )
    ###########################################
    ################# END of EXTRA ANALYSIS ###################
    
    
    
    print ('1st innings score:', runs, ' runs.')
    print('2nd innings revised target score for equal winning probability: ',rev_scr, 'runs')
    print ('Run difference is ', np.round (runs - rev_scr), 'runs')
    
    
    y_ratio = (1-pval(df.sInn,rev_scr,dist))/(1-pval(df.fInn,runs,dist))
    
    print ('Probability ratio of scoring after equal winning probability', np.round (y_ratio,2))    
    
   
    ############### ADDITIONAL ANALYSES #####################
    
    #rev_scr = InvPval (df2.sInn, 1 - pBatFW*ratio*y_ratio, dist)
    
    # print ('For', k, 'BSW', 1-pval(df2.sInn,rev_scr,dist),'BS', 1-pval(df.sInn,rev_scr,dist),
    #      'BF', 1-pval(df.fInn,runs,dist))
 
    #print ('vv',pBatFW*ratio, 'bb', y_ratio, 'hh',pBatFW*ratio*y_ratio )
         
        
   # Ratio of run scoring probability for first and second innings 
            
    xRange =np.linspace(250,runs,50)
    
    xm = []
    for x_data in xRange:
        
        
       # Lyx = (1- pval(df2.sInn,x_data,dist))/(1- pval(df.sInn,x_data,dist)) 
        
      #  Ryx = ((pBatFW))/(pBatF)
        
        
        # For a given first innings score, we plot probability ratio versus
        # a range of second innings score
        
        y_ratio = (1-pval(df.sInn,x_data,dist))/(1-pval(df.fInn,runs,dist)) 
        
        xm.append(y_ratio)
        
        
    #plt.figure()
    
    #plt.plot(xRange, xm)  
    
    return rev_scr
    ################# END OF ADDITIONAL ANALYSES ########################


################## Discrete probability distribution ######################
################ Negative Bionmial Distribution Fit #######################



    
def likelihood_f(P, x, neg=1):
    n=np.round(P[0]) #by definition, it should be an integer 
    p=P[1]
    loc=np.round(P[2])
    
    return neg*(np.log(ss.nbinom.pmf(x, n, p, loc))).sum()

def get_bnomi_param (X):

    
    result=[]
    for i in range(10,400): 
        _=so.fmin(likelihood_f, [i, 1, 0], args=(X,-1), full_output=True, disp=False)
        result.append((_[1], _[0]))

    #get the MLE
    P2 = sorted(result, key=lambda x: x[0])[0][1]


  #  plt.hist(X, bins=20, normed=True)
  #  plt.plot(range(0,400), ss.nbinom.pmf(range(0,400), np.round(P2[0]), P2[1]), 'r-')

    n = np.round(P2[0])
    p = P2[1]
    #loc = np.round(P2[2])

    return n, p


def Pval_nbinom (df, x_val):
    
    n, p = get_bnomi_param(df)
    
    p_val = ss.nbinom.cdf(x_val, n, p)
           
    return p_val

# Inverse CDF to obtain X-val for a given probability
def InvPval_binom(df, p_val):
    
    n, p = get_bnomi_param(df)
    
    x_val = ss.nbinom.ppf(p_val, n, p)
    
    return x_val


def PlotCDF_binom(df, label):
    
    
    n, p = get_bnomi_param(df)
    
    plt.plot(range(400), 1- ss.nbinom.cdf(range(400), n, p), label= label)
             
    plt.xlim(0, 400)
    plt.xlabel('Score in runs')
    plt.ylabel ('Probability')
    plt.legend ()
    

def PLOT_ALL_CDF_nbinom (df, filename):
    
    
    # Data of Win by runs (first bat team win)
    df1= df.loc[df.Type=='runs']
    
    # Data of Win by wickets (first bowl team win)
    df2= df.loc[df.Type=='wickets']
    
    fig = plt.figure()
    fig.suptitle(filename[:-5], fontsize=10)
    
    PlotCDF_binom(df1.fInn, 'Bat First win')
    PlotCDF_binom(df1.sInn, 'Bat Second lose')
    PlotCDF_binom (df2.sInn,  'Bat Second win')
    PlotCDF_binom(df2.fInn, 'Bat First lose')   
    
    fig.savefig('Results/Figure/Discrete/'+filename[:-5]+ '.jpg')   # save the figure to file
   
    
def AnalysisData_nBinom (df, runs):
    
                
    # Data of Bat-first-win
    df1= df.loc[df.Type =='runs']
    
    # Probability of scoring "runs" given that bat-first-win case
    # f(x) = 1 - phi (x)
    
    # P (S = Xb | Bat-first-win)
        
    pBatFW =  1- Pval_nbinom(df1.fInn,runs)

   
    # Data of Bat-second-win 
    df2= df.loc[df.Type=='wickets']
    
    # number of matches bat-second-win versus bat-first-win
    ratio = sum(df.Type == 'runs')/sum(df.Type == 'wickets')


    rev_scr = InvPval_binom (df2.sInn, 1- pBatFW*ratio)
    
    
    print ('1st innings score:', runs, ' runs.')
    print('2nd innings revised target score for equal winning probability: ',rev_scr, 'runs')
    print ('Run difference is ', runs - rev_scr, 'runs')
    
    
    y_ratio = (1-Pval_nbinom(df.sInn,rev_scr))/(1-Pval_nbinom(df.fInn,runs))
    
    print ('Probability ratio of scoring after equal winning probability', np.round (y_ratio,2)) 

  

    
    pp = Pval_nbinom(df.fInn,runs)
    
    # print the runs in the second innings that is equally probable as the
    # run scored in the first innings
    
    print('Revised score for equal scoring probability: ', np.round (InvPval_binom (df.sInn, pp) ))
    
    return rev_scr
        
   

#def DiscretePlot (df):
    
   