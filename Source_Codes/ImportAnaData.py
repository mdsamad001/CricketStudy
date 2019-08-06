#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 10:35:58 2019

@author: mdsamad
"""


import pandas as pd

import numpy as np

import os

def mylistdir(directory):
    """A specialized version of os.listdir() that ignores files that
    start with a leading period."""
    filelist = os.listdir(directory)
    return [x for x in filelist
            if not (x.startswith('.'))]



def SummaryData (df):
    
    df1= df.loc[df.Type=='runs']
    df2= df.loc[df.Type=='wickets']
    
    print('Total match', df.shape[0])
    
    print ('Per bat first win', (df1.shape[0]*100)/df.shape[0])
    
    print ('Per bat second win', (df2.shape[0]*100)/df.shape[0])

    print ('Bat f win', np.round (df1.fInn.mean()),
           'Bat s lose', np.round(df1.sInn.mean()),
           'Bat s win', np.round(df2.sInn.mean()),
           'Bat f lose', np.round(df2.fInn.mean()))
    

def dataExtract (filename):
    
    df = pd.read_excel(filename, index_col=0)
    df = df.iloc[:,:-2]
    df.Result = df.Result.shift(-2)
    df= df.dropna()
    df[['Margin','Type','D/L']]= df.Result.str.split(" ",expand=True)

    df.fInn = df.fInn.str.split("-",expand=True,)
    df.sInn = df.sInn.str.split("-",expand=True,)

    df.fInn = df.fInn.str.extract('(\d+)', expand=False).astype(int)
    df.sInn = df.sInn.str.extract('(\d+)', expand=False).astype(int)
    
    return df