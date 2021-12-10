import pandas_datareader as web
import datetime as dt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels import regression

def get_price_data(tickers, start, end,interval,adj_close=False):
    """
    Input:
        Ticker: Uppercase ticker symbol
        Start, end: datetime object
        Interval: "y","m","d","h"
        Adj_close: Gives only adjusted close price if True. 
    Output: List of dataframes
    """
    data = []
    for t in tickers:
        data.append(web.get_data_yahoo(t,start,end,interval=interval))
    if adj_close==True:
        data_adj_closed = []
        for d in data:
            adj_closed = d[['Adj Close']].copy()
            data_adj_closed.append(adj_closed)
        return data_adj_closed
    return data

def calculate_returns(data):
    """
    Input: 
        data: dataframe with historical prices
    Output:
        returns for Adjusted Close price by formula R = log(P_t/P_(t-1))
    """
    returns = []
    for i in range(0,len(data)):
        if i == len(data)-1:
            break
        r = np.log(float(data['Adj Close'][i+1])/float(data['Adj Close'][i]))
        returns.append(r)
    return returns

def yearly_returns(weekly_returns):
    """
    Input: 
        List of weekly returns as obtained by the 'calculate_returns' function
    Output:
        Yearly returns (sum of daily)
    Note: Assumes 52 weeks per year
    """
    yr_returns = np.add.reduceat(weekly_returns,np.arange(0,len(weekly_returns),52))
    return yr_returns

def covariance_matrix(yrly_data):
    cov_mat = []
    for i in range(len(yrly_data)):
        cov_mat.append(np.array(yrly_data[i]))
    cov_mat = np.array(cov_mat)
    cov_mat = np.cov(cov_mat)
    return cov_mat

def linreg(x,y):
    """
    Linear regression using statsmodels package
    """
    x = sm.add_constant(x)
    model = regression.linear_model.OLS(y,x).fit()
    return model

def calculate_beta(x,y):
    model = linreg(x,y)
    #alpha = model.params[0]
    beta = model.params[1]
    #var_ei = np.var(model.resid,ddof=1)
    #return alpha, beta, var_ei
    return beta