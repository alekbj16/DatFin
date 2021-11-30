import pandas_datareader as web
import datetime as dt
import numpy as np

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

def covariance_matrix(data):
    """
    Input:
        data is array on form [yrly_mcd,yrly_ko,yrly_msft]
    Output:
        Corvariance matrix as described in assignment
    """
    
    #Variance of returns
    var_returns = np.array([np.var(data[0]),np.var(data[1]),np.var(data[2])])
    s = np.zeros((3,3))
    #Covar coefficients
    s_mcd_ko = np.corrcoef(np.array((data[0],data[1])))[0,1]
    s_mcd_msft = np.corrcoef(np.array((data[0],data[2])))[0,1]
    s_ko_msft = np.corrcoef(np.array((data[1],data[2])))[0,1]
    s[0,1] = s_mcd_ko
    s[1,0] = s_mcd_ko
    s[0,2] = s_mcd_msft
    s[2,0] = s_mcd_msft
    s[1,2] = s_ko_msft
    s[2,1] = s_ko_msft
    np.fill_diagonal(s,var_returns)
    return s




