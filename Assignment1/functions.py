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

def portofolio_strategies():
    """
    X = [x1,x2,1-x1-x2] for all possibilities,
    e.g.: 
        [[1.0,0.0,0.0],
         [0.9,0.1,0.0],
         [0.8,0.2,0.0],
        ...
         [0.0,0.0,1.0]]
    """
    X = []
    x1 = np.arange(0.0,1.0,0.1)[::-1]
    x2 = np.arange(0.0,1.0,0.1)
    for i in x1:
        for j in x2:
            if (j+i) <= 1.0:
                X.append([i,j,1-i-j])
    return np.array(X)
        
def mean_vs_stdevs(portofolio_strategy,mean_returns):
    """
    Too lazy to write description but solves part 5
    """
    mean_matrix = []
    for asset_strategy in portofolio_strategy:
        mcd_returns = asset_strategy[0]*mean_returns[0]
        ko_returns = asset_strategy[1]*mean_returns[1]
        msft_returns = asset_strategy[2]*mean_returns[2]
        mean_matrix.append([mcd_returns,ko_returns,msft_returns])

    returns = []
    stds = []
    for row in mean_matrix:
        returns.append(sum(row))
        stds.append(np.std(row))

    return returns, stds




