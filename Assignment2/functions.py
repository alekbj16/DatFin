import numpy as np
from numpy.core.fromnumeric import var
from numpy.lib.twodim_base import diag 
import pandas_datareader as web

def optimal_portofolio(r_f,r_1,r_2,s_1,s_2,rho):
    correlation_matrix = np.zeros((2,2))
    correlation_matrix[0,0] = 1
    correlation_matrix[1,1] = 1
    correlation_matrix[0,1] = rho
    correlation_matrix[1,0] = rho
    mu = np.array([r_1,r_2])
    sigma = np.array([s_1,s_2])
    omega = np.dot(np.dot(np.diag(sigma),correlation_matrix),diag(sigma)) #Could have used numpy multidot
    A = 2*omega
    A = np.c_[ A, -(mu-r_f)]
    A = np.vstack([A, np.append([(mu-r_f)],0)])
    r_p = 0.4 #Assuming a trivial value for expected return. Does not affect when finding optimal X because of normalization
    b = np.zeros((2,1))
    b = np.vstack([b,r_p-r_f])
    X = np.dot(np.linalg.inv(A),b)
    X_opt = X[0:2]*(1/np.sum(X[0:2]))
    return X_opt

def portofolio_return(r_1,r_2,X_opt):
    return X_opt[0]*r_1 + X_opt[1]*r_2

def portofolio_variance(r_1,r_2,s_1,s_2,X_opt,rho):
    return (X_opt[0]**2)*(s_1**2)+(X_opt[1]**2)*(s_2**2)+2*X_opt[0]*X_opt[1]*s_1*s_2*rho

def get_slope(r_f,portofolio_return,portofolio_variance):
    return (portofolio_return-r_f)/portofolio_variance

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

def portofolio_strategies():
    x1 = np.linspace(0.0,1.0,11)
    X = [1-x1,x1]
    return np.array(X)

# def calculate_returns(strategies,r_1,r_2):
#     returns = []
#     for i in range(0,len(strategies[0])):
#         returns.append(strategies[0][i]*r_1+strategies[1][i]*r_2)
#     return np.array(returns)

def calculate_variances(strategies,r_1,r_2,s_1,s_2,rho):
    variances = []
    for i in range(0,len(strategies[0])):
        variances.append((strategies[0][i]**2)*(s_1**2)+(strategies[1][i]**2)*(s_2**2)+2*strategies[0][i]*strategies[0][i]*s_1*s_2*rho)
    return np.array(variances)

def mean_to_var(means,vars):
    return np.divide(means,vars)


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
