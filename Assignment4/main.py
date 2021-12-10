import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from functions import *


#Stocks
#Amex, Apple, Amazon, McDonald, WallMart, CocaCola and the SP500
start = dt.datetime(2000,1,1)
end = dt.datetime(2013,1,1)
tickers = ["APX","AAPL","AMZN","MCD","WMT","KO","^GSPC"]
data = get_price_data(tickers=tickers,start=start,end=end,interval="w",adj_close=True)

#Estimate avg retrun and covariances for the assets
weekly_returns = []
for d in data:
    weekly_returns.append(calculate_returns(data=d))
#print(weekly_returns)

yrly_returns = []
for i in weekly_returns:
    yrly_returns.append(yearly_returns(i))

company_returns = yrly_returns[:-1] #All elements in list except last
sp500_returns = yrly_returns[-1] #Last element in list
#print(f"Compnay yearly returns {company_returns}")
#print(f"sp500 yearly  returns {sp500_returns}")

#Avg expected returns for each company
mean = np.mean(company_returns,axis=1)
#Uncomment line below to see avg expected returns
#print(f"\nAvg expected returns for each company: {mean}")

#Covariance matrix
covar_mat = covariance_matrix(company_returns)
#Uncomment line below to see covariance matrix
#print(f"\nCovariance matrix: {covar_mat}")


# Portfolio with equal weights for Amex, Apple and Amazon:
X = [(1/3) for i in range(3)]

# Assuming that beta should be calculated with regression as
# in the single-index-model assignment
betas = []
for comp_ret in company_returns:
    betas.append(calculate_beta(sp500_returns,comp_ret))
#Uncomment line below to display various betas
#print(f"The different betas: {betas}")

#Beta for the portfolio Amex, Apple, Amazon
beta_p = np.dot(X,betas[0:3]) 

print("===================")
print("Portfolio: Equal share (1/3) of Amex, Apple and Amzon")
print(f"Beta_p (portfolio beta) {beta_p}")
print(f"Expected return of portfolio: ")
print(f"Variance of portfolio:")
print("===================")



#Pick portofolio where stocks are weighted equally
#Calculate beta of portofolio, and expected return and variance

