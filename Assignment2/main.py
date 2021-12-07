#Imports
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import std
plt.style.use("seaborn")
from numpy.lib.twodim_base import diag
from  functions import * 
import datetime as dt

#***1: Efficient frontier with and withoud riskless lending and borrowing***
r_f = 0.02 #Risk free rate: 2%
r_1 = 0.10 #Asset 1 yearly expected return: 10%
s_1 = 0.10 #Asset 1 standard dev
r_2 = 0.20 #Asset 2 yearly expected return: 20%
s_2 = 0.20 #Asset 2 standard dev

returns = np.array([r_1,r_2]) #Array of returns
stdevs = np.array([s_1,s_2]) #Array of standard deviations

corr_coeff = np.array([1,0.5,0,-1]) #The different correlation coefficients between asset 1 and 2

#***1.1: How to plot efficient frontier *with* borrowing

#****1.1.1: 
#For the case rho = 0.5
rho = corr_coeff[1]

#Calculate the optimal portofilio strategy
opt_portofolio = optimal_portofolio(r_f=r_f,returns=returns,stdevs=stdevs,rho=rho)

#Calculate the returns that are acheived with this strategy
opt_portofolio_returns = portofolio_return(returns=returns,X_opt=opt_portofolio)

#Calculate the variance acheived with this strategy
opt_portofolio_var = portofolio_variance_2d(stdevs=stdevs,X_opt=opt_portofolio,rho=rho)

#Plotting the efficient set
x = np.linspace(0,0.04,10)
#Calcualting the slope
slope = get_slope(r_f=r_f,portofolio_return=opt_portofolio_returns,portofolio_variance=opt_portofolio_var)
#Defining the function
f_x = slope*x+r_f
#Plot
plt.plot(x,f_x,label=f"r_f = {r_f}")
plt.title(f"Efficient frontier with \u03C1 = {rho}")
plt.legend()
plt.xlabel("Variance")
plt.ylabel("Expected return")
plt.show()


### Without borrowing ### 

#Adapting procedure as described in the book [p. 100]:
#***
# Assume that a riskless lending and borrowing rate exists
# and find the optimum portfolio. Then assume that a different riskless lending and borrowing
# rate exists and find the optimum portfolio that corresponds to this second rate.
# Continue changing the assumed riskless rate until the full efficient frontier is determined.
#***

r_fs = np.array([0.02,0.04,0.06,0.08]) #Various values for the risk free rate
plt.figure()
for r_f in r_fs: 
    opt_portofolio = optimal_portofolio(r_f=r_f,returns=returns,stdevs=stdevs,rho=rho)
    opt_portofolio_returns = portofolio_return(returns=returns,X_opt=opt_portofolio)
    opt_portofolio_var = portofolio_variance_2d(stdevs=stdevs,X_opt=opt_portofolio,rho=rho)
    x = np.linspace(0,0.03,10)
    slope = get_slope(r_f=r_f,portofolio_return=opt_portofolio_returns,portofolio_variance=opt_portofolio_var)
    f_x = slope*x+r_f
    plt.plot(x,f_x,label=f"r_f = {r_f}")
    plt.title(f"Efficient frontier, no lending and borrowing, with \u03C1 = {rho}")
plt.legend()
plt.xlabel("Standard deviation")
plt.ylabel("Variance")
plt.show()


### *** With actual data *** ###

#***Data collection***
tickers = ["PEP","KO","MSFT"] #Pepsi, Coca-Cola, Microsoft

#10 years of data
start = dt.datetime(1991,1,1)
end = dt.datetime(2001,1,1)
data = get_price_data(tickers=tickers,start=start,end=end,interval="w",adj_close=True)
data_pep = data[0]
data_ko = data[1]
data_msft = data[2]

returns_pep = calculate_returns(data_pep)
returns_ko = calculate_returns(data_ko)
returns_msft = calculate_returns(data_msft)

#***3: Mean and covariance matrix of yearly returns***
yrly_pep = yearly_returns(returns_pep)
yrly_ko = yearly_returns(returns_ko)
yrly_msft = yearly_returns(returns_msft)

mean_returns = np.array([np.mean(yrly_pep),np.mean(yrly_ko),np.mean(yrly_msft)]) #Mean of yearly returns
covar_mat = covariance_matrix([yrly_pep,yrly_ko,yrly_msft])
stdevs = np.array([np.sqrt(covar_mat[0][0]),np.sqrt(covar_mat[1][1]),np.sqrt(covar_mat[2][2])]) #standard deviations are 



#***1: Efficient frontier with and withoud riskless lending and borrowing***
r_f = 0.02 #Risk free rate: 2%

#****2.1: 
#For the case rho = 0.5
rho = corr_coeff[1]
opt_portofolio = optimal_portofolio(r_f=r_f,returns=mean_returns[0:2],stdevs=stdevs[0:2],rho=rho)
opt_portofolio_returns = portofolio_return(returns=mean_returns[0:2],X_opt=opt_portofolio)
opt_portofolio_var = portofolio_variance_2d(stdevs=stdevs[0:2],X_opt=opt_portofolio,rho=rho)

x = np.linspace(0,0.04,10)
#r_f = np.linspace(0.02,0.08,4)
slope = get_slope(r_f=r_f,portofolio_return=opt_portofolio_returns,portofolio_variance=opt_portofolio_var)
f_x = slope*x+r_f
plt.plot(x,f_x,label=f"r_f = {r_f}")
plt.title(f"Efficient frontier with \u03C1 = {rho} and real data")
plt.legend()
plt.xlabel("Variance")
plt.ylabel("Expected return")
plt.show()

r_fs = np.array([0.02,0.04,0.06,0.08])
plt.figure()
for r_f in r_fs:
    opt_portofolio = optimal_portofolio(r_f=r_f,returns=mean_returns[0:2],stdevs=stdevs[0:2],rho=rho)
    opt_portofolio_returns = portofolio_return(returns=mean_returns[0:2],X_opt=opt_portofolio)
    opt_portofolio_var = portofolio_variance_2d(stdevs=stdevs[0:2],X_opt=opt_portofolio,rho=rho)
    x = np.linspace(0,0.03,10)
    slope = get_slope(r_f=r_f,portofolio_return=opt_portofolio_returns,portofolio_variance=opt_portofolio_var)
    f_x = slope*x+r_f
    plt.plot(x,f_x,label=f"r_f = {r_f}")
    plt.title(f"Efficient frontier, no lending and borrowing, with \u03C1 = {rho}. Using real data")
plt.legend()
plt.xlabel("Standard deviation")
plt.ylabel("Variance")
plt.show()
