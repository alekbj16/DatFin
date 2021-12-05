#Import
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn")
from numpy.lib.twodim_base import diag
from  functions import * 

#***1: Efficient frontier with and withoud riskless lending and borrowing***

r_f = 0.02 #Risk free rate: 2%
r_1 = 0.10 #Asset 1 yearly expected return: 10%
s_1 = 0.10 #Asset 1 standard dev
r_2 = 0.20 #Asset 2 yearly expected return: 20%
s_2 = 0.20 #Asset 2 standard dev

corr_coeff = np.array([1,0.5,0,-1]) #The different correlation coefficients between asset 1 and 2


#***1.1: How to plot efficient frontier *with* borrowing

#****1.1.1: 
#For the case rho = 0.5

rho = corr_coeff[1]
opt_portofolio = optimal_portofolio(r_f=r_f,r_1=r_1,r_2=r_2,s_1=s_1,s_2=s_2,rho=rho)
opt_portofolio_returns = portofolio_return(r_1=r_1,r_2=r_2,X_opt=opt_portofolio)
opt_portofolio_var = portofolio_variance(r_1,r_2,s_1=s_1,s_2=s_2,X_opt=opt_portofolio,rho=rho)


x = np.linspace(0,0.04,10)
r_f = np.linspace(0.02,0.08,4)
for i in r_f:
    slope = get_slope(r_f=i,portofolio_return=opt_portofolio_returns,portofolio_variance=opt_portofolio_var)
    f_x = slope*x+i
    plt.plot(x,f_x,label=f"r_f = {i}")
plt.title(f"\u03C1 = {rho}")
plt.legend()
plt.xlabel("Standard deviation")
plt.ylabel("Expected return")
plt.show()
# Xopt is the point which has highest slope and 
# describes the optimal amount of stock

#print(correlation_matrix)
#print(omega)

# Assume that a riskless lending and borrowing rate exists
# and find the optimum portfolio. Then assume that a different riskless lending and borrowing
# rate exists and find the optimum portfolio that corresponds to this second rate.
# Continue changing the assumed riskless rate until the full efficient frontier is determined.

