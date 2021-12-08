from functions import *
import datetime as dt


#*** Signle Asset Model***

#Data collection
tickers = ["APX","MCD","GOOG","XOM","IBM","WMT","KO","^GSPC"] #American Express, McDonalds, Google, Exxon Mobil Corportation, IBM, Nike, Wal-Mart, Coca-Cola, SP500
#Note: dropping nike due to data fetch error from yahoo. 

start = dt.datetime(2007,1,1)
end = dt.datetime(2013,1,1)
data = get_price_data(tickers=tickers,start=start,end=end,interval="w",adj_close=True)
data_apx = data[0]
data_mcd = data[1]
data_goog = data[2]
data_xom = data[3]
data_ibm = data[4]
#data_nike = data[5]
data_wmt = data[5]
data_ko = data[6]
data_sp500 = data[7]

#Yearly returns and variances for all the companies
#American express
apx_returns_wk = calculate_returns(data=data_apx) 
apx_returns_yr = yearly_returns(apx_returns_wk)
apx_variance_yr = np.var(apx_returns_yr)

#McDonalds
mcd_returns_wk = calculate_returns(data=data_mcd) 
mcd_returns_yr = yearly_returns(mcd_returns_wk)
mcd_variance_yr = np.var(mcd_returns_yr)

#Google 
goog_returns_wk = calculate_returns(data=data_goog) 
goog_returns_yr = yearly_returns(goog_returns_wk)
goog_variance_yr = np.var(goog_returns_yr)

#Exxon
xom_returns_wk = calculate_returns(data=data_xom) 
xom_returns_yr = yearly_returns(xom_returns_wk)
xom_variance_yr = np.var(xom_returns_yr)

#IBM
ibm_returns_wk = calculate_returns(data=data_ibm) 
ibm_returns_yr = yearly_returns(ibm_returns_wk)
ibm_variance_yr = np.var(ibm_returns_yr)

#WallMart
wmt_returns_wk = calculate_returns(data=data_wmt) 
wmt_returns_yr = yearly_returns(wmt_returns_wk)
wmt_variance_yr = np.var(wmt_returns_yr)

#Coke
ko_returns_wk = calculate_returns(data=data_ko) 
ko_returns_yr = yearly_returns(ko_returns_wk)
ko_variance_yr = np.var(ko_returns_yr)

#For the "market" represented by S&P 500
sp500_returns_wk = calculate_returns(data=data_sp500) #Weekly returns
sp500_returns_yr = yearly_returns(sp500_returns_wk)
sp500_variance_yr = np.var(sp500_returns_yr)



# *** Regression model and parameters***

#American Express
alpha_apx, beta_apx, var_ei_apx = calculate_parameters(sp500_returns_yr, apx_returns_yr)

#McDonalds
alpha_mcd, beta_mcd, var_ei_mcd = calculate_parameters(sp500_returns_yr, mcd_returns_yr)

#Google
alpha_goog, beta_goog, var_ei_goog = calculate_parameters(sp500_returns_yr, goog_returns_yr)

#Exxon 
alpha_xom, beta_xom, var_ei_xom = calculate_parameters(sp500_returns_yr, xom_returns_yr)

#IBM
alpha_ibm, beta_ibm, var_ei_ibm = calculate_parameters(sp500_returns_yr, ibm_returns_yr)

#WallMart
alpha_wmt, beta_wmt, var_ei_wmt = calculate_parameters(sp500_returns_yr, wmt_returns_yr)

#Coke
alpha_ko, beta_ko, var_ei_ko = calculate_parameters(sp500_returns_yr, ko_returns_yr)

# *** Co-variance matrix *** 
yearly_returns = np.array([apx_returns_yr,mcd_returns_yr,goog_returns_yr,xom_returns_yr,ibm_returns_yr,wmt_returns_yr,ko_returns_yr])
#print(f"\nReal yearly co-variance matrix of the continous stock returns:\n\n{covariance_matrix(yrly_data=yearly_returns)}\n\n")

# *** 
#Equally weighted portofolio using real data
X = np.array([1/7 for i in range(7)])
mean_returns = np.array([np.mean(i) for i in yearly_returns])
stds = np.array([np.std(i) for i in yearly_returns])

expected_return = np.sum((1/7)*mean_returns)
variance = np.var((1/7)*mean_returns)
print("===========")
print("Real estimation, equal weighted portofolio")
print(f"Expected return: {expected_return}")
print(f"Variance: {variance}")
print("===========")


print("\n\n===========")
print("Single index model, equal weighted portofolio")
alphas = np.array([alpha_apx,alpha_mcd,alpha_goog,alpha_xom,alpha_ibm,alpha_wmt,alpha_ko])
betas = np.array([beta_apx,beta_mcd,beta_goog,beta_xom,beta_ibm,beta_wmt,beta_ko])
vars = np.array([var_ei_apx,var_ei_mcd,var_ei_goog,var_ei_xom,var_ei_ibm,var_ei_wmt,var_ei_ko])
expected_return_sim = np.sum((1/7)*alphas) + np.sum((1/7)*np.mean(sp500_returns_yr)*betas)
beta_portfolio = np.sum([(1/7)*betas]) # formula from in p.135 in book
variance_smi = (beta_portfolio**2)*sp500_variance_yr + np.sum(((1/7)**2)*vars)
print(f"Expected return: {expected_return_sim}")
print(f"Variance: {variance_smi}")
print("===========")

print("\nThe two methods show similar expected return, but large differences in variances\n")





