from functions import *
import datetime as dt
from sklearn.linear_model import LinearRegression

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




