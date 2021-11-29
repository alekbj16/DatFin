import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
from get_price_data import get_price_data

#***1: Data collection***
tickers = ["MCD","KO","MSFT"] #McDonalds, Coca-Cola, Microsoft

#10 years of data
start = dt.datetime(1991,1,1)
end = dt.datetime(2001,1,1)
data = get_price_data(tickers=tickers,start=start,end=end,interval="w",adj_close=True)
data_mcd = data[0]
data_ko = data[1]
data_msft = data[2]


#***2: Calculate continous returns***



