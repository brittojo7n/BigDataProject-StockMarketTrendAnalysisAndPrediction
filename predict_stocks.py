from __future__ import division
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
from pandas_datareader import DataReader
from datetime import datetime
import yfinance as yf
from datetime import datetime, timedelta
tech_list = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
end = datetime.now()
start = end - timedelta(days=365)
stock_data = {}
for stock in tech_list:
    try:
        stock_data[stock] = yf.download(stock, start=start, end=end)
    except Exception as e:
        print(f"Failed to fetch data for {stock}: {e}")
AAPL = stock_data.get('AAPL')  # Extract the data for AAPL from the dictionary
if AAPL is not None:
    print("AAPL Stock Data:")
    print(AAPL.head())  # Display the first few rows of the AAPL DataFrame
else:
    print("Data for AAPL is not available.")
AAPL.head()
AAPL.describe()
AAPL.info()
AAPL['Close'].plot(legend=True, figsize=(10,4))

AAPL['Volume'].plot(legend=True, figsize=(10,4))
import pandas as pd
MA_day = [10, 20, 50, 100]
for ma in MA_day:
    column_name = f"MA for {ma} days"
    AAPL[column_name] = AAPL['Close'].rolling(window=ma).mean()
print(AAPL.head())
AAPL[['Close','MA for 10 days','MA for 20 days','MA for 50 days','MA for 100 days']].plot(subplots=False,figsize=(10,4))
AAPL['Daily Return'] = AAPL['Close'].pct_change()
AAPL['Daily Return'].plot(figsize=(12,4), legend=True, linestyle='--', marker='o')
AAPL['Daily Return'].hist(bins=100)
sns.displot(AAPL['Daily Return'].dropna(), bins=100, color='magenta')
closingprice_df = pd.DataFrame()

for stock in tech_list:
    try:
        data = yf.download(stock, start=start, end=end)
        closingprice_df[stock] = data['Close']
    except Exception as e:
        print(f"Failed to fetch data for {stock}: {e}")
closingprice_df.head(10)
tech_returns = closingprice_df.pct_change()
tech_returns.head()
import seaborn as sns
sns.jointplot(x='GOOGL', y='GOOGL', data=tech_returns, kind='scatter', color='orange')
import seaborn as sns
sns.jointplot(x='GOOGL', y='AMZN', data=tech_returns, kind='scatter', height=8, color='skyblue')
import seaborn as sns
sns.jointplot(x='GOOGL', y='AMZN', data=tech_returns, kind='hex', height=8, color='skyblue')
import seaborn as sns
sns.jointplot(x='AAPL', y='MSFT', data=tech_returns, kind='reg', height=8, color='skyblue')
from IPython.display import SVG
SVG(url='http://upload.wikimedia.org/wikipedia/commons/d/d4/Correlation_examples2.svg')
sns.pairplot(tech_returns.dropna(),size=3)
returns_fig = sns.PairGrid(tech_returns.dropna())
returns_fig.map_upper(plt.scatter,color='purple')
returns_fig.map_lower(sns.kdeplot,cmap='cool_d')
returns_fig.map_diag(plt.hist,bins=30)
returns_fig = sns.PairGrid(closingprice_df.dropna())
returns_fig.map_upper(plt.scatter,color='purple')
returns_fig.map_lower(sns.kdeplot,cmap='cool_d')
returns_fig.map_diag(plt.hist,bins=30)
sns.heatmap(tech_returns.corr(),annot=True,fmt=".3g",cmap='YlGnBu')
sns.heatmap(closingprice_df.corr(),annot=True,fmt=".3g",cmap='YlGnBu')
rets = tech_returns.dropna()
rets.head()
area = np.pi*20

plt.scatter(rets.mean(),rets.std(),s=area)
plt.xlim([-0.0025,0.0025])
plt.ylim([0.001,0.025])
plt.xlabel('Expected returns')
plt.ylabel('Risk')
for label, x, y in zip(rets.columns, rets.mean(), rets.std()):
    plt.annotate(
        label,
        xy = (x, y), xytext = (50, 50),
        textcoords = 'offset points', ha = 'right', va = 'bottom',
        arrowprops = dict(arrowstyle = 'fancy', connectionstyle = 'arc3,rad=-0.3'))
sns.displot(AAPL['Daily Return'].dropna(),bins=100,color='purple')
rets["AAPL"].quantile(0.05)
rets["AMZN"].quantile(0.05)
rets["GOOGL"].quantile(0.05)
rets["MSFT"].quantile(0.05)
rets.head()
days = 365
dt = 1/days
mu = rets.mean()['GOOGL']
sigma = rets.std()['GOOGL']
def stock_monte_carlo(start_price,days,mu,sigma):
    ''' This function takes in starting stock price, days of simulation,mu,sigma, and returns simulated price array'''
    price = np.zeros(days)
    price[0] = start_price
    shock = np.zeros(days)
    drift = np.zeros(days)
    for x in range(1,days):
        shock[x] = np.random.normal(loc=mu * dt, scale=sigma * np.sqrt(dt))
        drift[x] = mu * dt
        price[x] = price[x-1] + (price[x-1] * (drift[x] + shock[x]))

    return price
print(closingprice_df['GOOGL'].head())
start_price = 830.09

for run in range(100):
    plt.plot(stock_monte_carlo(start_price, days, mu, sigma))

plt.xlabel("Days")
plt.ylabel("Price")
plt.title('Monte Carlo Analysis for Google')
print(closingprice_df['AMZN'].head())
start_price = 824.95

for run in range(100):
    plt.plot(stock_monte_carlo(start_price, days, mu, sigma))

plt.xlabel("Days")
plt.ylabel("Price")
plt.title('Monte Carlo Analysis for Amazon')
AAPL.head()
start_price = 117.10

for run in range(100):
    plt.plot(stock_monte_carlo(start_price, days, mu, sigma))

plt.xlabel("Days")
plt.ylabel("Price")
plt.title('Monte Carlo Analysis for Apple')
print(closingprice_df['MSFT'].head())
start_price = 59.94

for run in range(100):
    plt.plot(stock_monte_carlo(start_price, days, mu, sigma))

plt.xlabel("Days")
plt.ylabel("Price")
plt.title('Monte Carlo Analysis for Microsoft')
start_price = 830.09
runs = 10000
simulations = np.zeros(runs)

for run in range(runs):
    simulations[run] = stock_monte_carlo(start_price,days,mu,sigma)[days-1]
q = np.percentile(simulations, 1)
plt.hist(simulations, bins=200)
plt.figtext(0.6, 0.8, s='Start Price: $%.2f' % start_price)
plt.figtext(0.6, 0.7, s='Mean Final Price: $%.2f' % simulations.mean())
plt.figtext(0.6, 0.6, s='VaR(0.99): $%.2f' % (start_price - q))
plt.figtext(0.15, 0.6, s="q(0.99): $%.2f" % q)
plt.axvline(x=q, linewidth=4, color='r')
plt.title("Final price distribution for Google Stock(GOOGL) after %s days" % days, weight='bold', color='yellow')
start_price = 824.95
runs = 10000
simulations = np.zeros(runs)

for run in range(runs):
    simulations[run] = stock_monte_carlo(start_price,days,mu,sigma)[days-1]
q = np.percentile(simulations, 1)
plt.hist(simulations, bins=200)
plt.figtext(0.6, 0.8, s='Start Price: $%.2f' % start_price)
plt.figtext(0.6, 0.7, s='Mean Final Price: $%.2f' % simulations.mean())
plt.figtext(0.6, 0.6, s='VaR(0.99): $%.2f' % (start_price - q))
plt.figtext(0.15, 0.6, s="q(0.99): $%.2f" % q)
plt.axvline(x=q, linewidth=4, color='r')
plt.title("Final price distribution for Amazon Stock(AMZN) after %s days" % days, weight='bold', color='green')
start_price = 117.10
runs = 10000
simulations = np.zeros(runs)

for run in range(runs):
    simulations[run] = stock_monte_carlo(start_price,days,mu,sigma)[days-1]
q = np.percentile(simulations, 1)
plt.hist(simulations, bins=200)
plt.figtext(0.6, 0.8, s='Start Price: $%.2f' % start_price)
plt.figtext(0.6, 0.7, s='Mean Final Price: $%.2f' % simulations.mean())
plt.figtext(0.6, 0.6, s='VaR(0.99): $%.2f' % (start_price - q))
plt.figtext(0.15, 0.6, s="q(0.99): $%.2f" % q)
plt.axvline(x=q, linewidth=4, color='r')
plt.title("Final price distribution for Apple Stock(AAPL) after %s days" % days, weight='bold', color='blue')
start_price = 59.94
runs = 10000
simulations = np.zeros(runs)

for run in range(runs):
    simulations[run] = stock_monte_carlo(start_price,days,mu,sigma)[days-1]
q = np.percentile(simulations, 1)
plt.hist(simulations, bins=200)
plt.figtext(0.6, 0.8, s='Start Price: $%.2f' % start_price)
plt.figtext(0.6, 0.7, s='Mean Final Price: $%.2f' % simulations.mean())
plt.figtext(0.6, 0.6, s='VaR(0.99): $%.2f' % (start_price - q))
plt.figtext(0.15, 0.6, s="q(0.99): $%.2f" % q)
plt.axvline(x=q, linewidth=4, color='r')
plt.title("Final price distribution for Microsoft Stock(MSFT) after %s days" % days, weight='bold', color='magenta')
import yfinance as yf
from datetime import datetime, timedelta
NYSE_list = ['JNJ', 'NKE', 'WMT']
end = datetime.now()
start = end - timedelta(days=365)
stock_data = {}
for stock in NYSE_list:
    try:
        stock_data[stock] = yf.download(stock, start=start, end=end)
    except Exception as e:
        print(f"Failed to fetch data for {stock}: {e}")
if 'JNJ' in stock_data:
    JNJ = stock_data['JNJ']
else:
    print("JNJ data is not available.")
JNJ.head()
JNJ.describe()
JNJ.info()
JNJ['Close'].plot(title='Closing Price - JNJ',legend=True, figsize=(10,4))
if 'NKE' in stock_data:
    NKE = stock_data['NKE']
else:
    print("NKE data is not available.")
NKE['Close'].plot(title='Closing Price - NKE', legend=True, figsize=(10, 4))
if 'WMT' in stock_data:
    WMT = stock_data['WMT']
else:
    print("WMT data is not available.")
WMT['Close'].plot(title='Closing Price - WMT', legend=True, figsize=(10, 4))
JNJ['Daily Return'] = JNJ['Close'].pct_change()
sns.displot(JNJ['Daily Return'].dropna(), bins=100, color='r')
(JNJ['Daily Return'].dropna()).quantile(0.05)
WMT['Daily Return'] = WMT['Close'].pct_change()
sns.displot(WMT['Daily Return'].dropna(), bins=100, color='g')
(WMT['Daily Return'].dropna()).quantile(0.05)
NKE['Daily Return'] = NKE['Close'].pct_change()
sns.displot(NKE['Daily Return'].dropna(), bins=100, color='b')
(NKE['Daily Return'].dropna()).quantile(0.05)
