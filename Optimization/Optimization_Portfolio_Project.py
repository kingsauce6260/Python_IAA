
#%%
import pandas as pd
import numpy as np
from gurobipy import *
import matplotlib.pyplot as plt
from math import sqrt
import datetime
#%%
amd =  pd.read_csv(r'/Users/thomasgow/Documents/IAA/Optimization/Homework/Data/AMD.csv')
ftv = pd.read_csv(r'/Users/thomasgow/Documents/IAA/Optimization/Homework/Data/FTV.csv')
it = pd.read_csv(r'/Users/thomasgow/Documents/IAA/Optimization/Homework/Data/IT.csv')
now = pd.read_csv(r'/Users/thomasgow/Documents/IAA/Optimization/Homework/Data/NOW.csv')
stx = pd.read_csv(r'/Users/thomasgow/Documents/IAA/Optimization/Homework/Data/STX.csv')

# Fixing Dates
it['Date'] = pd.to_datetime(it.Date)
it['Date'] = it['Date'].dt.strftime('%-m/%-d/%-y')

now['Date'] = pd.to_datetime(now.Date)
now['Date'] = now['Date'].dt.strftime('%-m/%-d/%-y')


stocks = [amd, ftv, it, now, stx]

#%%
amd.set_index('Date', inplace=True)
ftv.set_index('Date', inplace=True)
it.set_index('Date', inplace=True)
now.set_index('Date', inplace=True)
stx.set_index('Date', inplace=True)

#%%
for i in stocks:
    i['Adj Close'].plot()
    plt.xlabel("Date")
    plt.ylabel("Adjusted")
    plt.title("Price data")
    plt.show()

#%%
amd_daily_returns = amd['Adj Close'].pct_change()
ftv_daily_returns = ftv['Adj Close'].pct_change()
it_daily_returns = it['Adj Close'].pct_change()
now_daily_returns = now['Adj Close'].pct_change()
stx_daily_returns = stx['Adj Close'].pct_change()

returns = [amd_daily_returns, ftv_daily_returns, it_daily_returns, now_daily_returns, stx_daily_returns]
#%%
for i in returns:
    i = pd.DataFrame(i)

#%%
data = pd.merge(amd_daily_returns, ftv_daily_returns, left_index=True, right_index=True)
data = pd.merge(data, it_daily_returns, left_index=True, right_index=True)
data = pd.merge(data, now_daily_returns, left_index=True, right_index=True)
data = pd.merge(data, stx_daily_returns, left_index=True, right_index=True)

stocks = ['amd', 'ftv', 'it', 'now', 'stx']

data.columns = stocks

#%%
for i in returns:
    fig = plt.figure()
    plt.tight_layout()
    ax1 = fig.add_axes([0.1,0.1,0.8,0.8])
    ax1.plot(i)
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Percent")
    ax1.set_title("Daily returns data")
    plt.xticks(rotation=30)
    plt.show()

#%%
for i in returns:
    fig = plt.figure()
    ax1 = fig.add_axes([0.1,0.1,0.8,0.8])
    i.plot.hist(bins = 15)
    ax1.set_xlabel("Daily returns %")
    ax1.set_ylabel("Percent")
    ax1.set_title("Daily returns data")
    ax1.text(-0.35,200,"Extreme Low\nreturns")
    ax1.text(0.25,200,"Extreme High\nreturns")
    plt.show()

#%%
for i in returns:
    temp = (i + 1).cumprod()
    fig = plt.figure()
    ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    temp.plot()
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Growth of $1 investment")
    ax1.set_title("Daily cumulative returns data")
    plt.show()

#%%
fig = plt.figure()
ax1 = fig.add_subplot(321)
ax2 = fig.add_subplot(322)
ax3 = fig.add_subplot(323)
ax4 = fig.add_subplot(324)
ax5 = fig.add_subplot(325)
ax1.plot(amd['Adj Close'])
ax1.set_title("AMD")
ax2.plot(ftv['Adj Close'])
ax2.set_title("FTV")
ax3.plot(it['Adj Close'])
ax3.set_title("IT")
ax4.plot(now['Adj Close'])
ax4.set_title("NOW")
ax5.plot(stx['Adj Close'])
ax5.set_title("STX")
plt.tight_layout()
ax1.xaxis.set_major_locator(plt.MaxNLocator(5))
ax2.xaxis.set_major_locator(plt.MaxNLocator(5))
ax3.xaxis.set_major_locator(plt.MaxNLocator(5))
ax4.xaxis.set_major_locator(plt.MaxNLocator(5))
ax5.xaxis.set_major_locator(plt.MaxNLocator(5))
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=30, ha='right')
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=30, ha='right')
ax3.set_xticklabels(ax1.get_xticklabels(), rotation=30, ha='right')
ax4.set_xticklabels(ax1.get_xticklabels(), rotation=30, ha='right')
ax5.set_xticklabels(ax1.get_xticklabels(), rotation=30, ha='right')
plt.show()

#%%
fig = plt.figure()
(data + 1).cumprod().plot()
plt.show()

#%%
stocks = data.columns

#%%
# Calculate basic summary statistics for individual stocks
stock_volatility = data.std()
stock_return = data.mean()

#%%
# Create an empty model
m = Model('portfolio')

#%%
# Add a variable for each stock
vars = pd.Series(m.addVars(stocks), index=stocks)

#%%
# Objective is to minimize risk (squared).  This is modeled using the
# covariance matrix, which measures the historical correlation between stocks.
sigma = data.cov()
portfolio_risk = sigma.dot(vars).dot(vars)
m.setObjective(portfolio_risk, GRB.MINIMIZE)

#%%
# Fix budget with a constraint
m.addConstr(vars.sum() == 1, 'budget')

#%%
# Optimize model to find the minimum risk portfolio
m.setParam('OutputFlag', 0)
m.optimize()

#%%
# Create an expression representing the expected return for the portfolio
portfolio_return = stock_return.dot(vars)

#%%
# Display minimum risk portfolio
print('Minimum Risk Portfolio:\n')
for v in vars:
    if v.x > 0:
        print('\t%s\t: %g' % (v.varname, v.x))
minrisk_volatility = sqrt(portfolio_risk.getValue())
print('\nVolatility      = %g' % minrisk_volatility)
minrisk_return = portfolio_return.getValue()
print('Expected Return = %g' % minrisk_return)

#%%
# Add (redundant) target return constraint
target = m.addConstr(portfolio_return >= minrisk_return, 'target')

#%%
# Solve for efficient frontier by varying target return
frontier = pd.Series()
for r in np.linspace(stock_return.min(), stock_return.max(), 100):
    target.rhs = r
    m.optimize()
    frontier.loc[sqrt(portfolio_risk.getValue())] = r

# Plot volatility versus expected return for individual stocks
ax = plt.gca()
ax.scatter(x=stock_volatility, y=stock_return,
           color='Blue', label='Individual Stocks')
for i, stock in enumerate(stocks):
    ax.annotate(stock, (stock_volatility[i], stock_return[i]))

# Plot volatility versus expected return for minimum risk portfolio
ax.scatter(x=minrisk_volatility, y=minrisk_return, color='DarkGreen')
ax.annotate('Minimum\nRisk\nPortfolio', (minrisk_volatility, minrisk_return),
            horizontalalignment='right')

# Plot efficient frontier
frontier.plot(color='DarkGreen', label='Efficient Frontier', ax=ax)

# Format and display the final plot
ax.axis([0.005, 0.06, -0.02, 0.025])
ax.set_xlabel('Volatility (standard deviation)')
ax.set_ylabel('Expected Return')
ax.legend()
ax.grid()
plt.show()

#%%
m.optimize()
for v in m.getVars():
    if v.X != 0:
        print("%s %f" % (v.Varname, v.X))

