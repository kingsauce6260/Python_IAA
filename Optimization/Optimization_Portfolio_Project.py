
#%%
import pandas as pd
import numpy as np
from gurobipy import *
import matplotlib.pyplot as plt
from math import sqrt
#%%
data = pd.read_csv(r'/Users/thomasgow/Documents/IAA/Optimization/Homework/Data/all_stocks_new.csv')

#%%
data['Date'] = pd.to_datetime(data.Date)
data['Date'] = data['Date'].dt.strftime('%-m/%-d/%-y')

#%%
data.set_index('Date', inplace=True)

#%% Plotting change in return daily for  all stocks
fig = plt.figure()
ax1 = fig.add_subplot(321)
ax2 = fig.add_subplot(322)
ax3 = fig.add_subplot(323)
ax4 = fig.add_subplot(324)
ax5 = fig.add_subplot(325)
ax1.hist(data['AMD'], bins=15)
ax1.set_title("AMD")
ax2.hist(data['FTV'], bins=15)
ax2.set_title("FTV")
ax3.hist(data['IT'], bins=15)
ax3.set_title("IT")
ax4.hist(data['NOW'], bins=15)
ax4.set_title("NOW")
ax5.hist(data['STX'], bins=15)
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

plt.draw()
fig.savefig('/Users/thomasgow/Documents/IAA/Optimization/Homework/variance_stock_hist.png', dpi=100)

#%% Plotting change in return daily for  all stocks
fig = plt.figure()
ax1 = fig.add_subplot(321)
ax2 = fig.add_subplot(322)
ax3 = fig.add_subplot(323)
ax4 = fig.add_subplot(324)
ax5 = fig.add_subplot(325)
ax1.plot(data['AMD'])
ax1.set_title("AMD")
ax2.plot(data['FTV'])
ax2.set_title("FTV")
ax3.plot(data['IT'])
ax3.set_title("IT")
ax4.plot(data['NOW'])
ax4.set_title("NOW")
ax5.plot(data['STX'])
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

plt.draw()
fig.savefig('/Users/thomasgow/Documents/IAA/Optimization/Homework/variance_stock.png', dpi=100)

#%% Plotting variance per stock
fig, ax = plt.subplots()
ax.plot(data)
ax.legend()
plt.tight_layout()
ax.xaxis.set_major_locator(plt.MaxNLocator(5))
ax.set_xticklabels(ax1.get_xticklabels(), rotation=30, ha='right')
plt.show()

plt.draw()
fig.savefig('/Users/thomasgow/Documents/IAA/Optimization/Homework/variance_stock_together.png', dpi=100)

#%% Plotting cumulative return for all stocks
fig = plt.figure()
(data + 1).cumprod().plot()
plt.draw()
fig.savefig('/Users/thomasgow/Documents/IAA/Optimization/Homework/cumulative_stock.png', dpi=100)
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
# convert Series to Python strings
data = data.astype(np.float16)


data['AMD'] = pd.to_numeric(data['AMD'])
sigma = data.cov()
portfolio_risk = sigma.dot(vars).dot(vars)
m.setObjective(portfolio_risk, GRB.MINIMIZE)

#%%
# Fix budget with a constraint
m.addConstr(vars.sum() == 1, 'budget')
#%%
# Create an expression representing the expected return for the portfolio
portfolio_return = stock_return.dot(vars)
m.addConstr(portfolio_return >= 0.0005, 'target')

m.update()
#%%
# Optimize model to find the minimum risk portfolio
m.setParam('OutputFlag', 0)
m.optimize()

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
target = m.addConstr(portfolio_return >= 0.0005, 'target')

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
plt.draw()
fig.savefig('/Users/thomasgow/Documents/IAA/Optimization/Homework/volatility_expected_return.png', dpi=100)
