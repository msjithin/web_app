"""
following
https://www.learndatasci.com/tutorials/python-finance-part-yahoo-finance-api-pandas-matplotlib/
https://www.learndatasci.com/tutorials/python-finance-part-2-intro-quantitative-trading-strategies/
https://www.learndatasci.com/tutorials/python-finance-part-3-moving-average-trading-strategy/

"""

import pandas_datareader.data as web
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import datetime as dt
import plotly.express as px
import streamlit as st 

import seaborn as sns
sns.set(style='darkgrid', context='talk', palette='Dark2')
my_year_month_fmt = mdates.DateFormatter('%m/%y')
st.set_page_config(page_title='Stock web app')

# web = pd.read_pickle('./data.pkl')
# web.head(10)

# Define the instruments to download. We would like to see Apple, Microsoft and the S&P500 index.
tickers = ['AAPL', 'MSFT', '^GSPC']


# User pandas_reader.data.DataReader to load the desired data. As simple as that.
#panel_data = data.DataReader(name='', 'yahoo', start_date, end_date)
# We would like all available data from 01/01/2000 until today.
start_date = dt.datetime(2000, 1, 1)
end_date = dt.date.today()

panel_data = web.DataReader(tickers, 'yahoo', start_date, end_date)
#print(panel_data.head(9))
# Getting just the adjusted closing prices. This will return a Pandas DataFrame
# The index in this DataFrame is the major index of the panel_data.
close = panel_data['Close']
#print(all_weekdays)
#print(close.head(10))
# Getting all weekdays between 01/01/2000 and 12/31/2016
all_weekdays = pd.date_range(start=start_date, end=end_date, freq='B')

# How do we align the existing prices in adj_close with our new set of dates?
# All we need to do is reindex close using all_weekdays as the new index
close = close.reindex(all_weekdays)

# Reindexing will insert missing values (NaN) for the dates that were not present
# in the original set. To cope with this, we can fill the missing by replacing them
# with the latest available price for each instrument.
close = close.fillna(method='ffill')
#print(close.head(10))
#print(close.describe())


def get_time_series(name='MSFT'):
    # Get the MSFT timeseries. This now returns a Pandas Series object indexed by date.
    msft = close.loc[:, name]

    # Calculate the 20 and 100 days moving averages of the closing prices
    short_rolling_msft = msft.rolling(window=20).mean()
    long_rolling_msft = msft.rolling(window=100).mean()

    # Plot everything by leveraging the very powerful matplotlib package
    fig_ts, ax_ts = plt.subplots(figsize=(12,5))

    ax_ts.plot(msft.index, msft, label=name)
    ax_ts.plot(short_rolling_msft.index, short_rolling_msft, label='20 days rolling')
    ax_ts.plot(long_rolling_msft.index, long_rolling_msft, label='100 days rolling')

    ax_ts.set_xlabel('Date')
    ax_ts.set_ylabel('Adjusted closing price ($)')
    ax_ts.legend()
    return fig_ts

def get_line_plotly(name='MSFT'):
    # Get the MSFT timeseries. This now returns a Pandas Series object indexed by date.
    stock_ts = close.loc[:, name]

    # Calculate the 20 and 100 days moving averages of the closing prices
    short_rolling_msft = stock_ts.rolling(window=20).mean()
    long_rolling_msft = stock_ts.rolling(window=100).mean()
    #print(short_rolling_msft)
    fig = px.line(stock_ts, width=900
                            , height=500
                            )
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    )
    fig.update_xaxes(title_text='Date')
    fig.update_yaxes(title_text='Value ($)')
    return fig
#get_time_series('MSFT')

# Calculating the short-window moving average
short_rolling = close.rolling(window=20).mean()

# Calculating the short-window moving average
long_rolling = close.rolling(window=100).mean()

# Relative returns
returns = close.pct_change(1)

# Log returns - First the logarithm of the prices is taken and the the difference of consecutive (log) observations
log_returns = np.log(close).diff()

def plot_return():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12,7))

    for c in log_returns:
        ax1.plot(log_returns.index, log_returns[c].cumsum(), label=str(c))

    ax1.set_ylabel('Cumulative log returns')
    ax1.legend(loc='best')

    for c in log_returns:
        ax2.plot(log_returns.index, 100*(np.exp(log_returns[c].cumsum()) - 1), label=str(c))

    ax2.set_ylabel('Total relative returns (%)')
    ax2.legend(loc='best')

    return fig

#plot_return()


# Last day returns. Make this a column vector
r_t = log_returns.tail(1).transpose()
#print(r_t)

# Weights as defined above
weights_vector = pd.DataFrame(1 / 3, index=r_t.index, columns=r_t.columns)
# Total log_return for the portfolio is:
portfolio_log_return = weights_vector.transpose().dot(r_t)

weights_matrix = pd.DataFrame(1 / 3, index=close.index, columns=close.columns)
weights_matrix.tail()

# Initially the two matrices are multiplied. Note that we are only interested in the diagonal, 
# which is where the dates in the row-index and the column-index match.
temp_var = weights_matrix.dot(log_returns.transpose())
temp_var.head().iloc[:, 0:5]

# The numpy np.diag function is used to extract the diagonal and then
# a Series is constructed using the time information from the log_returns index
portfolio_log_returns = pd.Series(np.diag(temp_var), index=log_returns.index)
portfolio_log_returns.tail()

total_relative_returns = (np.exp(portfolio_log_returns.cumsum()) - 1)

def portfolio_returns():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9))

    ax1.plot(portfolio_log_returns.index, portfolio_log_returns.cumsum())
    ax1.set_ylabel('Portfolio cumulative log returns')

    ax2.plot(total_relative_returns.index, 100 * total_relative_returns)
    ax2.set_ylabel('Portfolio total relative returns (%)')
    return fig
    

# Calculating the time-related parameters of the simulation
days_per_year = 52 * 5
total_days_in_simulation = close.shape[0]
number_of_years = total_days_in_simulation / days_per_year

# The last data point will give us the total portfolio return
total_portfolio_return = total_relative_returns[-1]
# Average portfolio return assuming compunding of returns
average_yearly_return = (1 + total_portfolio_return)**(1 / number_of_years) - 1

#print('Total portfolio return is: ' +
#      '{:5.2f}'.format(100 * total_portfolio_return) + '%')
#print('Average yearly return is: ' +
#      '{:5.2f}'.format(100 * average_yearly_return) + '%')

start_date = dt.datetime(2015, 1, 1)
end_date = dt.datetime(2016, 12, 31)

def plot_price(name='MSFT', start=start_date, end=end_date):
    fig, ax = plt.subplots(figsize=(12,5))

    ax.plot(close.loc[start:end, :].index, close.loc[start:end, name], label='Price')
    ax.plot(long_rolling.loc[start:end, :].index, long_rolling.loc[start:end, name], label = '100-days SMA')
    ax.plot(short_rolling.loc[start:end, :].index, short_rolling.loc[start:end, name], label = '20-days SMA')

    ax.legend(loc='best')
    ax.set_ylabel('Price in $')
    ax.xaxis.set_major_formatter(my_year_month_fmt)
    return fig

#plot_price()
# Using Pandas to calculate a 20-days span EMA. adjust=False specifies that we are interested in the recursive calculation mode.

ema_short = close.ewm(span=20, adjust=False).mean()

def plot_ema(name='MSFT', start=start_date, end=end_date):
    fig, ax = plt.subplots(figsize=(15,9))

    ax.plot(close.loc[start:end, :].index, close.loc[start:end, name], label='Price')
    ax.plot(ema_short.loc[start:end, :].index, ema_short.loc[start:end, name], label = 'Span 20-days EMA')
    ax.plot(short_rolling.loc[start:end, :].index, short_rolling.loc[start:end, name], label = '20-days SMA')

    ax.legend(loc='best')
    ax.set_ylabel('Price in $')
    ax.xaxis.set_major_formatter(my_year_month_fmt)
    return fig

# Taking the difference between the prices and the EMA timeseries
trading_positions_raw = close - ema_short

# Taking the sign of the difference to determine whether the price or the EMA is greater and then multiplying by 1/3
trading_positions = trading_positions_raw.apply(np.sign) * 1/3

# Lagging our trading signals by one day.
trading_positions_final = trading_positions.shift(1)

def plot_trading_position(name='MSFT', start=start_date, end=end_date):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5))

    ax1.plot(close.loc[start:end, :].index, close.loc[start:end, name], label='Price')
    ax1.plot(ema_short.loc[start:end, :].index, ema_short.loc[start:end, name], label = 'Span 20-days EMA')

    ax1.set_ylabel('$')
    ax1.legend(loc='best')
    ax1.xaxis.set_major_formatter(my_year_month_fmt)

    ax2.plot(trading_positions_final.loc[start:end, :].index, trading_positions_final.loc[start:end, 'MSFT'], 
            label='Trading position')

    ax2.set_ylabel('Trading position')
    ax2.xaxis.set_major_formatter(my_year_month_fmt)
    return fig

# Log returns - First the logarithm of the prices is taken and the the difference of consecutive (log) observations
asset_log_returns = np.log(close).diff()
strategy_asset_log_returns = trading_positions_final * asset_log_returns

# Get the cumulative log-returns per asset
cum_strategy_asset_log_returns = strategy_asset_log_returns.cumsum()

# Transform the cumulative log returns to relative returns
cum_strategy_asset_relative_returns = np.exp(cum_strategy_asset_log_returns) - 1

def plot_best_returns():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12,7))

    for c in asset_log_returns:
        ax1.plot(cum_strategy_asset_log_returns.index, cum_strategy_asset_log_returns[c], label=str(c))

    ax1.set_ylabel('Cumulative log-returns')
    ax1.legend(loc='best')
    ax1.xaxis.set_major_formatter(my_year_month_fmt)

    for c in asset_log_returns:
        ax2.plot(cum_strategy_asset_relative_returns.index, 100*cum_strategy_asset_relative_returns[c], label=str(c))

    ax2.set_ylabel('Total relative returns (%)')
    ax2.legend(loc='best')
    ax2.xaxis.set_major_formatter(my_year_month_fmt)
    return fig


# Total strategy relative returns. This is the exact calculation.
cum_relative_return_exact = cum_strategy_asset_relative_returns.sum(axis=1)

# Get the cumulative log-returns per asset
cum_strategy_log_return = cum_strategy_asset_log_returns.sum(axis=1)

# Transform the cumulative log returns to relative returns. This is the approximation
cum_relative_return_approx = np.exp(cum_strategy_log_return) - 1

def total_retuns():
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(cum_relative_return_exact.index, 100*cum_relative_return_exact, label='Exact')
    ax.plot(cum_relative_return_approx.index, 100*cum_relative_return_approx, label='Approximation')

    ax.set_ylabel('Total cumulative relative returns (%)')
    ax.legend(loc='best')
    ax.xaxis.set_major_formatter(my_year_month_fmt)

    return fig