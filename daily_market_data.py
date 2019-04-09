import os
import pytz
from datetime import datetime

from catalyst.api import record, symbol, symbols
from catalyst.utils.run_algo import run_algorithm

def initialize(context):
    # Portfolio assets list
    context.asset = symbol('btc_usdt') # Bitcoin on Poloniex

def handle_data(context, data):
    # Variables to record for a given asset: price and volume
    price = data.current(context.asset, 'price')
    volume = data.current(context.asset, 'volume')
    record(price=price, volume=volume)

def analyze(context=None, results=None):
    # Generate DataFrame with Price and Volume only
    data = results[['price','volume']]

    # Save results in CSV file
    filename = os.path.abspath('')+'daily_market_data'
    data.to_csv(filename + '.csv')

''' Bitcoin data is available on Poloniex since 2015-3-1.
     Dates vary for other tokens. In the example below, we choose the
     full month of July of 2017.
'''
start = datetime(2017, 1, 1, 0, 0, 0, 0, pytz.utc)
end = datetime(2017, 7, 31, 0, 0, 0, 0, pytz.utc)
results = run_algorithm(initialize=initialize,
                                handle_data=handle_data,
                                analyze=analyze,
                                start=start,
                                end=end,
                                exchange_name='poloniex',
                                capital_base=10000,
                                quote_currency = 'usdt')