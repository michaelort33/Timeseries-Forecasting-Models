
import os
import csv
import pytz
from datetime import datetime
import pandas as pd

# You need to install the catalyst library and its dependencies for this
from catalyst.api import record, symbol, symbols
from catalyst.utils.run_algo import run_algorithm


# Most this code is from the Catalyst tutorial example
def initialize(context):
    # Portfolio assets list
    context.asset = symbol('btc_usdt')  # Bitcoin on Poloniex

    # Create an empty DataFrame to store results
    context.pricing_data = pd.DataFrame()


def handle_data(context, data):
    # Variables to record for a given asset: price and volume
    # Other options include 'open', 'high', 'open', 'close'
    # 'price' equals 'close'
    current = data.history(context.asset, ['price', 'volume'], 1, '1T')

    # Append the current information to the pricing_data DataFrame
    context.pricing_data = context.pricing_data.append(current)


def analyze(context=None, results=None):
    # Save pricing data to a CSV file
    filename = os.path.abspath('') + 'minute_market_data'
    context.pricing_data.to_csv(filename + '.csv')


''' Bitcoin data is available on Poloniex since 2015-3-1.
     Dates vary for other tokens.
     The frun_algorithm function is where you can enter parameters like
     data_frequency, start, and end. 
     Note: Besides for running the script with run_algorithm, you can also run the script in a command-line interface like this:
     catalyst run -f read_minute_data.py --start 2018-3-3 --end 2018-4-3 --capital-base 100000 -x binance -c usdt -o minute.csv --data-frequency minute
'''
start = datetime(2018, 1, 3, 0, 0, 0, 0, pytz.utc)
end = datetime(2018, 6, 3, 0, 0, 0, 0, pytz.utc)
results = run_algorithm(
    initialize=initialize,
    handle_data=handle_data,
    analyze=analyze,
    start=start,
    end=end,
    exchange_name='binance',
    data_frequency='minute',
    quote_currency='usdt',
    capital_base=10000)
