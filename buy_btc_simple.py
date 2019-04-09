import matplotlib.pyplot as plt
from logbook import Logger

import pandas as pd
from catalyst import run_algorithm
from catalyst.api import (symbol,order,)


NAMESPACE = 'buy_simple'
log = Logger(NAMESPACE)

def initialize(context):
    context.asset = symbol('btc_usds')


def handle_data(context, data):
    order(context.asset, 1)


if __name__ == '__main__':

    run_algorithm(
            capital_base=100000,
            data_frequency='minute',
            initialize=initialize,
            handle_data=handle_data,
            exchange_name='binance',
            algo_namespace=NAMESPACE,
            live=True,
            end=pd.to_datetime('2019-9-23', utc=True),

    )
