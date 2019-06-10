import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from logbook import Logger
from datetime import datetime
import pytz

from catalyst import run_algorithm
from catalyst.api import (record, symbol, order_target_percent, )
from catalyst.exchange.utils.stats_utils import extract_transactions

import predictor_lstm


NAMESPACE = 'first_test'
log = Logger(NAMESPACE)


def initialize(context):
    context.i = 0
    context.asset = symbol('btc_usdt')
    context.base_price = None
    context.set_commission(maker=0.001, taker=0.001)
    context.set_slippage(slippage=0.001)
    context.last_prediction = 1000000
    context.prediction = 0
    context.num_hours = 0
    context.train = pd.read_csv('../input/jan_june_btc_minute.csv')


def handle_data(context, data):
    # define the windows for the moving averages
    long_window = 240

    # Skip as many bars as long_window to properly compute the average
    context.i += 1
    if context.i < long_window or context.i % 60 != 0:
        return

    # Compute moving averages calling data.history() for each
    # moving average with the appropriate parameters. We choose to use
    # minute bars for this simulation -> freq="1m"
    # Returns a pandas dataframe.

    long_data = data.history(context.asset,
                             'price',
                             bar_count=long_window,
                             frequency="1T",
                             )

    def read_catalyst_history(df):

        df = pd.DataFrame(df)
        df.columns = ['price']

        # convert to date type
        df.index = df.index.astype('datetime64[ns]')

        return df

    my_price_data = read_catalyst_history(long_data)


    # Create features
    test_x = predictor_lstm.create_features(my_price_data)

    # read training data for scaling inversion
    train = predictor_lstm.read(context.train)

    # create features from training data for scaling inversion
    train_x = predictor_lstm.create_features(train)

    train_x = train_x.iloc[:-1, :]

    train_y = predictor_lstm.create_y(train.price)


    predictions_test = predictor_lstm.get_predictions(train_x, test_x, train_y)

    context.last_prediction = context.prediction
    context.prediction = predictions_test

    print(context.prediction)
    print(context.last_prediction)

    # Save asset price
    price = data.current(context.asset, 'price')

    # If base_price is not set, we use the current value. This is the
    # price at the first bar which we reference to calculate price_change.
    if context.base_price is None:
        context.base_price = price
    price_change = (price - context.base_price) / context.base_price

    # Save values for later inspection
    record(price=price,
           cash=context.portfolio.cash,
           price_change=price_change)

    # Since we are using limit orders, some orders may not execute immediately
    # we wait until all orders are executed before considering more trades.
    orders = context.blotter.open_orders
    if len(orders) > 0:
        return

    # Exit if we cannot trade
    if not data.can_trade(context.asset):
        return

    # We check what's our position on our portfolio and trade accordingly
    pos_amount = context.portfolio.positions[context.asset].amount

    def trade_decision(prediction, last_prediction, pos_amount):

        # Trading logic
        if (prediction/last_prediction) > 1.002 and pos_amount == 0:
            # we buy 100% of our portfolio for this asset
            order_target_percent(context.asset, 1)
        elif (prediction/last_prediction) < 0.998 and pos_amount > 0:
            # we sell all our positions for this asset
            order_target_percent(context.asset, 0)

    trade_decision(context.prediction, context.last_prediction, pos_amount)
    context.num_hours += 1
    print(context.num_hours)
    print(context.portfolio.cash)


def analyze(context, perf):
    # Get the quote_currency that was passed as a parameter to the simulation
    exchange = list(context.exchanges.values())[0]
    quote_currency = exchange.quote_currency.upper()

    # First chart: Plot portfolio value using quote_currency
    ax1 = plt.subplot(411)
    perf.loc[:, ['portfolio_value']].plot(ax=ax1)
    ax1.set_ylabel('Portfolio Value\n({})'.format(quote_currency))
    start, end = ax1.get_ylim()
    ax1.yaxis.set_ticks(np.arange(start, end, (end - start) / 5))

    # Second chart: Plot asset price, moving averages and buys/sells
    ax2 = plt.subplot(412, sharex=ax1)
    perf.loc[:, ['price']].plot(
        ax=ax2,
        label='Price')
    ax2.set_ylabel('{asset}\n({quote})'.format(
        asset=context.asset.symbol,
        quote=quote_currency
    ))
    start, end = ax2.get_ylim()
    ax2.yaxis.set_ticks(np.arange(start, end, (end - start) / 5))

    transaction_df = extract_transactions(perf)
    if not transaction_df.empty:
        buy_df = transaction_df[transaction_df['amount'] > 0]
        sell_df = transaction_df[transaction_df['amount'] < 0]
        ax2.scatter(
            buy_df.index.to_pydatetime(),
            perf.loc[buy_df.index, 'price'],
            marker='^',
            s=100,
            c='green',
            label=''
        )
        ax2.scatter(
            sell_df.index.to_pydatetime(),
            perf.loc[sell_df.index, 'price'],
            marker='v',
            s=100,
            c='red',
            label=''
        )

    # Third chart: Compare percentage change between our portfolio
    # and the price of the asset
    ax3 = plt.subplot(413, sharex=ax1)
    perf.loc[:, ['algorithm_period_return', 'price_change']].plot(ax=ax3)
    ax3.set_ylabel('Percent Change')
    start, end = ax3.get_ylim()
    ax3.yaxis.set_ticks(np.arange(start, end, (end - start) / 5))

    # Fourth chart: Plot our cash
    ax4 = plt.subplot(414, sharex=ax1)
    perf.cash.plot(ax=ax4)
    ax4.set_ylabel('Cash\n({})'.format(quote_currency))
    start, end = ax4.get_ylim()
    ax4.yaxis.set_ticks(np.arange(0, end, end / 5))

    plt.show()


if __name__ == '__main__':
    start = datetime(2018, 7, 13, 0, 0, 0, 0, pytz.utc)
    end = datetime(2018, 8, 15, 0, 0, 0, 0, pytz.utc)

    run_algorithm(
        capital_base=1000,
        start=start,
        end=end,
        data_frequency='minute',
        initialize=initialize,
        handle_data=handle_data,
        analyze=analyze,
        exchange_name='binance',
        live=False,
        algo_namespace=NAMESPACE,
        quote_currency='usdt',
        simulate_orders=True,
    )
