import matplotlib.pyplot as plt
import backtrader as bt
import tensorflow as tf
from scripts import data_process as dp
import pickle


class FeedforwardStrategy(bt.Strategy):
    def log(self, txt, dt=None):
        ''' Logging function fot this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        # Keep a reference to the "close" line in the data[0] dataseries
        self.dataclose = self.datas[0].close

        # To keep track of pending orders and buy price/commission
        self.order = None
        self.buyprice = None
        self.buycomm = None

        print("Loading pre-trained model...")
        self.sess = tf.Session()
        self.saver = tf.train.import_meta_graph("data/model/feedforward/feedforward.ckpt.meta")
        self.saver.restore(self.sess, tf.train.latest_checkpoint('data/model/feedforward'))
        print("Model loaded...")

        self.graph = tf.get_default_graph()
        self.x = self.graph.get_tensor_by_name('input:0')
        self.prediction = self.graph.get_tensor_by_name('output:0')
        _, _, _, _, self.oil_price, self.stock_price = dp.create_data()

        self.prediction_graph()

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return
        # Check if an order has been completed
        # Attention: broker could reject order if not enougth cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    'BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                    (order.executed.price,
                     order.executed.value,
                     order.executed.comm))
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:  # Sell
                self.log('SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                         (order.executed.price,
                          order.executed.value,
                          order.executed.comm))
            self.bar_executed = len(self)
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')
        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        self.log('OPERATION PROFIT, GROSS %.2f, NET %.2f' %
                 (trade.pnl, trade.pnlcomm))

    def prediction_graph(self):
        predictions = []
        for i in self.oil_price:
            predictions.append(self.sess.run(self.prediction, feed_dict={self.x: [[i]]})[0][0])
        plt.plot(self.oil_price.values, label='Oil Prices')
        plt.plot(self.stock_price.values, label='Stock Prices')
        plt.plot(predictions, label="Predictions")
        plt.legend()
        plt.ylabel('Price')
        plt.xlabel('Date')
        plt.show()

    def next(self):
        # Simply log the closing price of the series from the reference
        #self.log('Close, %.2f' % self.dataclose[0])
        # Check if an order is pending ... if yes, we cannot send a 2nd one
        if self.order:
            return
        # Check if we are in the market
        if not self.position:
            # Not yet ... we MIGHT BUY if ...
            if self.datas[0].datetime.date(0) in self.oil_price:
                if self.sess.run(self.prediction,
                                 feed_dict={self.x: [[self.oil_price[self.datas[0].datetime.date(0)]]]}) > self.dataclose[0]:
                    # previous close less than the previous close
                    # BUY, BUY, BUY!!! (with default parameters)
                    self.log('BUY CREATE, %.2f' % self.dataclose[0])
                    # Keep track of the created order to avoid a 2nd order
                    self.order = self.buy()

        else:
            # Already in the market ... we might sell
            if len(self) >= (self.bar_executed + 2):
                # SELL, SELL, SELL!!! (with all possible default parameters)
                self.log('SELL CREATE, %.2f' % self.dataclose[0])
                # Keep track of the created order to avoid a 2nd order
                self.order = self.sell()