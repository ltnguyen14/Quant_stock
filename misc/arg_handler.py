import argparse
from pipeline import backtest
from pipeline.backtest import TestStrategy
#from pipeline.strategies import ff_strat
from scripts import feedforward_nn
from scripts import data_process as dp
def arg_parser():
    parser = argparse.ArgumentParser(description="Stock prediction model", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-b', '--btest', help='Run backtest with the model',
            default=None, choices=['test', 'feedforward', 'rnn', 'cnn'])
    parser.add_argument('-t', '--train', help='Train a model', default=None,
            choices=['feedforward', 'rnn', 'cnn'])
    parser.add_argument('-s','--sword', help='Enter the stop words file name', default=None)
    parser.add_argument('-a','--algorithm', help='Choose the algorithm', choices=['heapq','counter','sorted'], default='heapq')
    parser.add_argument('-g', '--graphical', help='Graphical Histogram', action="store_true")
    #parser.add_argument('-r', '--repeat', help='Repeat Time', type = int, default = 1)

    args = parser.parse_args()
    return args

class inputHandler:
    def __init__(self, inputs):
        self.inputs = inputs
        if self.inputs.btest:
            if self.inputs.btest == "test":
                self.run(TestStrategy)
            elif self.inputs.btest == "feedforward":
                self.run(ff_strat)

        if self.inputs.train:
            self.train(self.inputs.train)

    def run(self, strategy):
        backtest_obj = backtest.backtest(stock_symbol='XOM', strategy=strategy)
        backtest_obj.run(plot=True)

    def train(self, model):
        if model == "feedforward":
            inputs = dp.create_data()
            feedforward_nn.feedforward_neural_network(inputs)
