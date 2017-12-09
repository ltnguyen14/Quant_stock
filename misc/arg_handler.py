import argparse
from pipeline import backtest
from pipeline.backtest import TestStrategy
from pipeline.strategies.ff_strat import FeedforwardStrategy
from scripts import feedforward_nn, recurrent_lstm
from scripts import data_process as dp
from misc.graph import graph


def arg_parser():
    parser = argparse.ArgumentParser(description="Stock prediction model", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-b', '--btest', help='Run backtest with the model',
            default=None, choices=['test', 'feedforward', 'recurrent', 'cnn'])
    parser.add_argument('-t', '--train', help='Train a model', default=None,
            choices=['feedforward', 'recurrent', 'cnn'])
    parser.add_argument('-g', '--graph', help='Graph', nargs='*', choices=['test', 'feedforward', 'recurrent', 'cnn'])

    args = parser.parse_args()
    return args


class InputHandler:
    def __init__(self, inputs):
        self.inputs = inputs
        if self.inputs.train:
            self.train(self.inputs.train)
        if self.inputs.btest:
            if self.inputs.btest == "test":
                self.run(TestStrategy)
            elif self.inputs.btest == "feedforward":
                self.run(FeedforwardStrategy)
        if self.inputs.graph:
            if len(self.inputs.graph) != 0:
                graph(self.inputs.graph)
            else:
                graph(['feedforward', 'recurrent', 'cnn'])

    @staticmethod
    def run(strategy):
        backtest_obj = backtest.Backtest(stock_symbol='XOM', strategy=strategy)
        backtest_obj.run(plot=True)

    @staticmethod
    def train(model):
        inputs = dp.create_data()
        if model == "feedforward":
            feedforward_nn.feedforward_neural_network(inputs)
        elif model == "recurrent":
            recurrent_lstm.recurrent_neural_network(inputs)
