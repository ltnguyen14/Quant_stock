import argparse
from pipeline import backtest

def arg_parser():
    parser = argparse.ArgumentParser(description="Stock prediction model", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-r', '--run', help='Run backtest with the model', default=None)
    parser.add_argument('-f', '--file_name', help='Enter the text file name', default=None)
    parser.add_argument('-s','--sword', help='Enter the stop words file name', default=None)
    parser.add_argument('-a','--algorithm', help='Choose the algorithm', choices=['heapq','counter','sorted'], default='heapq')
    parser.add_argument('-g', '--graphical', help='Graphical Histogram', action="store_true")
    parser.add_argument('-r', '--repeat', help='Repeat Time', type = int, default = 1)

    args = parser.parse_args()
    return args

class inputHandler:
    def __init__(self, inputs):
        self.inputs = inputs
        if self.inputs.run:
            self.run()
    def run():
        backtest_obj = backtest.backtest('XOM')
        backtest_obj.run(True)
