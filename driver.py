from scripts import feedforward_nn
from scripts import data_process as dp

inputs = dp.create_data()
feedforward_nn.feedforward_neural_network(inputs)
