import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pprint import pprint
from googlefinance.client import get_price_data, get_prices_data, get_prices_time_data
import datetime

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 1
batch_size = 100

x = tf.placeholder('float')
y = tf.placeholder('float')

def add_lag(dataset_1, dataset_2, lag):
    if lag != 0:
        dataset_2 = dataset_2[lag:]
        dataset_1 = dataset_1[:-lag]

    return dataset_1, dataset_2

def create_data():
    url = "CME_CL1.csv"
    crude_oil = pd.read_csv(url, index_col=0, parse_dates=True)
    crude_oil.sort_index(inplace=True)
    crude_oil_last = crude_oil['Last']

    param = {
            'q': 'XOM',
            'i': 86400,
            'x': "NYSE",
            'p': '40Y'
    }
    df = get_price_data(param)
    df.set_index(df.index.normalize(), inplace=True)
    stock_close = df['Close']

    oil_price, stock_price = crude_oil_last.align(stock_close, join='inner')

    split_index = int(3*len(oil_price)/4)
    oil_train = oil_price.iloc[:split_index]
    stock_train = oil_price.iloc[:split_index]

    oil_test = oil_price.iloc[split_index:]
    stock_test = oil_price.iloc[split_index:]

    return oil_train, stock_train, oil_test, stock_test

def neural_network_model(data):
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([1, n_nodes_hl1])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                    'biases':tf.Variable(tf.random_normal([n_classes])),}

    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.add(tf.matmul(l3,output_layer['weights']),
            output_layer['biases'])

    return output

def refine_input_with_lag(oil_train, stock_train, oil_test, stock_test):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.square(y-prediction, name="cost") )
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    #Adding lag
    all_lag_losses = []
    lag_range = 30
    lag_epoch_num = 50
    for i in range(lag_range):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            oil_lag, stock_lag = add_lag(oil_train, stock_train, i)
            for epoch in range(lag_epoch_num):
                lag_loss = 0
                for (X,Y) in zip(oil_lag.values, stock_lag.values):
                    _, c = sess.run([optimizer, cost], feed_dict={x: [[X]], y: [[Y]]})
                    lag_loss += c
                print('Lag', i, 'epoch', epoch, 'loss:', lag_loss)
            all_lag_losses.append(lag_loss)
    lag = all_lag_losses.index(min(all_lag_losses))
    oil_train, stock_train = add_lag(oil_train, stock_train, lag)
    oil_test, stock_test = add_lag(oil_test, stock_test, lag)
    print("The best lag is:", lag)
    return oil_train, stock_train, oil_test, stock_test

def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.square(y-prediction, name="cost") )
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    oil_train, stock_train, oil_test, stock_test = create_data()

    hm_epochs = 400
    oil_train, stock_train, oil_test, stock_test = refine_input_with_lag(oil_train, stock_train, oil_test, stock_test)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
       #Running neural net
        for epoch in range(hm_epochs):
            epoch_loss = 0
            for (X,Y) in zip(oil_train.values, stock_train.values):
                _, c = sess.run([optimizer, cost], feed_dict={x: [[X]], y: [[Y]]})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

        correct = tf.subtract(prediction, y)
        total = 0
        cor = 0
        for (X,Y) in zip(oil_test.values, stock_test.values):
            total += 1
            if abs(correct.eval({x:[[X]], y:[[Y]]})) < 5:
                cor += 1
        print('Accuracy:', cor/total)
train_neural_network(x)
