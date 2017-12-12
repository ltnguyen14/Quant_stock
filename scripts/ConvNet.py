from scripts import data_process as dp
from scripts.constants import *
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pickle

n_classes = 20
batch_size = 128
input_size = (20, 160)
keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')


def maxpool2d(x):
    #                        size of window         movement of window
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


x = tf.placeholder('float', name='input_cnn')
y = tf.placeholder('float', name='train_output_cnn')


def cnn_model(data):
    weights = {'W_conv1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
               'W_conv2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
               'W_fc': tf.Variable(tf.random_normal([5 * 40 * 64, 1024])),
               'out': tf.Variable(tf.random_normal([1024, n_classes]))}

    biases = {'b_conv1': tf.Variable(tf.random_normal([32])),
              'b_conv2': tf.Variable(tf.random_normal([64])),
              'b_fc': tf.Variable(tf.random_normal([1024])),
              'out': tf.Variable(tf.random_normal([n_classes]))}

    data = tf.reshape(data, shape=[-1, 20, 160, 1])

    conv1 = tf.nn.relu(conv2d(data, weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool2d(conv1)

    conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool2d(conv2)

    fc = tf.reshape(conv2, [-1, 5 * 40 * 64])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])
    fc = tf.nn.dropout(fc, keep_rate)

    output = tf.add(tf.matmul(fc, weights['out']), biases['out'], name='cnn_output')

    return output


prediction = cnn_model(x)


def refine_input_with_lag(oil_train, stock_train, oil_test, stock_test):
    cost = tf.reduce_mean(tf.square(prediction-y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    #Adding lag
    all_lag_losses = []
    for i in range(lag_range):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            oil_lag, stock_lag = dp.add_lag(oil_train, stock_train, i)
            for epoch in range(lag_epoch_num):
                lag_loss = 0
                for index in range(int(len(oil_lag.values)/input_size[0])):
                    x_in = np.zeros((input_size[1], input_size[0], 1, 1))
                    for index_in, value in enumerate(oil_lag.values[index * input_size[0]:index * input_size[0] + input_size[0]]):
                        x_in[int(value), index_in, 0, 0] = 1
                    y_in = stock_lag.values[index * input_size[0]:index * input_size[0] + input_size[0]]
                    _, c = sess.run([optimizer, cost], feed_dict={x: x_in, y: y_in})
                    lag_loss += c
                print('Lag', i, 'epoch', epoch, 'loss:', lag_loss)
            all_lag_losses.append(lag_loss)
    lag = all_lag_losses.index(min(all_lag_losses))
    oil_train, stock_train = dp.add_lag(oil_train, stock_train, lag)
    oil_test, stock_test = dp.add_lag(oil_test, stock_test, lag)
    print("The best lag is:", lag)
    pickle.dump(lag, open("data/lag.p", "wb"))
    return oil_train, stock_train, oil_test, stock_test


def conv_neural_network(inputs):
    oil_train, stock_train, oil_test, stock_test, oil_price, stock_price = inputs
    cost = tf.reduce_mean(tf.square(prediction-y))
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    #oil_train, stock_train, oil_test, stock_test = inputs

    oil_train, stock_train, oil_test, stock_test = refine_input_with_lag(oil_train, stock_train, oil_test, stock_test)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #Running neural net
        for epoch in range(hm_epoch):
            epoch_loss = 0
            for index in range(int(len(oil_train.values) / input_size[0])):
                x_in = np.zeros((input_size[1], input_size[0], 1, 1))
                for index_in, value in enumerate(
                        oil_train.values[index * input_size[0]:index * input_size[0] + input_size[0]]):
                    x_in[int(value), index_in, 0, 0] = 1
                y_in = stock_train.values[index * input_size[0]:index * input_size[0] + input_size[0]]
                _, c = sess.run([optimizer, cost], feed_dict={x: x_in, y: y_in})
                epoch_loss += c
            print('Epoch', epoch, 'completed out of', hm_epoch, 'loss:', epoch_loss)
        correct = tf.reduce_mean(tf.square(tf.subtract(prediction, y)))
        total = 0
        cor = 0
        for index in range(int(len(oil_test.values) / input_size[0])):
            x_in = np.zeros((input_size[1], input_size[0], 1, 1))
            for index_in, value in enumerate(
                    oil_test.values[index * input_size[0]:index * input_size[0] + input_size[0]]):
                x_in[int(value), index_in, 0, 0] = 1
            y_in = stock_test.values[index * input_size[0]:index * input_size[0] + input_size[0]]
            total += input_size[0]
            if abs(correct.eval(feed_dict={x: x_in, y: y_in})) < 5:
                cor += input_size[0]

        saver = tf.train.Saver()
        print('Accuracy:', cor/total)
        save_path = saver.save(sess, "data/model/recurrent/recurrent.ckpt")
        print("Model saved in file: %s" % save_path)

        predictions = []
        for index in range(int(len(oil_price.values) / input_size[0])):
            x_in = np.zeros((input_size[1], input_size[0], 1, 1))
            for index_in, value in enumerate(
                    oil_price.values[index * input_size[0]:index * input_size[0] + input_size[0]]):
                x_in[int(value), index_in, 0, 0] = 1
            predictions += sess.run(prediction, feed_dict={x: x_in})[0].tolist()

        date_labels = oil_price.index
        date_labels = matplotlib.dates.date2num(date_labels.to_pydatetime())[:-14]

        plt.plot_date(date_labels, predictions, 'b-', label="RNN Predictions")
        plt.plot_date(date_labels, stock_price.values[:-14], 'r-', label='Stock Prices')
        plt.legend()
        plt.ylabel('Price')
        plt.xlabel('Year')
        plt.show()


if __name__ == "__main__":
    conv_neural_network(x)
