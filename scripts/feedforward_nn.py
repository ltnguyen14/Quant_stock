from scripts import data_process as dp
import matplotlib
from scripts.constants import *
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 1
batch_size = 100

x = tf.placeholder('float')
y = tf.placeholder('float')


def neural_network_model(data):
    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([1, n_nodes_hl1])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                    'biases': tf.Variable(tf.random_normal([n_classes])),}

    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.add(tf.matmul(l3, output_layer['weights']),
            output_layer['biases'], name="output")

    return output


def refine_input_with_lag(oil_train, stock_train, oil_test, stock_test):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.square(tf.transpose(prediction)-y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    #Adding lag
    all_lag_losses = []
    for i in range(lag_range):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            oil_lag, stock_lag = dp.add_lag(oil_train, stock_train, i)
            for epoch in range(lag_epoch_num):
                lag_loss = 0
                for (X, Y) in zip(oil_lag.values, stock_lag.values):
                    _, c = sess.run([optimizer, cost], feed_dict={x: [[X]], y: [[Y]]})
                    lag_loss += c
                print('Lag', i, 'epoch', epoch, 'loss:', lag_loss)
            all_lag_losses.append(lag_loss)
    lag = all_lag_losses.index(min(all_lag_losses))
    oil_train, stock_train = dp.add_lag(oil_train, stock_train, lag)
    oil_test, stock_test = dp.add_lag(oil_test, stock_test, lag)
    print("The best lag is:", lag)
    pickle.dump(lag, open("data/save.p", "wb"))
    return oil_train, stock_train, oil_test, stock_test


def feedforward_neural_network(inputs):
    x = tf.placeholder('float', name='input')
    oil_train, stock_train, oil_test, stock_test, oil_price, stock_price = inputs
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.square(tf.transpose(prediction)-y))
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    #oil_train, stock_train, oil_test, stock_test = inputs

    oil_train, stock_train, oil_test, stock_test = refine_input_with_lag(oil_train, stock_train, oil_test, stock_test)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #Running neural net
        for epoch in range(hm_epoch):
            epoch_loss = 0
            for (X, Y) in zip(oil_train.values, stock_train.values):
                _, c = sess.run([optimizer, cost], feed_dict={x: [[X]], y: [[Y]]})
                epoch_loss += c
            print('Epoch', epoch, 'completed out of', hm_epoch, 'loss:', epoch_loss)
        correct = tf.subtract(prediction, y)
        total = 0
        cor = 0
        for (X,Y) in zip(oil_test.values, stock_test.values):
            total += 1
            if abs(correct.eval({x: [[X]], y: [[Y]]})) < 5:
                cor += 1
        print('Accuracy:', cor/total)
        save_path = saver.save(sess, "data/model/feedforward/feedforward.ckpt")
        print("Model saved in file: %s" % save_path)

        date_labels = oil_price.index
        date_labels = matplotlib.dates.date2num(date_labels.to_pydatetime())

        predictions = []
        for i in oil_price:
            predictions.append(sess.run(prediction, feed_dict={x: [[i]]})[0][0])
        plt.plot_date(date_labels, predictions, 'b-', label="Feedforward Predictions")
        plt.plot_date(date_labels, stock_price.values, 'r-', label='Stock Prices')
        plt.legend()
        plt.ylabel('Price')
        plt.xlabel('Year')
        plt.show()


if __name__ == "__main__":
    feedforward_neural_network(x)