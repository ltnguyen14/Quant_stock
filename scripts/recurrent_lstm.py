from scripts import data_process as dp
import tensorflow as tf
from tensorflow.contrib import rnn
import matplotlib.pyplot as plt
import pickle

n_classes = 10
rnn_size = 512
chunk_size = 10
n_chunks = 1

x = tf.placeholder('float', name='input_recurrent')
y = tf.placeholder('float', name='train_output_recurrent')


def rnn_model(data):
    layer = {'weights': tf.Variable(tf.random_normal([rnn_size, n_classes])),
             'biases': tf.Variable(tf.random_normal([n_classes]))}

    data = tf.transpose(data, [1, 0, 2])
    data = tf.reshape(data, [-1, chunk_size])
    data = tf.split(data, n_chunks, 0)

    lstm_cell = rnn.BasicLSTMCell(rnn_size, state_is_tuple=True)
    outputs, states = rnn.static_rnn(lstm_cell, data, dtype=tf.float32)

    output = tf.add(tf.matmul(outputs[-1], layer['weights']), layer['biases'], name='output_recurrent')

    return output


prediction = rnn_model(x)


def refine_input_with_lag(oil_train, stock_train, oil_test, stock_test):
    cost = tf.reduce_mean(tf.square(tf.transpose(prediction)-y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    #Adding lag
    all_lag_losses = []
    lag_range = 1
    lag_epoch_num = 1
    for i in range(lag_range):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            oil_lag, stock_lag = dp.add_lag(oil_train, stock_train, i)
            for epoch in range(lag_epoch_num):
                lag_loss = 0
                for index in range(int(len(oil_lag.values)/chunk_size)):
                    x_in = oil_lag.values[index*chunk_size:index*chunk_size + chunk_size].reshape((1, n_chunks, chunk_size))
                    y_in = stock_lag.values[index*chunk_size:index*chunk_size + chunk_size].reshape((1, n_chunks, chunk_size))
                    _, c = sess.run([optimizer, cost], feed_dict={x: x_in, y: y_in})
                    lag_loss += c
                print('Lag', i, 'epoch', epoch, 'loss:', lag_loss)
            all_lag_losses.append(lag_loss)
    lag = all_lag_losses.index(min(all_lag_losses))
    oil_train, stock_train = dp.add_lag(oil_train, stock_train, lag)
    oil_test, stock_test = dp.add_lag(oil_test, stock_test, lag)
    print("The best lag is:", lag)
    pickle.dump(lag, open("data/save.p", "wb"))
    return oil_train, stock_train, oil_test, stock_test


def recurrent_neural_network(inputs):
    oil_train, stock_train, oil_test, stock_test, oil_price, stock_price = inputs
    cost = tf.reduce_mean(tf.square(tf.transpose(prediction)-y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
    #oil_train, stock_train, oil_test, stock_test = inputs

    hm_epochs = 5
    oil_train, stock_train, oil_test, stock_test = refine_input_with_lag(oil_train, stock_train, oil_test, stock_test)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #Running neural net
        for epoch in range(hm_epochs):
            epoch_loss = 0
            for index in range(int(len(oil_train.values) / chunk_size)):
                x_in = oil_train.values[index*chunk_size:index*chunk_size + chunk_size].reshape((1, n_chunks, chunk_size))
                y_in = stock_train.values[index*chunk_size:index*chunk_size + chunk_size].reshape((1, n_chunks, chunk_size))
                _, c = sess.run([optimizer, cost], feed_dict={x: x_in, y: y_in})
                epoch_loss += c
            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)
        correct = tf.subtract(prediction, y)
        total = 0
        cor = 0
        for index in range(int(len(oil_test.values) / chunk_size)):
            x_in = oil_test.values[index*chunk_size:index*chunk_size + chunk_size].reshape((1, n_chunks, chunk_size))
            y_in = stock_test.values[index*chunk_size:index*chunk_size + chunk_size].reshape((1, n_chunks, chunk_size))
            total += chunk_size
            if abs(correct.eval({x: x_in, y: y_in})).all() < 5:
                cor += chunk_size
        print('Accuracy:', cor/total)
        save_path = saver.save(sess, "data/model/recurrent.ckpt")
        print("Model saved in file: %s" % save_path)

        predictions = []
        for index in range(int(len(oil_price.values) / chunk_size)):
            x_in = oil_price.values[index*chunk_size:index*chunk_size + chunk_size].reshape((1, n_chunks, chunk_size))
            print(sess.run(prediction, feed_dict={x: x_in})[0].reshape(chunk_size).tolist())
            predictions += sess.run(prediction, feed_dict={x: x_in})[0].reshape(chunk_size).tolist()
        plt.plot(oil_price.values, label='Oil Prices')
        plt.plot(stock_price.values, label='Stock Prices')
        plt.plot(predictions, label="Predictions")
        plt.legend()
        plt.ylabel('Price')
        plt.xlabel('Date')
        plt.show()


if __name__ == "__main__":
    recurrent_neural_network(x)
