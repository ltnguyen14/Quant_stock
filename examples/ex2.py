import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib import rnn

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

n_classes = 10
batch_size = 128
epochs_num = 10

chunk_size = 28
chunk_num = 28
rnn_size = 128

x = tf.placeholder('float', [None, chunk_num, chunk_size])
y = tf.placeholder('float')

learning_rate = 0.001

def recurrent_neural_network(x):
    layer = {'weights': tf.Variable(tf.random_normal([rnn_size, n_classes])),
             'biases': tf.Variable(tf.random_normal([n_classes]))}

    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, chunk_size])
    x = tf.split(x, chunk_num, 0)

    lstm_cell = rnn.BasicLSTMCell(rnn_size)
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    output = tf.matmul(outputs[-1], layer['weights']) + layer['biases']

    return output

def train_neural_network(x):
     prediction = recurrent_neural_network(x)
     cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction) )

     optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

     with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(epochs_num):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                epoch_x = epoch_x.reshape((batch_size, chunk_num, chunk_size ))
                _, c = sess.run([optimizer, cost], feed_dict = {x:epoch_x, y:epoch_y})
                epoch_loss += c
            print ('Epoch', epoch, 'completed out of', epochs_num, 'loss:', epoch_loss)

        correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x:mnist.test.images.reshape((-1,chunk_num,chunk_size)), y:mnist.test.labels }))

train_neural_network(x)
