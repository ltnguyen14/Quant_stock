import matplotlib.pyplot as plt
import matplotlib
import tensorflow as tf
from scripts import data_process as dp


n_classes = 2
chunk_size = 1
rnn_size = 512
n_chunks = 2
total_chunk_size = chunk_size*n_chunks


def graph(models):
    for model in models:
        print("Loading pre-trained model...")
        sess = tf.Session()
        saver = tf.train.import_meta_graph("data/model/"+str(model)+'/'+str(model)+'.ckpt.meta')
        saver.restore(sess, tf.train.latest_checkpoint('data/model/'+str(model)))
        print("Model loaded...")

        graph = tf.get_default_graph()
        if model == 'feedforward':
            x = graph.get_tensor_by_name('input:0')
            prediction = graph.get_tensor_by_name('output:0')
        elif model == 'recurrent':
            x = graph.get_tensor_by_name('input_recurrent:0')
            prediction = graph.get_tensor_by_name('output_recurrent:0')
        _, _, _, _, oil_price, stock_price = dp.create_data()

        predictions = []
        if model == 'feedforward':
            date_labels = oil_price.index
            date_labels = matplotlib.dates.date2num(date_labels.to_pydatetime())
            for i in oil_price:
                predictions.append(sess.run(prediction, feed_dict={x: [[i]]})[0][0])
        elif model == 'recurrent':
            predictions = []
            for index in range(int(len(oil_price.values) / total_chunk_size)):
                x_in = oil_price.values[index * total_chunk_size:index * total_chunk_size + total_chunk_size].reshape(
                    (1, n_chunks, chunk_size))
                predictions += sess.run(prediction, feed_dict={x: x_in})[0].reshape(total_chunk_size).tolist()

        plt.plot_date(date_labels, predictions, 'b-', label="Feedforward Predictions")
        plt.plot_date(date_labels, stock_price.values, 'r-', label='Stock Prices')
        plt.legend()
    plt.ylabel('Price')
    plt.xlabel('Year')
    plt.show()
