"""
@author : karthikk
@date : 9/4/2018
"""

import tensorflow as tf
print("Using TensorFlow version: {}".format(tf.__version__))


def initialize_weights(shape):
    """

    :param shape:
    :return:
    """
    weights = tf.truncated_normal(stddev=0.1, shape=shape)
    return tf.Variable(weights)

def initialize_biases(shape):
    """
    
    :param shape: 
    :return: 
    """
    biases = tf.constant(0.1, shape=shape)
    return tf.Variable(biases)

# convolution operation
def conv2d(X, W):
    """

    :param X:
    :param W:
    :return:
    """
    return tf.nn.conv2d(input=X,
                        filter=W,
                        strides=[1, 1, 1, 1],
                        padding="SAME")

# pooling operation
def pool(X):
    """

    :param X:
    :return:
    """
    return tf.nn.max_pool(value=X,
                          ksize=[1,2,2,1],
                          strides=[1,2,2,1],
                          padding="SAME")

# convolution layer
def convolution(X, shape):
    """

    :param X:
    :param shape:
    :return:
    """
    W = initialize_weights(shape=shape)
    b = initialize_biases(shape=[shape[3]])
    # return activated conv2d
    return tf.nn.relu(conv2d(X=X, W=W) + b)

# fully connected layer
def dense(X, neurons):
    """

    :param X: input
    :param neurons: number of neurons used to construct in this layer
    :return:
    """
    W = initialize_weights(shape=[int(X.get_shape()[1]), neurons])
    b = initialize_biases(shape=[neurons])
    # return regular W.X + b
    return tf.matmul(X, W) + b

def cost(Y, y_hat):
    """
    computes the loss for given labels against predictions

    :param Y: actual outputs
    :param y_hat: predicted outputs
    :return:
    """
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y,
                                                                  logits=y_hat))

def main(mnist):
    """

    :param mnist:
    :return:
    """

    ## placeholders
    # shape = [batch, pixels]
    X = tf.placeholder(tf.float32, shape=[None, 784])

    # shape = [batch, number of output classes]
    Y = tf.placeholder(tf.float32, shape=[None, 10])

    ## network architecture
    # Width = 28, Height = 28, Depth = 1 (grayscale)
    _input_layer = tf.reshape(X, shape=[-1, 28, 28, 1])

    # hidden layer = 1, type=convolution
    # filter = 5x5, number of filters = 32, depth = 1
    _convolution_1 = convolution(X=_input_layer, shape=[6, 6, 1, 32])

    # hidden layer = 2, type=pooling
    _convolution_1_pooling = pool(X=_convolution_1)

    # hidden layer = 3, type=convolution
    # filter = 5x5, number of filters = 64, depth = 32
    _convolution_2 = convolution(X=_convolution_1_pooling, shape=[6, 6, 32, 64])

    # hidden layer = 4, type=pooling
    _convolution_2_pooling = pool(X=_convolution_2)

    # flattening layer
    _flattened = tf.reshape(_convolution_2_pooling, [-1, 7*7*64])

    # hidden layer = 5, fully connected layer
    _dense_1 = tf.nn.relu(dense(X=_flattened, neurons=1024))

    # adding dropout to the dense layer to avoid overfitting
    keep_prob = tf.placeholder(tf.float32)
    _dense_1_dropout = tf.nn.dropout(_dense_1, keep_prob=keep_prob)

    # output layer with 10 classes of outputs - one hot encoded
    y_hat = dense(_dense_1_dropout, neurons=10)

    # optimizer
    adam = tf.train.AdamOptimizer(learning_rate=0.01)

    train = adam.minimize(loss=cost(Y=Y, y_hat=y_hat))


    init = tf.global_variables_initializer()

    epochs = 1000

    with tf.Session() as sess:
        sess.run(init)
        for i in range(epochs):
            x_batch, y_batch = mnist.train.next_batch(50)
            sess.run(train, feed_dict={
                X: x_batch,
                Y: y_batch,
                keep_prob: 0.5
            })
            # for every 100 epochs
            if i%100 == 0:
                print("Step: {}".format(i))
                print("Accuracy: ")
                matches = tf.equal(tf.argmax(y_hat, 1), tf.argmax(Y, 1))
                accuracy = tf.reduce_mean(tf.cast(matches, tf.float32))
                print(sess.run(accuracy, feed_dict={
                    X: mnist.test.images,
                    Y: mnist.test.labels,
                    keep_prob: 1.0
                }))
                print("\n")

if __name__ == "__main__":

    # proxy crap
    import os
    os.environ["HTTP_PROXY"] = "http://proxy-web.micron.com:80"
    os.environ["HTTPS_PROXY"] = "https://proxy-web.micron.com:80"
    os.environ["NO_PROXY"] = ".micron.com,localhost"
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("./src/MNIST_data/",one_hot=True)
    main(mnist)
