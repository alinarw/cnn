import pickle
import random
import time
import itertools
import tensorflow as tf
import numpy as np

tf.set_random_seed(42)
np.random.seed(42)
random.seed(42)

tf.logging.set_verbosity(tf.logging.FATAL)

tf.reset_default_graph()

def data_load():
    with open(r'mnist_short.pickle', 'rb') as fb:
        data = pickle.load(fb)

    with open(r'mnist_short_test.pickle', 'rb') as fb:
        data_test = pickle.load(fb)

    x_train = data['x'].reshape((-1, 28, 28, 1))
    x_test = data_test['x'].reshape((-1, 28, 28, 1))

    y_train = data['y'].flatten()
    y_test = data_test['y'].flatten()

    # y_train = tf.one_hot(y_train, depth=10, on_value=1, off_value=0)

    training_label_zeros = np.zeros((x_train.shape[0], len(np.unique(y_train))))
    training_label_zeros[np.arange(x_train.shape[0]), y_train.flatten().astype(int)] = 1
    y_train = training_label_zeros

    test_label_zeros = np.zeros((x_test.shape[0], len(np.unique(y_test))))
    test_label_zeros[np.arange(x_test.shape[0]), y_test.astype(int)] = 1
    y_test = test_label_zeros
    return x_train, y_train, x_test, y_test


def new_conv_layer(input, num_input_channels, filter_size, num_filters, name):

    with tf.variable_scope(name):

        shape = [filter_size, filter_size, num_input_channels, num_filters]

        weights = tf.Variable(tf.truncated_normal(shape, stddev=0.05))

        biases = tf.Variable(tf.constant(0.05, shape=[num_filters]))

        layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding='VALID')

        layer += biases

        tf.summary.histogram("weights", weights)
        tf.summary.histogram("layer", layer)

        return layer, weights


def new_pool_layer(input, name):
    with tf.variable_scope(name):

        layer = tf.nn.max_pool(value=input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        return layer


def new_relu_layer(input, name):
    with tf.variable_scope(name):

        layer = tf.nn.relu(input)

        tf.summary.histogram("activations", layer)

        return layer


def new_fc_layer(input, num_inputs, num_outputs, name):
    with tf.variable_scope(name):

        weights = tf.Variable(tf.truncated_normal([num_inputs, num_outputs], stddev=0.05))
        biases = tf.Variable(tf.constant(0.05, shape=[num_outputs]))

        layer = tf.matmul(input, weights) + biases

        tf.summary.histogram("weights", weights)
        tf.summary.histogram("layer", layer)

        return layer


if __name__ == '__main__':

    x_train, y_train, x_test, y_test = data_load()

    x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1), name='X')
    y_true = tf.placeholder(tf.float32, shape=(None, 10), name='y_true')
    y_true_cls = tf.argmax(y_true, dimension=1)

    # -------------------------------- ConvNet Implementation ---------------------------------------------------------
    # Convolutional Layer 1
    conv1, weights_conv1 = new_conv_layer(input=x, num_input_channels=1, filter_size=5, num_filters=16,
                                          name="conv1")

    # Pooling Layer 1
    pool1 = new_pool_layer(conv1, name="pool1")

    # RelU layer 1
    relu1 = new_relu_layer(pool1, name="relu1")

    # Convolutional Layer 2
    conv2, weights_conv2 = new_conv_layer(input=relu1, num_input_channels=16, filter_size=5, num_filters=32,
                                          name="conv2")

    # Pooling Layer 2
    pool2 = new_pool_layer(conv2, name="pool2")

    # RelU layer 2
    relu2 = new_relu_layer(pool2, name="relu2")

    # Flatten Layer
    num_features = relu2.get_shape()[1:4].num_elements()
    flat = tf.reshape(relu2, [-1, num_features])

    # Fully-Connected Layer 1
    fc1 = new_fc_layer(flat, num_inputs=num_features, num_outputs=128, name="fc1")
    keep_prob = tf.placeholder(tf.float32)
    dropout_reg = tf.nn.dropout(fc1, keep_prob=keep_prob, name='dropout', seed=42)
    
    # RelU layer 3
    relu3 = new_relu_layer(dropout_reg, name="relu3")

    # Fully-Connected Layer 2
    fc2 = new_fc_layer(input=relu3, num_inputs=128, num_outputs=10, name="fc2")

    with tf.variable_scope("Softmax"):
        y_pred = tf.nn.softmax(fc2)
        y_pred_cls = tf.argmax(y_pred, dimension=1)

    # ---------------------------------------- Loss -------------------------------------------------------------------
    with tf.name_scope("cross_ent"):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=fc2, labels=y_true)
        cost = tf.reduce_mean(cross_entropy)

    with tf.name_scope("optimizer"):
        optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)
# GradientDescentOptimizer
    with tf.name_scope("accuracy"):
        correct_prediction = tf.equal(y_pred_cls, y_true_cls)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # ----------------------------------- TensorBoard Viz --------------------------------------------------------------
    writer = tf.summary.FileWriter(r'.\tmp\Training')
    writer1 = tf.summary.FileWriter(r'.\tmp\Testing')

    tf.summary.scalar('loss', cost)
    tf.summary.scalar('accuracy', accuracy)

    merged_summary = tf.summary.merge_all()

    # -------------------------------------- Training ------------------------------------------------------------------

    n = x_train.shape[0]
    num_epochs = 30
    mb_size = 100
    
    #rotate_image = tf.contrib.image.rotate(images=x_train, angles=2*np.pi/180, interpolation='NEAREST')

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    #rotated_image = sess.run(rotate_image)
    #print(rotate_image)

    x_train_copy = np.copy(x_train)
    y_train_copy = np.copy(y_train)
    
    print('X shape origin: ', x_train.shape)
    print('Y shape origin: ', y_train.shape)
    
    #Rotating by 1-3 degres in clockwise and contrclockwise directions
    for i in range(1, 4):
        x2_contr = sess.run(tf.contrib.image.rotate(x_train_copy, i*np.pi/180))
        x2_clock = sess.run(tf.contrib.image.rotate(x_train_copy, -i*np.pi/180))
        x_train = np.vstack((x_train, x2_contr, x2_clock))
        y_train = np.vstack((y_train, y_train_copy, y_train_copy))
    
    print('X shape rotation: ', x_train.shape)
    print('Y shape rotation: ', y_train.shape)
   
    #Shifting by 3 pixels in 8 directions
    for a, b in itertools.product((-3, 0 , 3), repeat=2):
        if not(a == 0 and b == 0):    
            x2_translate = sess.run(tf.contrib.image.translate(x_train_copy, [a,b]))
            x_train = np.vstack((x_train, x2_translate))
            y_train = np.vstack((y_train, y_train_copy))
    
    data = np.hstack((x_train.reshape((-1, 784)), y_train))
    np.random.shuffle(data)
    x_train, y_train = data[:, :-10].reshape((-1, 28, 28, 1)), data[:, -10:]
    
    writer.add_graph(sess.graph)
    
    start_time1 = time.time()
    
    accuracy_train = []
    accuracy_test = []
    
    for epoch in range(num_epochs):

        start_time2 = time.time()
        # train_accuracy = 0

        for i in range(int(n / mb_size)):

            idx = i * mb_size

            sess.run(optimizer, feed_dict={
                x: x_train[idx:idx+mb_size, :],
                y_true: y_train[idx:idx+mb_size, :],
                keep_prob: 0.6
            })

        train_accuracy = sess.run(accuracy, feed_dict={
                x: x_train,
                y_true: y_train,
                keep_prob: 1
            })

        summ = sess.run(merged_summary, feed_dict={
                x: x_test,
                y_true: y_test,
                keep_prob: 1
            })

        writer.add_summary(summ, epoch)

        summ, test_accuracy = sess.run([merged_summary, accuracy], feed_dict={
                x: x_test,
                y_true: y_test,
                keep_prob: 1
            })
        writer1.add_summary(summ, epoch)

        end_time2 = time.time()
        
        accuracy_train.append(train_accuracy)
        accuracy_test.append(test_accuracy)
        
        print("Epoch " + str(epoch + 1) + " completed : Time usage " + str(int(end_time2 - start_time2)) + " seconds")
        print("\tAccuracy:")
        print(f"\t- Training Accuracy:\t{train_accuracy * 100:{4}.{3}}%")
        print(f"\t- Test Accuracy:\t{test_accuracy * 100:{4}.{3}}%")
    end_time1 = time.time()
    print("Time usage: " + str(int(end_time1 - start_time1)) + " seconds")
    print("Train accuracy: ", accuracy_train)
    print("Test accuracy: ", accuracy_test)
