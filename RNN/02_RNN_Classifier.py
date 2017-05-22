import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import fully_connected

tf.reset_default_graph()
n_steps = 28
n_inputs = 28
n_neurons = 150
n_outputs = 10
learning_rate = 0.01
n_layers = 3 # add multiple layers to the network

# input 3 dimension: 1: instance, 2: row pixels, 3: column pixels
X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.int32, [None])

with tf.variable_scope("rnn", initializer=tf.contrib.layers.variance_scaling_initializer()):
    # a cell means 1 layer with n_neurons neurons, here, 150
    basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu)
    multi_cells = tf.contrib.rnn.MultiRNNCell([basic_cell]*n_layers,state_is_tuple=False)
    outputs, states = tf.nn.dynamic_rnn(multi_cells, X, dtype=tf.float32)

print("Shape of Outputs: ",outputs.shape, "shape of states: ", states.shape)
# in a fully-connected manner
# this can be done by setting state_is_tuple as False
#states = tf.concat(axis=1, values=states)
classifier= fully_connected(states, n_outputs, activation_fn=None)
print("Shape of classifier: ",classifier.shape, "Shape of y: ", y.shape)
# softmax + cross entropy calculation
xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=classifier)
# define loss function & optimize method
loss = tf.reduce_mean(xentropy)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
# measurement
training_op = optimizer.minimize(loss)
correct = tf.nn.in_top_k(classifier, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
init = tf.global_variables_initializer()

# execution
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/home/hadoop/code/tensorflow_tutorials/python/MNIST_data/")
epoch_number = 10
batch_size = 150
X_test= mnist.test.images.reshape((-1, n_steps, n_inputs))
y_test = mnist.test.labels
print("Y_TEST: TYPE: ", type(y_test)," shape: ", y_test.shape)
print("X_TEST: TYPE: ", type(X_test)," shape: ", X_test.shape)
# run the model
with tf.Session() as sess:
    init.run()
    # training the data over & over again for #epoch times
    for epoch in range(epoch_number):
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            # reshape the data to fit the format of input
            X_batch = X_batch.reshape((-1, n_steps, n_inputs))
            # print("Y_batch: TYPE: ", type(y_batch)," shape: ", y_batch.shape)
            
            acc_train,_=sess.run([accuracy,training_op], feed_dict={X: X_batch, y: y_batch})
            accuracy_test=sess.run(accuracy,feed_dict={X: X_test,y: y_test})
            acc_test = accuracy.eval(feed_dict={X: X_test,y: y_test})
        print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)
        print(accuracy_test)
    save_path = saver.save(sess,"./my_model_final.ckpt")

 
