import tensorflow as tf
import numpy as np
### deal with MNIST image classification

# function of neuron definition
def neuron_layer(X, name, neuron_number, activation = None):
	with tf.name_scope(name):
		input_size = int(X.get_shape()[1])
		standard_deviation = 2 / np.sqrt(input_size)
		# randomly initialize Weight to avoid Symmetry problem of Gradient Descent
		init = tf.truncated_normal((input_size,neuron_number), stddev = standard_deviation)
		W = tf.Variable(init, name = 'Weights')
		b = tf.Variable(tf.zeros([neuron_number]), name = "Bias")
		# build a subgraph for computing
		z = tf.matmul(X,W) + b
		if activation == 'ReLu':
			z = tf.nn.relu(z)
		return z

####################################
###### Construction Phase ##########
####################################

# initialize hyperparameters 
input_size = 28 * 28
hidden1_neuron = 300
hidden2_neuron = 100
output_class = 10	
learning_rate = 0.01

# define variable
# not knowing the instance number, so set None
X = tf.placeholder(tf.float32, shape = (None, input_size), name= 'X')
y = tf.placeholder(tf.int32, shape = (None), name = 'y')

# construct Neural Network
with tf.name_scope('DNN'):
	hidden1 = neuron_layer(X,"hidden1", hidden1_neuron, 'ReLu')
	hidden2 = neuron_layer(hidden1,'hidden2', hidden2_neuron, 'ReLu')
	output = neuron_layer(hidden2, 'output', output_class, None)

# softmax activation function layer & cost function(cross-entropy) define
with tf.name_scope('Loss'):
	entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output, labels=y)
	loss = tf.reduce_mean(entropy, name='Loss')

# training with SGD
with tf.name_scope('Training'):
	optimizer = tf.train.GradientDescentOptimizer(learning_rate)
	train_ops = optimizer.minimize(loss)

# measure the performance of model
# compare top1 hypothesis with label
with tf.name_scope('Evaluating'):
	answer = tf.nn.in_top_k(output, y, 1) 
	performance = tf.reduce_mean(tf.cast(answer, tf.float32))

# init all variables
init = tf.global_variables_initializer()

# save the computing graph to disk
saver = tf.train.Saver()

####################################
######## Execution Phase ###########
####################################

epoch_number = 400
batch_size = 50

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/home/hadoop/code/tensorflow_tutorials/python/MNIST_data/")

# run the model
with tf.Session() as sess:
	init.run()
	for epoch in range(epoch_number):
		for iteration in range(mnist.train.num_examples // batch_size):
			X_batch, y_batch = mnist.train.next_batch(batch_size)
			sess.run(train_ops, feed_dict={X: X_batch, y: y_batch})
			acc_train = performance.eval(feed_dict={X: X_batch, y: y_batch})
			acc_test = performance.eval(feed_dict={X: mnist.test.images,y: mnist.test.labels})
		print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)
	save_path = saver.save(sess, "./my_model_final.ckpt")















