import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.contrib.layers import fully_connected

t_min, t_max = 0, 30
step = 0.1

# generate simulated time series data
def time_series(t):
    return t * np.sin(t) / 3 + 2 * np.sin(t*5)

x_series = np.linspace(t_min, t_max, (t_max - t_min)//step)
n_steps = 20
t_instance = np.linspace(10.,10.+(n_steps+1)*step, n_steps+1)
'''
# plot figure
plt.figure(figsize=(11,4))
plt.subplot(121)
plt.title("A time series (generated)", fontsize=14)
plt.plot(x_series, time_series(x_series), label=r"$t . \sin(t) / 3 + 2 . \sin(5t)$")
plt.plot(t_instance[:-1], time_series(t_instance[:-1]), "b-", linewidth=3, label="A training instance")
plt.legend(loc="lower left", fontsize=14)
plt.axis([0, 30, -17, 13])
plt.xlabel("Time")
plt.ylabel("Value")

plt.subplot(122)
plt.title("A training instance", fontsize=14)
# training instance
plt.plot(t_instance[:-1], time_series(t_instance[:-1]), "bo", markersize=10, label="instance")
# target instance, 1 step into the future
plt.plot(t_instance[1:], time_series(t_instance[1:]), "w*", markersize=10, label="target")
plt.legend(loc="upper left")
plt.xlabel("Time")

plt.show()
'''

def next_batch(batch_size, n_steps):
    # return array[batch_size * 1]
    t0 = np.random.rand(batch_size, 1) * (t_max - t_min - n_steps * step)
    # print(t0)
    # return array[1 * (n_steps+1)]
    t1 = np.arange(0., n_steps+1)*step
    # print(t1)

    # return array[batch_size * (n_steps+1)]
    Xs = t0 + t1
 #   print("Xs", Xs.shape)
  #  print(Xs)
    # return array[batch_size * (n_steps+1)]
    ys = time_series(Xs)
   # print("ys", ys.shape)
   # print(ys)
    # return training batch & target batch
    return ys[:,:-1].reshape(-1, n_steps, 1), ys[:,1:].reshape(-1, n_steps, 1)
train, target = next_batch(4, 2) 
print("=== Next batch of 4,2")
print(train.shape, train)
print(target.shape, target)
print(np.c_[train[0], target[0]])

#########################
### contruction phase ###
#########################
tf.reset_default_graph()
n_neurons = 100
n_steps = 20
n_input = 1
n_output = 1
learning_rate = 0.001

X = tf.placeholder(tf.float32, [None, n_steps, n_input])
y = tf.placeholder(tf.float32, [None, n_steps, n_output])

# define network, 1 layer now
#cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu)
#cell =tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, actication=tf.rnn.relu)
#wrapped_cell = tf.contrib.rnn.OutputProjectionWrapper(cell, ouput_size=n_output)
#wrapped_cell = tf.contrib.rnn.OutputProjectionWrapper(
#    tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu),
#        output_size=n_output)
#outputs, states = tf.nn.dynamic_rnn(wrapped_cell, X, dtype=tf.float32)
# without using OutputProjectionWrapper
basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu)
rnn_outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)
stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, n_neurons])
stacked_outputs = fully_connected(stacked_rnn_outputs, n_output, activation_fn=None)
outputs = tf.reshape(stacked_outputs, [-1, n_steps, n_output])
# define loss optimizer
loss = tf.reduce_sum(tf.square(outputs-y))
optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(loss)

##############################
##### Training Phase #########
##############################
n_iterations = 1000
batch_size = 50

init = tf.global_variables_initializer()
with tf.Session() as sess:
    init.run()
    for iter in range(n_iterations):
        X_batch, y_batch = next_batch(batch_size, n_steps)
        sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        if iter % 100 == 0:
            mse = loss.eval(feed_dict={X: X_batch, y: y_batch})
            print(iter, "MSE: ", mse)
    # evaluate
    X_new = time_series(np.array(t_instance[:-1].reshape(-1, n_steps, n_input)))
    y_pred = sess.run(outputs, feed_dict={X: X_new})
    print(y_pred)
    y_target = time_series(np.array(t_instance[1:].reshape(-1, n_steps, n_input)))
    print(y_target)

plt.title("Testing the model", fontsize=14)
plt.plot(t_instance[:-1], time_series(t_instance[:-1]), "bo", markersize=10, label="instance")
plt.plot(t_instance[1:], time_series(t_instance[1:]), "g*", markersize=10, label="target")
plt.plot(t_instance[1:], y_pred[0,:,0], "r.", markersize=10, label="prediction")
plt.legend(loc="upper left")
plt.xlabel("Time")
plt.show()
