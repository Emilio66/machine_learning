import tensorflow as tf
import numpy as np

# manual RNN
def manual_rnn(X0_batch, X1_batch):
    tf.reset_default_graph()
    n_inputs = 3
    n_neurons = 5
    X0 = tf.placeholder(tf.float32, [None, n_inputs])
    X1 = tf.placeholder(tf.float32, [None, n_inputs])

    # randomly initialize with normal distribution
    Wx = tf.Variable(tf.random_normal(shape=[n_inputs, n_neurons],dtype=tf.float32))
    Wy = tf.Variable(tf.random_normal(shape=[n_neurons, n_neurons],dtype=tf.float32))
    b = tf.Variable(tf.zeros([1, n_neurons], dtype=tf.float32))
    
    # unroll calculating step
    Y0 = tf.tanh(tf.matmul(X0,Wx)+b)
    Y1 = tf.tanh(tf.matmul(Y0,Wy)+tf.matmul(X1,Wx)+b)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        init.run()
        Y0_val, Y1_val = sess.run([Y0, Y1], feed_dict={X0: X0_batch, X1: X1_batch})
    print(Y0_val,Y1_val)

X0_batch = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 0, 1]]) # t = 0
X1_batch = np.array([[9, 8, 7], [0, 0, 0], [6, 5, 4], [3, 2, 1]]) # t = 1
# manual_rnn(X0_batch, X1_batch)
# basic RNN use tensorflow cell
def basic_rnn(X0_batch, X1_batch):
    tf.reset_default_graph()
    n_inputs = 3
    n_neurons = 5
    X0 = tf.placeholder(tf.float32, [None, n_inputs])
    X1 = tf.placeholder(tf.float32, [None, n_inputs])
    
    #build rnn by using tf, automatcally handle Weight
    basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
    output, states = tf.contrib.rnn.static_rnn(basic_cell, [X0, X1], dtype=tf.float32)
    Y0, Y1= output

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        init.run()
        Y0_val, Y1_val = sess.run([Y0, Y1], feed_dict={X0: X0_batch, X1: X1_batch})
    print(Y0_val,Y1_val)

basic_rnn(X0_batch, X1_batch)

def basic_compact_rnn():
    tf.reset_default_graph()
    n_steps = 2
    n_inputs = 3
    n_neurons = 5

    X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
    # unpacking input X into permuted sequence in diffrent time steps
    X_seqs = tf.unstack(tf.transpose(X, perm=[1, 0, 2]))

    basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
    output_seqs, states = tf.contrib.rnn.static_rnn(basic_cell, X_seqs, dtype=tf.float32)
    outputs = tf.transpose(tf.stack(output_seqs), perm=[1, 0, 2])

    init = tf.global_variables_initializer()
    X_batch = np.array([
        # t = 0      t = 1 
        [[0, 1, 2], [9, 8, 7]], # instance 1
        [[3, 4, 5], [0, 0, 0]], # instance 2
        [[6, 7, 8], [6, 5, 4]], # instance 3
        [[9, 0, 1], [3, 2, 1]], # instance 4
    ])
    with tf.Session() as sess:
        init.run()
        outputs_val = outputs.eval(feed_dict={X: X_batch})
        print(np.transpose(outputs_val, axes=[1, 0, 2])[1])

basic_compact_rnn()

def basic_dynamic_rnn():
    tf.reset_default_graph()
    n_steps = 2
    n_inputs = 3
    n_neurons = 5

    X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
    basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
    # use off-the-shelf Neuron Network construction function, save the trouble of stack/unstack
    outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)

    init = tf.global_variables_initializer()
    X_batch = np.array([
        # t = 0      t = 1 
        [[0, 1, 2], [9, 8, 7]], # instance 1
        [[3, 4, 5], [8, 3, 2]], # instance 2
        [[6, 7, 8], [6, 5, 4]], # instance 3
        [[9, 0, 1], [3, 2, 1]], # instance 4
    ])
    with tf.Session() as sess:
        init.run()
        outputs_val = outputs.eval(feed_dict={X: X_batch})
        print(outputs_val)
        print(np.transpose(outputs_val, axes=[1, 0, 2])[1])

basic_dynamic_rnn()

def variable_input_rnn():
    tf.reset_default_graph()
    n_steps = 2
    n_inputs = 3
    n_neurons = 5

    seq_length = tf.placeholder(tf.int32,[None])
    X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
    basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
    # use off-the-shelf Neuron Network construction function, save the trouble of stack/unstack
    outputs, states = tf.nn.dynamic_rnn(basic_cell, X,sequence_length=seq_length, dtype=tf.float32)

    init = tf.global_variables_initializer()
    X_batch = np.array([
        # t = 0      t = 1 
        [[0, 1, 2], [9, 8, 7]], # instance 1
        [[3, 4, 5], [0, 0, 0]], # instance 2 (padded with zero vectors)
        [[6, 7, 8], [6, 5, 4]], # instance 3
        [[9, 0, 1], [0, 0, 0]], # instance 4 (padded with zero vectors)
    ])
    batch_seq_length = np.array([2,1,2,1])
    with tf.Session() as sess:
        init.run()
        outputs_val = outputs.eval(feed_dict={X: X_batch, seq_length:batch_seq_length})
        print(outputs_val)
        print(np.transpose(outputs_val, axes=[1, 0, 2])[1])

variable_input_rnn()

