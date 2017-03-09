'''
A Multilayer Perceptron implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import tensorflow as tf
import time
import sys

LOG_FILE="monitor_log.txt"

tf.logging.set_verbosity(tf.logging.DEBUG)
# Parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100
display_step = 1

# Network Parameters
n_hidden_1 = int(sys.argv[1]) # 1st layer number of features
n_hidden_2 = int(sys.argv[2]) # 2nd layer number of features
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)

#by CSB, tell GPU to allocate as only as much memory required during runtime
config = tf.ConfigProto(log_device_placement=False)
config.gpu_options.allow_growth = True

#options for session.run()
run_options = tf.RunOptions(trace_level=2)




# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])
#cost = tf.placeholder(tf.float32)

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Create model
    # Hidden layer with RELU activation
mlayer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
layer_1 = tf.nn.relu(mlayer_1)
    # Hidden layer with RELU activation
mlayer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
layer_2 = tf.nn.relu(mlayer_2)
    # Output layer with linear activation
pred = tf.matmul(layer_2, weights['out']) + biases['out']


# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

# Set run options
run_options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE,output_partition_graphs=True)

# Launch the graph
with tf.Session(config=config) as sess:
	run_metadata = tf.RunMetadata()
	sess.run(init)
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,y: batch_y}, options=run_options,run_metadata=run_metadata)
#start = time.time()	
	sess.run(cost, feed_dict={x: batch_x, y: batch_y},options=run_options)
	store_dir= "/home/sbchoi/git/gpu-cloud/models/hidden_size_to_memory_test/"
	store_dir= store_dir + "out_" + sys.argv[1]
	store_dir= store_dir + ".txt"
	with open(store_dir,"w") as out:
		out.write(str(run_metadata))
#cost_t = time.time() - start
#print("duration: ",cost_t)
