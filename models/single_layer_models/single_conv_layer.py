'''
A Convolutional Network implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import tensorflow as tf

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import subprocess
import time
import shutil
# Parameters
learning_rate = 0.001
training_iters = 200000
batch_size = 128
display_step = 10

# Network Parameters
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)
dropout = 0.75 # Dropout, probability to keep units

#by CSB, tell GPU to allocate as only as much memory required during runtime
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)
#subprocess.Popen(["/home/sbchoi/GPU_CLOUD/monitor/cpp/nvml_mon","/home/sbchoi/GPU_CLOUD/models/single_layer_models"])


# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')

# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    # 14*14*32 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([14*14*32, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}
forward_result = tf.placeholder(tf.float32)



# Create model
# Reshape input picture
tx = tf.reshape(x, shape=[-1, 28, 28, 1])

# Convolution Layer
conv1 = conv2d(tx, weights['wc1'], biases['bc1'])
# Max Pooling (down-sampling)
mconv1 = maxpool2d(conv1, k=2)

# Fully connected layer
# Reshape conv2 output to fit fully connected layer input
afc1 = tf.reshape(mconv1, [-1, weights['out'].get_shape().as_list()[0]])

# Output, class prediction
pred = tf.add(tf.matmul(afc1, weights['out']), biases['out'])

# Define loss and optimizer
forward_result = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(forward_result)
# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()
run_options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE,output_partition_graphs=True)
# Launch the graph
with tf.Session(config=config) as sess:
    run_metadata = tf.RunMetadata()
    sess.run(init)
#shutil.copyfile("monitor_log.txt", "conv_init_log.txt")
    # Keep training until reach max iterations
    batch_x, batch_y = mnist.train.next_batch(batch_size)
    # Run FeedForwward
    _, c = sess.run([optimizer, forward_result], feed_dict={x: batch_x, y: batch_y}, options=run_options,run_metadata=run_metadata)
    with open("/home/sbchoi/conv_out.txt","w") as out:
    	out.write(str(run_metadata))
    #shutil.copyfile("monitor_log.txt", "conv_forward.txt")
    #sess.run([forward_result,optimizer], feed_dict={ x: batch_x, y: batch_y})
    #shutil.copyfile("monitor_log.txt", "conv_backprop.txt")
    
    
#h = sess.partial_run_setup([conv1,mconv1,pred,forward_result], [x,y])
#shutil.copyfile("monitor_log.txt","conv_partial_run_setup.txt")
#cache = sess.partial_run(h,conv1,feed_dict={x: batch_x})
#shutil.copyfile("monitor_log.txt","conv_layer.txt")
    
    
#cache = sess.partial_run(h,mconv1)
#shutil.copyfile("monitor_log.txt","max_pool.txt")
    
#cache = sess.partial_run(h,pred)
   
#cache = sess.partial_run(h, forward_result, feed_dict={y: batch_y})
#shutil.copyfile("monitor_log.txt","conv_FC.txt")
print("Optimization Finished!")

    # Calculate accuracy for 256 mnist test images
    #print("Testing Accuracy:", \
    #    sess.run(accuracy, feed_dict={x: mnist.test.images[:256],
    #                                  y: mnist.test.labels[:256],
    #                                 keep_prob: 1.}))
