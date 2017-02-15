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
import subprocess
import signal
import os 
import time
import shutil
tf.logging.set_verbosity(tf.logging.DEBUG)     
# Parameters
learning_rate = 0.001
training_epochs = 1
batch_size = 100
display_step = 1

# Network Parameters
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)

#by CSB, tell GPU to allocate as only as much memory required during runtime
config = tf.ConfigProto()
config.gpu_options.allow_growth = True



# init nvml 
#nvmlInit()
#GPU0 = nvmlDeviceGetHandleByIndex(0)
#log = open("/home/sbchoi/GPU_CLOUD/monitor/monitor_log.txt",'w')
# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])
cost = tf.placeholder(tf.float32)
# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}


# Create model
    # Hidden layer with RELU activation
layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['out'])
pred = tf.nn.relu(layer_1)

#Output layer with linear activation

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session(config=config) as sess:
	sess.run(init)
	#shutil.copyfile("monitor_log.txt","init_log.txt")
	batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)

	#    subprocess.Popen(["/home/sbchoi/GPU_CLOUD/monitor/cpp/nvml_mon", "/home/sbchoi/GPU_CLOUD/models/single_layer_models"])
 	  
	
        #c=sess.run(cost,feed_dict={x: batch_x, y: batch_y})
	#shutil.copyfile("monitor_log.txt", "FC_forward.txt")
	#sess.run(backprop, feed_dict={x: batch_x, y: batch_y, cost:c})
	#shutil.copyfile("monitor_log.txt", "FC_backprop.txt")
	#h = sess.partial_run_setup([pred,cost,dummy], [x,y])
	#shutil.copyfile("monitor_log.txt","partial_run_setup_log.txt")
	#cache = sess.partial_run(h,pred, feed_dict={x: batch_x})
	#shutil.copyfile("monitor_log.txt","1st_FC.txt")
	#time.sleep(3)
	#time.sleep(1)	
	#cache = sess.partial_run(h,cost,feed_dict={y: batch_y})
	#shutil.copyfile("monitor_log.txt","cost_log.txt")
	#print(cache)
	# time.sleep(3)
	#sess.partial_run(h, dummy)
	#shutil.copyfile("monitor_log.txt","backprop_log.txt")
	_, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
	#shutil.copyfile("monitor_log.txt")
	#sess.run(backprop, feed_dict={x: batch_x, y: batch_y})
	#print(final)
	 #   subprocess.Popen(["python", "nvml_monitor.py", "/home/sbchoi/GPU_CLOUD/models/single_layer_models"])
        

            #feed forward output only
	    #sess.run(cost, feed_dict={x: batch_x,
	                                  #y: batch_y})
 #	    subprocess.Popen(["python", "nvml_monitor.py", "/home/sbchoi/GPU_CLOUD/models/single_layer_models"])
        

	    #feedforward + backprop 
#	    _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
 	    #backprop
            #mon.recordUtil()    
	# Compute average loss
            #avg_cost += c / total_batch
        # Display logs per epoch step
        #if epoch % display_step == 0:
        #    print("Epoch:", '%04d' % (epoch+1), "cost=", \
        #        "{:.9f}".format(avg_cost))
print("Single FC Layer Finished!")

    # Test model
    #correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    #accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    #print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
#mon.closeNvml()
