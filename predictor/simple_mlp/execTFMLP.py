'''
A Multilayer Perceptron implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import tensorflow as tf
import numpy as np
import csv
import random
def getNextBatch(data,label,batch_size):
	i=0
	data_batch = []
	label_batch = []
	N = len(data)
	while i<batch_size:
		rand_index = random.randrange(0,N)
		data_batch.append(data[rand_index])
		label_batch.append(label[rand_index])
		i = i+1
	return data_batch, label_batch

def readInput(input_txt):
	vectors = []
	with open(input_txt,"rb") as csvfile:
		reader = csv.reader(csvfile,delimiter=',')
		for row in reader:
			del row[-1]
			narray = np.array(row,dtype=np.float)
			vectors.append(narray)
	return vectors

def parse_args():
	parser = ArgumentParser(description=__doc__,
			formatter_class=ArgumentDefaultsHelpFormatter)
	parser.add_argument('input_vec_txt',
			help='file which contains input vectors') # for data
	parser.add_argument('output_vec_txt',
			help='file which containes output vectors') # for label
	parser.add_argument('data_type',
			help='the type of data you want to train for Ex) Convolution, Network')
	args = parser.parse_args()
	return args


def main():
	args=parse_args()
	input_data = readInput(args.input_vec_txt) # read input data
	input_label = readInput(args.output_vec_txt)
	input_vec_size = len(input_data[0])
	output_vec_size = len(input_label[0])
# Parameters
	learning_rate = 0.001
	training_epochs = 1
	batch_size = 1
	display_step = 1

# Network Parameters
	n_input = input_vec_size # needs to be allocated dynamically in the future
	n_hidden = 20
	n_output = output_vec_size # nees to be allocated dynamically in the future

#by CSB, tell GPU to allocate as only as much memory required during runtime
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
# tf Graph input
	x = tf.placeholder("float", [None, input_vec_size])
	y = tf.placeholder("float", [None, output_vec_size])
	cost = tf.placeholder(tf.float32)
# Store layers weight & bias
	weights = {
	'h1': tf.Variable(tf.random_normal([n_input, n_hidden])),
	'h2': tf.Variable(tf.random_normal([n_hidden,n_hidden])),
	'h3': tf.Variable(tf.random_normal([n_hidden,n_output]))
	}
	biases = {
	'b1': tf.Variable(tf.random_normal([n_hidden])),
	'b2': tf.Variable(tf.random_normal([n_hidden])),
	'out': tf.Variable(tf.random_normal([n_output]))
	}
# Create model# Hidden layer with RELU activation
	layer1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
	layer1_act = tf.nn.relu(layer1)
	layer2 = tf.add(tf.matmul(layer1_act, weights['h2']), biases['b2'])
	layer2_act = tf.nn.relu(layer2)
	pred = tf.add(tf.matmul(layer2_act, weights['h3']), biases['out'])

# Define loss and optimizer

#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
	cost = loss = tf.nn.l2_loss(pred - labels)
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
# Initializing the variables
	init = tf.global_variables_initializer()

# Launch the graph
	with tf.Session(config=config) as sess:
		sess.run(init)

# Training cycle
		for epoch in range(training_epochs):
			avg_cost = 0.
# Loop over all batches
			
			batch_x, batch_y = getRandBatch(input_data, input_label,batch_size)
# Run optimization op (backprop) and cost op (to get loss value)
			_, c = sess.run([optimizer, cost], feed_dict={x: batch_x,y: batch_y})
# Compute average loss
			avg_cost += c / total_batch
# Display logs per epoch step
			if epoch % display_step == 0:
				print("Epoch:", '%04d' % (epoch+1), "cost=", \
					"{:.9f}".format(avg_cost))
		print("Optimization Finished!")
# Test model
#		correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
# Calculate accuracy
#		accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
#		print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))


# execute Caffe
#write results


if __name__ == '__main__':
	main()

