'''
Predictor MLP model for gpu-cloud
'''

from __future__ import print_function
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import tensorflow as tf
import numpy as np
import csv
import random
def normalizeInput(vector, norm_info): # normalizes a 1d input vector
	for i  in range(len(vector)):
		min_num = norm_info[i][0]
		max_num = norm_info[i][1]
		if min_num != max_num: # prevents divide-by-zero
			vector[i] = (vector[i] - min_num)/(max_num-min_num)

def unnormalizeData(vectors, norm_info):
	for i in range(len(vectors[0])):
		min_num = norm_info[i][0]
		max_num = norm_info[i][1]
		if min_num != max_num:
			for j in range(len(vectors)):
				vectors[j][i] = (max_num - min_num) * vectors[j][i] + min_num		

def readInput(input_txt):
	vectors = []
	with open(input_txt,"rb") as csvfile:
		reader = csv.reader(csvfile,delimiter=',')
		for row in reader:
			del row[-1]
			narray = np.array(row,dtype=np.float)
			vectors.append(narray)
	return vectors

def predictPerf(input_vector, label_vec_size,en_num,layer_type):
	
	
	
	tf.reset_default_graph()

	#root directory
	checkpoint_root_dir="/home/sbchoi/git/gpu-cloud/predictor"

	# dir where norm info is
	data_dir=checkpoint_root_dir+"/data"+"/trn3_"+layer_type
	print("The input vector for "+layer_type+": ")
	print(input_vector)
	data_vec_size=len(input_vector[0])
#Load Data Normalization Information
	val_data_norm_info=readInput(data_dir+"/data_norm") # used for normalizing input data
	val_label_norm_info=readInput(data_dir+"/label_norm") # used for normalizing output data

# Normalize input data
	normalizeInput(input_vector,val_data_norm_info)	
	# Network Parameters
	n_input = data_vec_size
	n_hidden = 2 * data_vec_size  # allocated heuristically
	n_output = label_vec_size 	
	dropout = 0.75
	batch_size = 64


#by CSB, tell GPU to allocate as only as much memory required during runtime
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True


# tf Graph input
	x = tf.placeholder("float", [None,data_vec_size])
	y = tf.placeholder("float", [None,label_vec_size])
	cost = tf.placeholder(tf.float32)
# Store layers weight & bias
	weights = {
	'h1': tf.Variable(tf.random_normal([n_input, n_hidden]),name="h1"),
	'h2': tf.Variable(tf.random_normal([n_hidden,n_hidden]),name="h2"),
#	'h3': tf.Variable(tf.random_normal([n_hidden,n_hidden]),name="h3"),
	'h4': tf.Variable(tf.random_normal([n_hidden,n_output]),name="h4")
	}
	biases = {
	'b1': tf.Variable(tf.random_normal([n_hidden]),name="b1"),
	'b2': tf.Variable(tf.random_normal([n_hidden]),name="b2"),
#	'b3': tf.Variable(tf.random_normal([n_hidden]),name="b3"),
	'out': tf.Variable(tf.random_normal([n_output]),name="out")
	}
	global_step = tf.Variable(0,trainable=False)
# Create model# Hidden layer with RELU activation
	layer1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
	layer1_act = tf.nn.relu(layer1)
	layer2 = tf.add(tf.matmul(layer1_act, weights['h2']), biases['b2'])
	layer2_act = tf.nn.relu(layer2)
#	layer3 = tf.add(tf.matmul(layer2_act,weights['h3']),biases['b3'] )
#	layer3_act = tf.nn.relu(layer3)
#	d_layer3_act = tf.nn.dropout(layer3_act, dropout)
	pred = last_layer = tf.add(tf.matmul(layer2_act, weights['h4']), biases['out'])
#	pred = tf.nn.relu(last_layer)
	pred = tf.abs(pred)

#saver for saving check points
	saver = tf.train.Saver()

# Define loss and optimizer

#	cost = loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
	cost = loss = tf.reduce_mean(tf.nn.l2_loss(pred-y))
# Initializing the variables
	init = tf.global_variables_initializer()
# Launch the graph
	with tf.Session(config=config) as sess:	
		# creating vectors for storing prediction results
		infer_results = []
		for en_id in range(int(en_num)):

			checkpoint_dir=checkpoint_root_dir+"/"+layer_type+"-checkpoint-"+str(en_id) 
#			print (checkpoint_dir)
			saver.restore(sess,checkpoint_dir+"/model.ckpt")		
			output_vec = sess.run([pred], feed_dict={x: input_vector}) 
			unnormalizeData(output_vec,val_label_norm_info)
			infer_results.append(output_vec)

		# obtain new outputs from five models
		new_outputs = []
		for i in range(len(infer_results[0][0][0])):
			each_model_outputs = []
			for en_id in range(int(en_num)):
				each_model_outputs.append(infer_results[en_id][0][0][i])
			# sort and delete first and last element from list
#			print (each_model_outputs)
			each_model_outputs.sort()
			del each_model_outputs[0]
			del each_model_outputs[-1]
			# get the average of the remaining outputs
			summed_output = sum(each_model_outputs)
			new_output = summed_output / len(each_model_outputs)
			new_outputs.append(new_output)		
		print ("Output vector of "+layer_type+" : ")
		print (new_outputs)
	return new_outputs
#	np.savetxt('new_output_vec.txt',new_outputs,delimiter=',')


