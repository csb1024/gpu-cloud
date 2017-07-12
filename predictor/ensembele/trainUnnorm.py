'''
Predictor MLP model for gpu-cloud
'''

from __future__ import print_function
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import tensorflow as tf
import numpy as np
import csv
import random
def storeNormData(norm_info_file, norm_vectors):
	with open(norm_info_file,"w") as fp:
		for vector in norm_vectors:
			for item in vector:
				fp.write(str(item)+',')
			fp.write('\n')

def unnormalizeData(vectors, norm_info):
	for i in range(len(vectors[0])):
		min_num = norm_info[i][0]
		max_num = norm_info[i][1]
		if min_num != max_num:
			for j in range(len(vectors)):
				vectors[j][i] = (max_num - min_num) * vectors[j][i] + min_num
		
def getRandBatch(data,label,batch_size):
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

def writeToFile(output_file, vector):
	data_len=len(vector)
	fp = open(output_file,"w")
	i = 0
	for i in range(data_len):
		fp.write("%s\n" % vector[i])

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
	parser.add_argument('trn_dir',
			help='dir which contains training data and labels') # for data
	parser.add_argument('val_dir',
			help='dir which contains validation data and labels') # for validation data
	parser.add_argument('checkpoint_dir',
			help='directory where checkpoint files will be stored/loaded, pass "new" for training new parameters')
	args = parser.parse_args()
	return args
def main():
	args=parse_args()
	trn_data = readInput(args.trn_dir+"/input_vec.txt") # read input data
	trn_label = readInput(args.trn_dir+"/output_vec.txt")
	val_data = readInput(args.val_dir+"/input_vec.txt")
	val_label = readInput(args.val_dir+"/output_vec.txt")

	data_vec_size = len(trn_data[0])
#	print ("data_vec_size: "+data_vec_size)
	label_vec_size = len(trn_label[0])
#	print ("label_vec_size : "+label_vec_size)
#print("data_vec_size: ", '%4d'% data_vec_size, "label_vec_size: ",'%4d'% label_vec_size)
# Data Normalization
	# vectors needed to be stored for restoring information

# Hyper-Parameters
	if args.checkpoint_dir == "new":
		starter_learning_rate=0.01 # first training
	else:
		starter_learning_rate=0.001 # not first
	max_training_epochs = 100000
	batch_size = 64
	test_step = 1000 # steps for displaying validation error
	dropout = 0.75
#	beta = 0.01 # for L2 regularization
# Network Parameters
	n_input = data_vec_size # needs to be allocated dynamically in the future
	n_hidden =  2* data_vec_size  # allocated heuristically
	n_output = label_vec_size # nees to be allocated dynamically in the future

#by CSB, tell GPU to allocate as only as much memory required during runtime
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True


# tf Graph input
	x = tf.placeholder("float", [None, data_vec_size])
	y = tf.placeholder("float", [None, label_vec_size])
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
#	layer1_act = tf.nn.relu(layer1)
	layer2 = tf.add(tf.matmul(layer1, weights['h2']), biases['b2'])
	layer2_act = tf.nn.relu(layer2)
#	layer3 = tf.add(tf.matmul(layer2_act,weights['h3']),biases['b3'] )
#	layer3_act = tf.nn.relu(layer3)
	d_layer3_act = tf.nn.dropout(layer2_act, dropout)
	pred = last_layer = tf.add(tf.matmul(d_layer3_act, weights['h4']), biases['out'])
#	pred = tf.nn.relu(last_layer) 
	pred = tf.abs(pred)
#saver for saving check points
	saver = tf.train.Saver()

# Define loss and optimizer

#	cost = loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
	cost = loss = tf.reduce_mean(tf.nn.l2_loss(pred-y))
#	cost = loss = tf.metrics.mean_squared_error(y,pred)
#	regularizer = tf.nn.l2_loss(weights['h1']) + tf.nn.l2_loss(weights['h2']) +tf.nn.l2_loss(weights['h4'])
#	cost = loss = tf.reduce_mean(loss+ beta*regularizer)
#	cost = loss = tf.nn.l2_loss(pred - y)

#learning_rate = starter_learning_rate
#	learning_rate = tf.train.exponential_decay(starter_learning_rate,global_step,max_training_epochs/20,0.9,staircase=True)
	learning_rate = starter_learning_rate
#	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost,global_step=global_step)
#	optimizer = tf.train.MomentumOptimizer(learning_rate,0.9).minimize(cost,global_step=global_step)
	optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(cost,global_step=global_step)



# Initializing the variables
	init = tf.global_variables_initializer()

# Launch the graph
	with tf.Session(config=config) as sess:
		if args.checkpoint_dir == "new":
			sess.run(init)

		else:
			saver.restore(sess, args.checkpoint_dir+"/model.ckpt")

	        end = False
		prev_loss = float(1024*1024*1024) # a very large number
# Training cycle
		for epoch in range(max_training_epochs):
			avg_cost = 0.
# Loop over all batches
			
			batch_x, batch_y = getRandBatch(trn_data, trn_label,batch_size)
# Run optimization op (backprop) and cost op (to get loss value)
			_, c = sess.run([optimizer, cost], feed_dict={x: batch_x,y: batch_y})
# Compute average loss
			avg_cost = c / batch_size
# Display logs per epoch step
			if epoch % test_step == 0:
				print("Epoch:", '%04d' % (epoch+1))

#				batch_x, batch_y = getRandBatch(val_data,val_label,len(val_label))
#				if avg_cost > prev_loss:
#					print("lowest loss ",'%4d' % prev_loss)
#					break;
#				else:
#					prev_loss = avg_cost

				batch_x = val_data
				batch_y = val_label
				error = sess.run([cost],feed_dict={x: batch_x, y: batch_y})
				error[0] = error[0] / len(val_label)			
				print("Validation Error=",'{:.9f}'.format( error[0]))
				if avg_cost < 0.001:
					end = True
		print("Optimization finished at",'%4d'% epoch)
		if args.checkpoint_dir == "new":
			args.checkpoint_dir = "/home/sbchoi/git/gpu-cloud/predictor"
		save_path = saver.save(sess,args.checkpoint_dir+"/model.ckpt")
		print("check point file save in ",save_path)
		batch_x, batch_y = getRandBatch(val_data,val_label,batch_size)
		input_vec, label_vec, output_vec, cost = sess.run([x,y,pred,cost],feed_dict={x: batch_x, y: batch_y})
		fin_error = cost / batch_size
		print("Validation Error=",'{:.9f}'.format( fin_error))

		
#		unnormalizeData(input_vec,val_data_norm_info)

#		unnormalizeData(label_vec,val_label_norm_info)
#		unnormalizeData(output_vec,val_label_norm_info)

		np.savetxt('input_vec.txt',input_vec,delimiter=',')
		np.savetxt('label_vec.txt',label_vec,delimiter=',')
		np.savetxt('output_vec.txt',output_vec,delimiter=',')

		# Test Accuracy against validation data
	
if __name__ == '__main__':
	main()
