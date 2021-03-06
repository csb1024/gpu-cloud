'''
Predictor MLP model for gpu-cloud
'''

from __future__ import print_function
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import tensorflow as tf
import numpy as np
import csv
import random
def loadNormData(norm_file):
	ret_vectors = []
	with open(norm_file,"r") as fp:
		vector_lines = fp.readlines()
		for line in vector_lines:
			norm_info = []
			
def unnormalizeData(vectors, norm_info):
	for i in range(len(vectors[0])):
		min_num = norm_info[i][0]
		max_num = norm_info[i][1]
		if min_num != max_num:
			for j in range(len(vectors)):
				vectors[j][i] = (max_num - min_num) * vectors[j][i] + min_num

def returnLog(vectors):  
	new_vectors = []
	for i in range(len(vectors)):
		vector = []
#		for j in range(len(vectors[0])):
#			vector.append(np.log)
#		print(vector)
		new_vectors.append(np.log(vectors[i]))
	return new_vectors
def returnExp(vectors):
	new_vectors = []
	for i in range(len(vectors)):
		#vector = []
		#for j in range(len(vectors[0])):
		#vector.append(np.exp(vectors[i][j]))
		new_vectors.append(np.exp(vectors[i]))
	return new_vectors

		
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
	parser.add_argument('val_data_txt',
			help='file which contains validation data vectors') # for validation data
	parser.add_argument('val_label_txt',
			help='file which contains validation label vectors') # for validation labels
	parser.add_argument('val_norm_dir',
			help='dir which contains validation dir norm')
	parser.add_argument('en_num',
			help='parameter which stores how many models to ensemble')
	parser.add_argument('layer_type',
			help='type of layer to infer, for creating checkpoint')
	parser.add_argument('hidden_num',
			help='number of neurons in hidden layer')
	parser.add_argument('gpu_dir',
			help='dir which stores GPU specific results')
	parser.add_argument('chk_dir',
			help='dir which stores checkpoints of model')
	args = parser.parse_args()
	return args
def main():
	args=parse_args()
	val_data = readInput(args.val_data_txt)
	val_label = readInput(args.val_label_txt)
	print ("Testing : "+args.layer_type)
	data_vec_size = len(val_data[0])
	print("data_vec_size : "+str(data_vec_size))
#	print (val_label[0])
	label_vec_size = len(val_label[0])
	print("label_vec_size : "+str(label_vec_size))
#print("data_vec_size: ", '%4d'% data_vec_size, "label_vec_size: ",'%4d'% label_vec_size)
# Data Normalization
	# vectors needed to be stored for restoring information
	val_data_norm_info = []
	val_label_norm_info = []

	for i  in range(data_vec_size):
		vector = []
		norm_info = []
		for j in range(len(val_data)):
			vector.append(val_data[j][i])
		min_num = min(vector)
		max_num = max(vector)
		norm_info.append(min_num)
		norm_info.append(max_num)
		val_data_norm_info.append(norm_info)
#		print('min_num : ', '%d' % min_num,
#				'and max_num: ','%d' % max_num )
		if min_num != max_num:
			for j in range(len(val_data)):
				val_data[j][i] = (val_data[j][i] - min_num)/(max_num-min_num)

	val_label= returnLog(val_label)
	print("label_vec_size after log : "+str(len(val_label[0])))
	# Network Parameters
	n_input = data_vec_size
	n_hidden = int(args.hidden_num)  # allocated heuristically
	n_output = label_vec_size 
	
	dropout = 0.75
	batch_size = 64
	val_data_norm_info=readInput(args.val_norm_dir+"/data_norm")
#	val_label_norm_info=readInput(args.val_norm_dir+"/label_norm")
	

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
	'h3': tf.Variable(tf.random_normal([n_hidden,n_hidden]),name="h3"),
	'h4': tf.Variable(tf.random_normal([n_hidden,n_output]),name="h4")
	}
	biases = {
	'b1': tf.Variable(tf.random_normal([n_hidden]),name="b1"),
	'b2': tf.Variable(tf.random_normal([n_hidden]),name="b2"),
	'b3': tf.Variable(tf.random_normal([n_hidden]),name="b3"),
	'out': tf.Variable(tf.random_normal([n_output]),name="out")
	}
	global_step = tf.Variable(0,trainable=False)
# Create model# Hidden layer with RELU activation
	layer1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
#	layer1_act = tf.nn.relu(layer1)
	layer2 = tf.add(tf.matmul(layer1, weights['h2']), biases['b2'])
	layer2_act = tf.nn.sigmoid(layer2)
	layer3 = tf.add(tf.matmul(layer2_act,weights['h3']),biases['b3'] )
	layer3_act = tf.nn.sigmoid(layer3)
	d_layer3_act = tf.nn.dropout(layer3_act, dropout)
	pred = last_layer = tf.add(tf.matmul(d_layer3_act, weights['h4']), biases['out'])
#	pred = tf.nn.relu(last_layer)
#	pred = tf.abs(pred)

#saver for saving check points
	saver = tf.train.Saver()

# Define loss and optimizer

#	cost = loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
	cost = loss = tf.reduce_mean(tf.abs(tf.subtract(y,pred)))

# Initializing the variables
	init = tf.global_variables_initializer()
# Launch the graph
	with tf.Session(config=config) as sess:
		
		# create validation set from valid data
		#batch_x, batch_y = getRandBatch(val_data, val_label,batch_size)		
		batch_x = val_data
		batch_y = val_label
		# creating vectors for storing prediction results
		infer_results = []
		net_costs = []
		
		for en_id in range(int(args.en_num)):
			# init vector for storing results
			pred_results = []
			# be sure to change directory
			checkpoint_dir=args.chk_dir+"/"+args.layer_type+"-checkpoint-forward-"+str(en_id)
			saver.restore(sess,checkpoint_dir+"/model.ckpt")		
			input_vec, label_vec, output_vec,val_err = sess.run([x,y,pred, cost], feed_dict={x: batch_x, y: batch_y}) 
			avg_err = val_err / len(val_label)
			net_costs.append(cost)
			print("Average Validation Error=",'{:.9f}'.format(avg_err),"of ensemble model #",'%d'%en_id)
			
			output_file="output_vec"+str(en_id)+".txt"
			output_vec = returnExp(output_vec)
#			unnormalizeData(output_vec,val_label_norm_info)
			infer_results.append(output_vec)
#			print(output_vec)
			np.savetxt(args.gpu_dir+"/"+output_file,output_vec,delimiter=',',fmt="%s")

			# next files will be overwritten... but not bothering to optimize further(unnecessary)
			unnormalizeData(input_vec,val_data_norm_info)
			label_vec = returnExp(label_vec)
#			unnormalizeData(label_vec,val_label_norm_info)
#			print(label_vec)
			np.savetxt(args.gpu_dir+"/"+'input_vec.txt',input_vec,delimiter=',')
#		print(len(label_vec))
#		print(len(label_vec[0]))
#		print(label_vec[0])
#		print(label_vec)
#		print(label_vec[0])

	
		with open(args.gpu_dir+"/"+'label_vec.txt', 'w') as f:
			#for row in label_vec : 
			np.savetxt(f,label_vec,delimiter=',')
			#np.savetxt('label_vec.txt',label_vec,delimiter=',',fmt="%s")
		

		# obtain new outputs from five models
		new_outputs = []
		cost = []
#		print('%d'%len(infer_results[0]))
		for i in range(len(infer_results[0])):
			new_output=[]
			for j in range(len(infer_results[0][0])):
				each_model_outputs=[]
				for en_id in range(int(args.en_num)):
					each_model_outputs.append(infer_results[en_id][i][j])  
				output = sum(each_model_outputs) / (int(args.en_num) +1) 
				new_output.append(output)
			new_outputs.append(new_output)
		#calculate L2 Loss with new outputs and print them to file
#		for i in range(len(output_vec)):
#			err = 0.
			
#			truth = label_vec[i][0] # for now getting loss for single element  
#			err = err + ((new_outputs[i] - truth) * (new_outputs[i] - truth)/2)
#		err = err / batch_size
#		err = float(err)
#		print("Average Ensembled Error=",'{:.9f}'.format(err))
#				output_vec[i][j]
#				new_outputs


#		
		
			
#		print("Validation Error of Ensemble Network=",'{:.9f}'.format(new_err))

		with open(args.gpu_dir+"/"+'new_output_vec.txt','w') as f : 
			#print (len(new_outputs))
			#for row in new_outputs:
			np.savetxt(f,new_outputs,delimiter=',')	
		#np.savetxt('new_output_vec.txt',new_outputs,delimiter=',',fmt="float32")
		

		
		
	
if __name__ == '__main__':
	main()
