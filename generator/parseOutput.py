#!/usr/bin/env python

import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from google.protobuf import text_format
import caffe
from caffe.proto import caffe_pb2

"""
** Metric was monitored in the following order 
1. dram_utilization -> line num 7
2. achieved_occupancy -> line num 8
3. sysmem_utilization -> line num 9
4. executed_ipc -> line num 10
"""

def average(numbers):
	total = sum(numbers)
	total = float(total)
	return total / len(numbers) 
def parseNvprofLog(layer_type):
	kernel_list_dir="/home/sbchoi/git/gpu-cloud/generator/kernel_lists/" # fixed position
	nvprof_log_dir="/home/sbchoi/git/gpu-cloud/generator/nvprof_log/"
	#below are the vectors for each kernel 
	dram_util = []
	occupancy = []
	sysmem_util = []
	exec_ipc = []
	kernel_list = kernel_list_dir + layer_type
	with open(kernel_list) as fp:
		for kernel in fp:
			kernel = kernel[:-1] # erase the last character(carriage return)
			nvprof_log = nvprof_log_dir+ kernel + "-perf.txt"
			cnt = 1 # counter used for referencing certain lines
			with open(nvprof_log) as np:
				print "opened nvprof log" + nvprof_log
				for line in np:
					if cnt == 7: #dram utilization
						words = line.split(',')
						info = words[-2].split()[-1]
						info = info[:-2] # trim 
						info = info[1:] # trim again
						dram_util.append(int(info))	
					elif cnt == 8: #achieved occupancy
						words = line.split(',')
						occupancy.append(float(words[-2]))
					elif cnt == 9: #sysmem_utilization
						words = line.split(',')
						info = words[-2].split()[-1]
						info = info[:-2] # trim 
						info = info[1:] # trim again
						sysmem_util.append(int(info))
					elif cnt == 10: # executed_ipc
						words = line.split(',')
						exec_ipc.append(float(words[-2]))
						print words[-2]
					cnt = cnt + 1
	# get max of each kernel, max value is more useful for scheduling
	output_vector = []
	output_vector.append(max(dram_util))
	output_vector.append(max(occupancy))
	output_vector.append(max(sysmem_util))
	output_vector.append(max(exec_ipc))
	return output_vector

						
						
				

def parseCaffeLog(caffe_log, layer):
	fin = open(caffe_log,"r")
	layer_name = '['+layer+']' # adding big brackets to both side
	total_mem = 0
	with open(caffe_log) as fp:
	    for line in fp:
	    	words = line.split()
	    	for word in words:
	    		if word == layer_name:
	    			total_mem = total_mem + int(words[-1])
	fin.close()
	return total_mem

# leaving the argument function and main function for debugging
def parse_args():
	parser = ArgumentParser(description=__doc__,
			formatter_class=ArgumentDefaultsHelpFormatter)
	parser.add_argument('caffe',
			help='input caffe log file')
	parser.add_argument('layer_name',
			help='name of layer to parse')
	parser.add_argument('layer_type',
			help='type of layer to parse')
	parser.add_argument('output_csv_file',
			help='file to store output vector')
	args = parser.parse_args()
	return args
def main():
	args = parse_args()
	mem_allocated=parseCaffeLog(args.caffe, args.layer_name)
	
	result_vec = parseNvprofLog(args.layer_type)
	result_vec.append(mem_allocated)
		
	with open(args.output_csv_file,"a") as f:
        	for item in result_vec:
	           f.write(str(item)+',')
		f.write('\n')


if __name__ == '__main__':
	main()
