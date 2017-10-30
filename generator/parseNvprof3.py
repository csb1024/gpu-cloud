#!/usr/bin/env python

import matplotlib
matplotlib.use('Agg')
import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from google.protobuf import text_format
import caffe
from caffe.proto import caffe_pb2
import os

"""
** Metric was monitored in the following order 
1. sm_activity
"""
def average(numbers):
	total = sum(numbers)
	total = float(total)
	return total / len(numbers) 
def parseNvprofLog(nvprof_log_dir):
	#below are the vectors for each kernel 
	sm_activity = []
	for f in os.listdir(nvprof_log_dir):	#for every file 
		if not os.path.isfile(nvprof_log_dir+"/"+f) : #skip directory
			continue
		with open(nvprof_log_dir+"/"+f) as fp:
			cnt = 1 
			print "Opened "+f
			for line in fp:
				if cnt >= 6: #sm_activity
					words = line.split(',')
					info = words[-1]
					info = info[:-2] #exclude '%\n'
	#				print info
					sm_activity.append(float(info))
				cnt = cnt + 1
			# get max of each kernel, max value is more useful for scheduling
	output_vector = []
	if not sm_activity: # the same condition also holds for additional vectors 
		output_vector.append(0)
		output_vector.append(0)
	else:
		
		output_vector.append(average(sm_activity))
	return output_vector

def parse_args():
        parser = ArgumentParser(description=__doc__,
                        formatter_class=ArgumentDefaultsHelpFormatter)
        parser.add_argument('output_csv_file',
                        help='file to store output vector')

        parser.add_argument('nvprof_log_dir',
                        help='directory where nvprof logs are stored')
        args = parser.parse_args()
        return args
				
def main():
	args = parse_args()
	result_vec = []
	result_vec = parseNvprofLog(args.nvprof_log_dir)
		
	with open(args.output_csv_file,"a") as f:
        	for item in result_vec:
	           f.write(str(item)+',')
		f.write('\n')


if __name__ == '__main__':
	main()
