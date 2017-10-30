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
2. dram_utilization -> line num 6 and over for every even line number
3. achieved_occupancy -> line num 7 and aover and every odd line number
4. gld_throughput -> line num 9
5. gst_throughput
6. executed_ipc -> line num 10
"""
def average(numbers):
	total = sum(numbers)
	total = float(total)
	return total / len(numbers) 
def parseNvprofLog(nvprof_log_dir):
	#below are the vectors for each kernel 
	sm_activity = []
	dram_util = []
	occupancy = []
	gld_throughput = []
	gst_throughput = []
	exec_ipc = []
	for f in os.listdir(nvprof_log_dir):	#for every file 
		if not os.path.isfile(nvprof_log_dir+"/"+f) : #skip directory
			continue
		with open(nvprof_log_dir+"/"+f) as fp:
			cnt = 1 
			print "Opened "+f
			for line in fp:
				if cnt >= 7 and cnt%6 == 1 : #sm_activity
					words = line.split(',')
					info = words[-1]
					info = info[:-2] #exclude '%\n'
					sm_activity.append(float(info))
				elif cnt >= 7 and cnt%6 == 2 : #dram utilization
					words = line.split(',')
					info = words[-2].split()[-1]
					info = info[:-2] # trim 
					info = info[1:] # trim again
					dram_util.append(int(info))	
				elif cnt >= 7 and cnt%6 == 3 : #achieved occupancy
					words = line.split(',')
					occupancy.append(float(words[-1]))
				elif cnt >=7 and cnt%6 == 4 :
					words = line.split(',')
					info = words[-1]
					info = info[:-5] # trim "GB/s"
					gld_throughput.append(float(info))
				elif cnt >= 7 and cnt % 6 == 5 : 
					words = line.split(',')
					info = words[-1]
					info = info[:-5]
					gst_throughput.append(float(info))
				elif cnt >=7 and cnt%6 == 0 : # executed_ipc
					words = line.split(',')
					exec_ipc.append(float(words[-1]))
				cnt = cnt + 1
	# get max of each kernel, max value is more useful for scheduling
	output_vector = []
	if not dram_util or not occupancy : # the same condition also holds for additional vectors 
		output_vector.append(0)
		output_vector.append(0)
	else:
		
		output_vector.append(average(sm_activity))
		output_vector.append(max(dram_util))
		output_vector.append(max(occupancy))
		output_vector.append(max(gld_throughput))
		output_vector.append(max(gst_throughput))
		output_vector.append(max(exec_ipc))
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
