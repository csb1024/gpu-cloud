from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import csv
import numpy as np

def parse_args():
	parser = ArgumentParser(description=__doc__,
		formatter_class=ArgumentDefaultsHelpFormatter)
	parser.add_argument('input_file',
			help = 'input file you want to re-write')
	args = parser.parse_args()
	return args

def readInput(input_txt):
	vectors = []
	with open(input_txt,"rb") as csvfile:
		reader = csv.reader(csvfile,delimiter=',')
		for row in reader:
			del row[-1]
			narray = np.array(row,dtype=np.float)
			vectors.append(narray)
		return vectors

def main():
	args = parse_args()

	vectors = readInput(args.input_file)
	new_vector = []
	for i in range(len(vectors)):
		new_vector.append(vectors[i][0])

	with open (args.input_file,'w') as f:
#		np.savetxt(f,new_vector,delimiter=',')

		for item in new_vector:
			f.write(str(item)+',')
			f.write('\n')
		
		
	

	
	


if __name__=='__main__':
	main()
