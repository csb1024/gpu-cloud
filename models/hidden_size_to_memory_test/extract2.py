
writing_file=open("nvml_parsed.txt","w")
for i in xrange(1, 31): # 1 ~ 30
	i=i*10
	input_file = "log_"+str(i)+".txt"
	fin = open(input_file,"r")
	lines = fin.read().splitlines()
	last_line = lines[-1]
	last_line=last_line+'\n'
	writing_file.write(last_line)

