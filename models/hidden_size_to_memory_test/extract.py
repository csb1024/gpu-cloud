
file=open("duration.txt","r")
writing_file=open("parsed.txt","w")
i=1
row = file.readlines()
for line in row:
	if i % 6 == 5:
		writing_file.write(line)
	i=i+1
