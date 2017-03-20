import sys

"""
output vector
Maximum required Device Memory(%)
Required Maximum PCI TX Throughput(%)
Required Maximum PCI RX Throughput(%)
Maximum Core Util
Maximum Mem Util
"""
if len(sys.argv) != 2 :
	print "please specify log file"
	sys.exit(1)

i=0 # flag for skipping the first line  

dev_mem = []
pci_tx = []
pci_rx = []
core_util = []
mem_util = []
with open(sys.argv[1]) as fin:
	for line in fin:
		if i == 0:
			i = i + 1
			continue
		values = line.split(",")
		dev_mem.append(values[0])
		pci_tx.append(values[1])
		pci_rx.append(values[2])
		core_util.append(values[3])
		mem_util.append(values[4])
output_vector = []
output_vector.append(max(dev_mem))
output_vector.append(max(pci_tx))
output_vector.append(max(pci_rx))
output_vector.append(max(core_util))
output_vector.append(max(mem_util))

# for debugging
i = 0
while i < len(output_vector):
	print i+1,"th item : ",output_vector[i]
	i = i+1
