from pynvml import *
import time
import sys
# create reference objects for each GPU there are total four
#GPU1 = nvmlDeviceGetHandleByIndex(1)
#GPU2 = nvmlDeviceGetHandleByIndex(2)
#GPU3 = nvmlDeviceGetHandleByIndex(3)

#create log file to store resource usage results
if len(sys.argv) == 1:
	print("requires the directory of log file")
	sys.exit()	
log_dir=sys.argv[1]
log = open(log_dir+"/monitor_log.txt",'w')
nvmlInit()
GPU0 = nvmlDeviceGetHandleByIndex(0) 
check = 0
finished = 0
ResultString = ""
while finished == 0: 
	# checks how many process is running on GPU
	Procs=nvmlDeviceGetComputeRunningProcesses(GPU0)	
	if not Procs:
		if check == 0:
			continue
		elif check == 1:
			elapsed_time = time.time() - start_time
			print(elapsed_time)
			log.write(ResultString)
			finished = 1
	if check == 0:
		log.write("Memory Usage(%), PCI TX(GB/s), PCI RX(GB/s), Core Util(%), Mem Util(%)\n")	#for guidance(I keep forgeting ;;)
		start_time = time.time()
	check = 1
	# write performance counter
        #start_time = time.time()	
	TotalDeviceMem = nvmlDeviceGetMemoryInfo(GPU0).total
	UsedDeviceMem = nvmlDeviceGetMemoryInfo(GPU0).used
	UsedMemoryPercent = float(UsedDeviceMem) / float(TotalDeviceMem) * 100
	PCI_TX = nvmlDeviceGetPcieThroughput(GPU0,NVML_PCIE_UTIL_TX_BYTES)
	PCI_RX = nvmlDeviceGetPcieThroughput(GPU0,NVML_PCIE_UTIL_RX_BYTES)
	PCI_TX = float(PCI_TX) / 1024 / 1024
	PCI_RX = float(PCI_RX) /1024 / 1024
	#PCI = PCI_TX + PCI_RX
	#PCI_GB_S= (float(PCI) / 1024) /1024
	GPUUtil = nvmlDeviceGetUtilizationRates(GPU0).gpu
	MemUtil = nvmlDeviceGetUtilizationRates(GPU0).memory
	ResultString = ResultString + str(UsedMemoryPercent)+","+str(PCI_TX)+","+str(PCI_RX)+","+str(GPUUtil)+","+str(MemUtil)+"\n"
	#elapsed_time = time.time() - start_time
        #print(elapsed_time)
	#time.sleep(1) # updated in 1 second intervals 
log.close()
nvmlShutdown()
