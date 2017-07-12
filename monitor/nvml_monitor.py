from pynvml import *
import time
nvmlInit()
# create reference objects for each GPU there are total four
GPU0 = nvmlDeviceGetHandleByIndex(0)
#GPU1 = nvmlDeviceGetHandleByIndex(1)
#GPU2 = nvmlDeviceGetHandleByIndex(2)
#GPU3 = nvmlDeviceGetHandleByIndex(3)

#create log file to store resource usage results
log = open("/home/sbchoi/monitor_log.txt",'w')

 
check = 0
finished = 0

while finished == 0: 
	# checks how many process is running on GPU
	Procs=nvmlDeviceGetComputeRunningProcesses(GPU0)	
	if not Procs:
		if check == 0:
			continue
		elif check == 1:
			finished = 1
	if check == 0:
		log.write("Memory Usage(%), PCI Throughput(GB/s), Core Util(%), Mem Util(%)\n")	#for guidance(I keep forgeting ;;)
	check = 1
	print "nvml running! \n"
	# write performance counter	
	TotalDeviceMem = nvmlDeviceGetMemoryInfo(GPU0).total
	UsedDeviceMem = nvmlDeviceGetMemoryInfo(GPU0).used
	UsedMemoryPercent = float(UsedDeviceMem) / float(TotalDeviceMem) * 100
	PCI_TX = nvmlDeviceGetPcieThroughput(GPU0,NVML_PCIE_UTIL_TX_BYTES)
	PCI_RX = nvmlDeviceGetPcieThroughput(GPU0,NVML_PCIE_UTIL_RX_BYTES)
	PCI = PCI_TX + PCI_RX
	PCI_GB_S= (float(PCI) / 1024) /1024
	GPUUtil = nvmlDeviceGetUtilizationRates(GPU0).gpu
	MemUtil = nvmlDeviceGetUtilizationRates(GPU0).memory
	ResultString = str(UsedMemoryPercent)+","+str(PCI_GB_S)+","+str(GPUUtil)+","+str(MemUtil)+"\n"
	log.write(ResultString)
	#time.sleep(1) # updated in 1 second intervals 
log.close()
nvmlShutdown()
