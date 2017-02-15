# importable modulve version of nvml_monitor.py

from pynvml import *

nvmlInit()
GPU0 = nvmlDeviceGetHandleByIndex(0)
# create reference objects for each GPU there are total four

#GPU1 = nvmlDeviceGetHandleByIndex(1)
#GPU2 = nvmlDeviceGetHandleByIndex(2)
#GPU3 = nvmlDeviceGetHandleByIndex(3)

#create log file to store resource usage results
log = open("/home/sbchoi/GPU_CLOUD/models/single_layer_models/monitor_log.txt",'w')
log.write("Memory Usage(%), PCI Throughput(GB/s), Core Util(%), Mem Util(%)\n")	#for guidance(I keep forgeting ;;)

recordFlag = False

def startRecord():
	






def recordUtil():
	# checks how many process is running on GPU
	Procs=nvmlDeviceGetComputeRunningProcesses(GPU0)	
	if not Procs:
		return 1
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
	return 0
def closeNvml():
	log.close()
	nvmlShutdown()
	return 0
