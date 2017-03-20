#include <stdio.h>
#include <stdlib.h>
#include <nvml.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
int main(int argc, char* argv[])
{
    nvmlReturn_t result;
    unsigned int device_count, i;
    
    char check, finish; //flag values
    char buffer1[100]={0,};
    nvmlDevice_t gpu0, gpu1, gpu2, gpu3;
    unsigned int g0_link_bandwidth, currLinkGen, maxBandwidth;
    nvmlProcessInfo_t pInfo[32];
    unsigned int nProc = 32;
    unsigned int  pciInfo;
    float rx, tx;
    float memUsage;
    nvmlUtilization_t utilInfo;
    nvmlMemory_t memInfo;
    struct timespec stop, start;
    double seconds;

    FILE* fp;
    if( argc == 1){
	printf("Please specify the directory of the log\n");
        return 1;
    }   
    strcpy(buffer1,argv[1]);
    strcat(buffer1,"/monitor_log.txt");
    printf("%s \n",buffer1);
    fp = fopen(buffer1,"w");   
    //First initialize NVML library
    nvmlInit();
    nvmlDeviceGetHandleByIndex(0, &gpu0);
    nvmlDeviceGetCurrPcieLinkWidth(gpu0, &g0_link_bandwidth);
    nvmlDeviceGetCurrPcieLinkGeneration(gpu0, &currLinkGen); 
    if (currLinkGen == 1)
	    maxBandwidth = 0.25 * g0_link_bandwidth;
    else if(currLinkGen == 2)
	    maxBandwidth = 0.5 * g0_link_bandwidth;
    else if(currLinkGen == 3)
	    maxBandwidth = 0.985 * g0_link_bandwidth;
    else //gen4
	    maxBandwidth = 1.969 * g0_link_bandwidth;


    printf("link width : %u \n",g0_link_bandwidth);
    printf("link generation : %u\n",currLinkGen);
    printf("link max bandwidth : %u GB/s \n",maxBandwidth);

    check=0;

    finish=0;
    while (finish == 0){
    	nvmlDeviceGetComputeRunningProcesses(gpu0,&nProc,pInfo);
    	if(nProc == 0){
		if(check == 1 )
		{
			clock_gettime(CLOCK_MONOTONIC_RAW,&stop);
			finish=1;
			//printf("%4f elapsed \n",  difftime(end, begin));
			printf("Elapsed : %4f \n",((double)(stop.tv_sec - start.tv_sec)) + ((double)(stop.tv_nsec - start.tv_nsec)) / 1000000000);
		
		}
		continue;
	}	
	if (check == 0){
		fprintf(fp,"Memory Usage, PCI TX(%%), PCI RX(%%), Core Util, Mem Util \n");
		clock_gettime(CLOCK_MONOTONIC_RAW, &start);
	}
	check = 1;
	
	nvmlDeviceGetMemoryInfo(gpu0,&memInfo);
	nvmlDeviceGetPcieThroughput(gpu0, NVML_PCIE_UTIL_TX_BYTES,&pciInfo);
	tx=pciInfo;
	nvmlDeviceGetPcieThroughput(gpu0, NVML_PCIE_UTIL_RX_BYTES,&pciInfo);
	rx=pciInfo;
	
	tx= ((float)tx)/1024/1024;
	rx= ((float)rx)/1024/1024;
	tx = tx / maxBandwidth * 100;
	rx = tx / maxBandwidth * 100;

        
	memUsage = ((float)memInfo.used) / memInfo.total*100;
	result=nvmlDeviceGetUtilizationRates(gpu0, &utilInfo);
	fprintf(fp,"%3f,%3f,%3f,%u,%u \n",memUsage,tx,rx,utilInfo.gpu,utilInfo.memory);
	//printf("%3f,%3f,%3f,%u,%u \n",memUsage,tx,rx,utilInfo.gpu,utilInfo.memory);
	fflush(fp);
        
 		
    }
    // get handle of each GPU
       //nvmlDeviceGetHandleByIndex(i, &gpu1);
    //nvmlDeviceGetHandleByIndex(i, &gpu2);
    //nvmlDeviceGetHandleByIndex(i, &gpu3);
    fclose(fp);   
    result = nvmlShutdown();
    return 0;
}
