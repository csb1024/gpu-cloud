#include <stdio.h>
#include <stdlib.h>
#include <nvml.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include<signal.h>
#include<unistd.h>

//global flag for checking to end monitor program
int finish;

char *time_stamp(){
	char *timestamp = (char *)malloc(sizeof(char) * 16);
	time_t ltime;
	ltime=time(NULL);
	struct tm *tm;
	struct timeval tv; 
	int millis;
	tm=localtime(&ltime);
	gettimeofday(&tv,NULL);
	millis =(tv.tv_usec) / 1000 ;
	sprintf(timestamp,"%02d:%02d:%02d:%03d", tm->tm_hour, tm->tm_min, tm->tm_sec,millis);
	return timestamp;
}

void sig_handler(int signo);


int main(int argc, char* argv[])
{
    nvmlReturn_t result;
    unsigned int device_count, i;
    
    char check; //flag values
    char buffer1[100]={0,};
    nvmlDevice_t gpu0;
    unsigned int g0_link_bandwidth, currLinkGen;
    float maxBandwidth;
    nvmlProcessInfo_t pInfo[32];
    unsigned int initnProc, nProc = 32;
    unsigned int  pciInfo;
    float rx, tx;
    float memUsage;
    nvmlUtilization_t utilInfo;
    nvmlMemory_t memInfo;
    struct timespec stop, start;
    double seconds;
    int num;
   



    FILE* fp;
     
    if( argc == 1){
	printf("Please specify the directory of the log\n");
        return 1;
    } 
    if (argc == 2 ){
	printf("Please specify gpu id, monitor will exit\n");
	return 1;	
    }  
    else{
	num = atoi(argv[2]);
	if ( !( -1 <num || num < 4 )){
		printf("gpu ID must be between 0 ~ 3 ");
		return 1;
	}			
    }
      
    signal(SIGINT,(void *)sig_handler);

    strcpy(buffer1,argv[1]);
    strcat(buffer1,"/monitor_log.txt");
    printf("%s \n",buffer1);
    fp = fopen(buffer1,"w");   
    //First initialize NVML library
    nvmlInit();
    nvmlDeviceGetHandleByIndex(num, &gpu0);
    nvmlDeviceGetCurrPcieLinkWidth(gpu0, &g0_link_bandwidth);
    nvmlDeviceGetCurrPcieLinkGeneration(gpu0, &currLinkGen); 
    nvmlDeviceGetMemoryInfo(gpu0,&memInfo);

    if (currLinkGen == 1)
	    maxBandwidth = 0.25 * g0_link_bandwidth;
    else if(currLinkGen == 2)
	    maxBandwidth = 0.5 * g0_link_bandwidth;
    else if(currLinkGen == 3)
	    maxBandwidth = 0.985 * g0_link_bandwidth;
    else //gen4
	    maxBandwidth = 1.969 * g0_link_bandwidth;

    maxBandwidth = 15.76; // decided to hardcode this value
    printf("link width : %u \n",g0_link_bandwidth);
    printf("link generation : %u\n",currLinkGen);
    printf("link max bandwidth : %f GB/s \n",maxBandwidth);
    printf("Total memory : %3f MB\n",(float)memInfo.total/1024/1024);
    check=0;

    finish=0;
//    nvmlDeviceGetComputeRunningProcesses(gpu0,&nProc,pInfo);
//    initnProc = nProc;
    while (1){
//	printf("%d\n",finish);
 //   	nvmlDeviceGetComputeRunningProcesses(gpu0,&nProc,pInfo);
	if(finish == 1 )
	{
		clock_gettime(CLOCK_MONOTONIC_RAW,&stop);
		//printf("%4f elapsed \n",  difftime(end, begin));
		fprintf(fp,"Elapsed : %4f \n",((double)(stop.tv_sec - start.tv_sec)) + ((double)(stop.tv_nsec - start.tv_nsec)) / 1000000000);
		fflush(fp);
		printf("%d\n",finish);
		break;
		
	}
		
	if (check == 0){
		printf("Started Profiling\n");
		fprintf(fp,"Timestamp, Memory Usage(%%), PCI TX(%%), PCI RX(%%), Core Util, Mem Util \n");
		clock_gettime(CLOCK_MONOTONIC_RAW, &start);
	}
	check = 1;
	nvmlDeviceGetMemoryInfo(gpu0,&memInfo);
	nvmlDeviceGetPcieThroughput(gpu0, NVML_PCIE_UTIL_TX_BYTES,&pciInfo);
	tx=pciInfo;
	nvmlDeviceGetPcieThroughput(gpu0, NVML_PCIE_UTIL_RX_BYTES,&pciInfo);
	rx=pciInfo;
	
	tx = ((float)tx)/1024/1024;
	rx = ((float)rx)/1024/1024;
	tx = tx/maxBandwidth*100;
	rx = rx/maxBandwidth*100;       
	memUsage = ((float)memInfo.used) / memInfo.total*100;
	result=nvmlDeviceGetUtilizationRates(gpu0, &utilInfo);
	fprintf(fp,"%s, %3f,%3f,%3f,%u,%u \n",time_stamp(),memUsage,tx,rx,utilInfo.gpu,utilInfo.memory);
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

void sig_handler(int signo)
{
   printf("Recieved SIGINT\n");
   finish = 1;
}
