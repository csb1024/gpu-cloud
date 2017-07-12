
#include <math.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
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

int main()
{
	//struct timeval mytime;
	//gettimeofday(&mytime, NULL);
	//printf("%ld:%ld\n", mytime.tv_sec, mytime.tv_usec);
	printf("%s \n", time_stamp());
		    
			        return 0;
}
