ARCH   := $(shell getconf LONG_BIT)

ifeq (${ARCH},$(filter ${ARCH},32 64))
   NVML_LIB := /usr/src/gdk/nvml/lib
else
   $(error Unknown architecture!)
endif

CFLAGS  := -I ../include -I /usr/include/nvidia/gdk/
LDFLAGS := -lnvidia-ml -L $(NVML_LIB) 
mon: mon.o
	$(CC) $< $(LDFLAGS) -o $@
	-@mv mon nvml_mon

clean:
	-@rm -f mon.o
	-@rm -f nvml_mon
