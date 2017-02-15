#!/bin/bash

#This script processes the output of nvprof --print-gpu-trace
PROFILE_FILE='out.txt'
LIST_FILE='kernel_list.txt'
#The following sequence shows which field is storing what information
#"Start","Duration","Grid X","Grid Y","Grid Z","Block X","Block Y","Block Z","Registers Per Thread","Static SMem","Dynamic SMem","Size","Throughput","Device","Context","Stream","Name"

#export IFS=","
#cat ${FILE} | while read a b c d; do echo "$a:$b:$c:$d";done

awk -F "\"*,\"*" '{print $17}' ${PROFILE_FILE} > ${LIST_FILE}
