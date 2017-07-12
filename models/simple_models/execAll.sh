#!/bin/bash

seconds=('40' '35' '30' '25' '20' '15' '10' '5')
second_element=${#seconds[@]}

for (( h=0; h<${second_element}; h++));  
do
second=${seconds[${h}]}  
./exec.sh $second
done
