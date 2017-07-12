#!/bin/bash 


if [ -z "$1" ]
then
echo "please specify prefix Ex) CONV, POOLING, IP"
exit
fi

if [ -z "$2" ]
then 
echo "please specify type of layer of for prefix EX) CONV -> Convolutopn, POOLING -> Pooling"
exit
fi
rm -rf local-$1
./filterAllData.sh $1
LOCAL_IP="143.248.139.71"
TAR_FILE="deep5-"$1".tar"
DATA_DIR="/home/sbchoi/git/gpu-cloud/predictor/data"

tar cvf $TAR_FILE local-$1/
scp $TAR_FILE $LOCAL_IP:$PWD
# enter PW
echo "sended file "$TAR_FILE" to "$LOCAL_IP
echo "now releasing and combining"
ssh $LOCAL_IP "mv $PWD/$TAR_FILE $DATA_DIR; cd $DATA_DIR; rm -rf local-$1;tar xvf $TAR_FILE; cat local-$1/$1-input_vec.txt >> trn4_$2/input_vec.txt;cat local-$1/$1-output_vec.txt >> trn4_$2/output_vec.txt"

