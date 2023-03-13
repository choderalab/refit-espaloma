#!/bin/bash


DATASET_PATH=/home/takabak/data/qca-dataset/openff-default/spice-dipeptide
python check_status.py --hdf5 ${DATASET_PATH}/SPICE-DIPEPTIDE-OPENFF-DEFAULT.hdf5 > mylist
DIR=${PWD}

x=`head -n 1 mylist | wc -l`
if [ $x -eq 0 ]; then
    echo "finished"
    rm mylist
    touch finished
    exit
fi

while read line
do
    idx=`echo $line | awk '{print $1}'`
    keyname=`echo $line | awk '{print $2}'`
    #echo $idx $keyname

    # create directory
    mkdir -p data/${idx}
    sed -e 's/@@@JOBNAME@@@/id-'${idx}'/g' \
        -e 's/@@@KEYNAME@@@/'${keyname}'/g' \
        submit_template.sh > ./data/${idx}/submit.sh
    chmod u+x ./data/${idx}/submit.sh

    # submit job
    cd data/${idx}
    echo "submit job id-${idx}"
    bsub < submit.sh
    cd ${DIR}

    #sleep 1
done < mylist