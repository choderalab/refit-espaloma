#!/bin/bash

# Check incomplete HDF5 entries that have not been converted to graphs
DATASET_PATH=/home/takabak/data/exploring-rna/download-qca-dataset/openff-default/Dataset/spice-pubchem
python ../../script/check_status.py --hdf5 ${DATASET_PATH}/SPICE-PUBCHEM-OPENFF-DEFAULT.hdf5 > mylist
x=`head -n 1 mylist | wc -l`
if [ $x -eq 0 ]; then
    echo "finished"
    rm mylist
    touch finished
    exit
fi

# Convert incomplete HDF5 entries into graphs
DIR=${PWD}
while read line
do
    idx=`echo $line | awk '{print $1}'`
    keyname=`echo $line | awk '{print $2}'`
    
    mkdir -p data/${idx}
    sed -e 's/@@@JOBNAME@@@/id-'${idx}'/g' \
        -e 's/@@@KEYNAME@@@/'${keyname}'/g' \
        -e 's/@@@INDEX@@@/'${idx}'/g' \
        lsf_submit_template.sh > ./data/${idx}/submit.sh
    chmod u+x ./data/${idx}/submit.sh

    cd data/${idx}
    echo "submit job id-${idx}"
    bsub < submit.sh
    cd ${DIR}
done < mylist