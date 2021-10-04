#!/bin/bash

#
# Set your argument here
#
CONFIG=configs/voc_resnet38.yaml
DATASET=pascal_voc
FILELIST=../1sw/data/val_voc.txt
#FILELIST=../1sw/data/train_augvoc.txt

## You values here (see below how they're used)
#
OUTPUT_DIR=../1sw/output
EXP=v0930
RUN_ID=bsl
SNAPSHOT=e014Xs0.945
EXTRA_ARGS=
SAVE_ID=bsl
#
##

# limiting threads
NUM_THREADS=6

set OMP_NUM_THREADS=$NUM_THREADS
export OMP_NUM_THREADS=$NUM_THREADS

#
# Code goes here
#
LISTNAME=`basename $FILELIST .txt`
SAVE_DIR=$OUTPUT_DIR/$DATASET/$EXP/$SAVE_ID/$LISTNAME
LOG_FILE=$OUTPUT_DIR/$DATASET/$EXP/$SAVE_ID/$LISTNAME.log

CMD="python infer_val.py --dataset $DATASET \
                         --cfg $CONFIG \
                         --exp $EXP \
                         --run $RUN_ID \
                         --resume $SNAPSHOT \
                         --infer-list $FILELIST \
                         --workers $NUM_THREADS \
                         --mask-output-dir $SAVE_DIR \
                         $EXTRA_ARGS"

#if [ ! -d $SAVE_DIR ]; then
#  echo "Creating directory: $SAVE_DIR"
#  mkdir -p $SAVE_DIR
#else
#  echo "Saving to: $SAVE_DIR"
#fi

git rev-parse HEAD > ${SAVE_DIR}.head
git diff > ${SAVE_DIR}.diff
echo $CMD > ${SAVE_DIR}.cmd

echo $CMD
#$CMD
nohup $CMD
nohup $CMD > $LOG_FILE 2>&1 &

sleep 1
tail -f $LOG_FILE
