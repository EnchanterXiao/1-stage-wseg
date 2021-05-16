#!/bin/bash

DATASET=pascal_voc
FILELIST=data/val_voc.txt # validation

## You values here:
#
OUTPUT_DIR=./output
EXP=v0513
RUN_ID=ae
#
##


LISTNAME=`basename $FILELIST .txt`

# without CRF
SAVE_DIR=$OUTPUT_DIR/$DATASET/$EXP/$RUN_ID/$LISTNAME
nohup python eval_seg.py --data ./data --filelist $FILELIST --masks $SAVE_DIR > $SAVE_DIR.eval 2>&1 &

# with CRF
SAVE_DIR=$OUTPUT_DIR/$DATASET/$EXP/$RUN_ID/$LISTNAME/crf
nohup python eval_seg.py --data ./data --filelist $FILELIST --masks $SAVE_DIR > $SAVE_DIR.eval 2>&1 &

sleep 1

echo "Log: ${SAVE_DIR}.eval"
tail -f $SAVE_DIR.eval
