#!/bin/bash

DATASET=pascal_voc
FILELIST=../1sw/data/val_voc.txt # validation

## You values here:
#
OUTPUT_DIR=../1sw/output
EXP=v0920
RUN_ID=bsl
Thresh=_1
#
##


LISTNAME=`basename $FILELIST .txt`$Thresh

# with CRF
#SAVE_DIR=$OUTPUT_DIR/$DATASET/$EXP/$RUN_ID/$LISTNAME/crf
#nohup python eval_seg.py --data ./data --filelist $FILELIST --masks $SAVE_DIR > $SAVE_DIR.eval 2>&1 &

# without CRF
SAVE_DIR=$OUTPUT_DIR/$DATASET/$EXP/$RUN_ID/$LISTNAME/no_crf
nohup python eval_seg.py --data ../1sw/data --filelist $FILELIST --masks $SAVE_DIR > $SAVE_DIR.eval 2>&1 &


sleep 1

echo "Log: ${SAVE_DIR}.eval"
tail -f $SAVE_DIR.eval
