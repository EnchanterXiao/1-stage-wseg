#!/bin/bash

DATASET=pascal_voc
FILELIST=../1sw/data/val_voc.txt # validation
#FILELIST=../1sw/data/train_augvoc.txt
## You values here:
#
OUTPUT_DIR=../1sw/output
EXP=v1122
RUN_ID=cam_casa_wgap_tf_v7
Thresh0=_0
Thresh1=_1
Thresh3=_3
Thresh5=_5
Thresh7=_7



LISTNAME=`basename $FILELIST .txt`

# without CRF
SAVE_DIR=$OUTPUT_DIR/$DATASET/$EXP/$RUN_ID/$LISTNAME$Thresh0/no_crf
nohup python eval_seg.py --data ../1sw/data --filelist $FILELIST --masks $SAVE_DIR > $SAVE_DIR.eval 2>&1 &

SAVE_DIR=$OUTPUT_DIR/$DATASET/$EXP/$RUN_ID/$LISTNAME$Thresh1/no_crf
nohup python eval_seg.py --data ../1sw/data --filelist $FILELIST --masks $SAVE_DIR > $SAVE_DIR.eval 2>&1 &

#SAVE_DIR=$OUTPUT_DIR/$DATASET/$EXP/$RUN_ID/$LISTNAME$Thresh3/no_crf
#nohup python eval_seg.py --data ../1sw/data --filelist $FILELIST --masks $SAVE_DIR > $SAVE_DIR.eval 2>&1 &
#
#SAVE_DIR=$OUTPUT_DIR/$DATASET/$EXP/$RUN_ID/$LISTNAME$Thresh5/no_crf
#nohup python eval_seg.py --data ../1sw/data --filelist $FILELIST --masks $SAVE_DIR > $SAVE_DIR.eval 2>&1 &
#
#SAVE_DIR=$OUTPUT_DIR/$DATASET/$EXP/$RUN_ID/$LISTNAME$Thresh7/no_crf
#nohup python eval_seg.py --data ../1sw/data --filelist $FILELIST --masks $SAVE_DIR > $SAVE_DIR.eval 2>&1 &

# with CRF
# SAVE_DIR=$OUTPUT_DIR/$DATASET/$EXP/$RUN_ID/$LISTNAME/crf
# nohup python eval_seg.py --data ./data --filelist $FILELIST --masks $SAVE_DIR > $SAVE_DIR.eval 2>&1 &

sleep 1
echo "Log: ${SAVE_DIR}.eval"
tail -f $SAVE_DIR.eval

