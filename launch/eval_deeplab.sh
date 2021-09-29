#!/bin/bash

DATASET=pascal_voc
FILELIST=../1sw/data/val_voc.txt # validation

OUTPUT_DIR=/home/lwq/sdb1/xiaoxin/WSSS/1sw/output/baseline/
LISTNAME=`basename $FILELIST .txt`
SAVE_DIR=/home/lwq/sdb1/xiaoxin/WSSS/1sw/output/baseline/mask
nohup python eval_seg.py --data ../1sw/data --filelist $FILELIST --masks $SAVE_DIR > $SAVE_DIR.eval 2>&1 &

sleep 1

echo "Log: ${SAVE_DIR}.eval"
tail -f $SAVE_DIR.eval