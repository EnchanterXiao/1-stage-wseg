#!/bin/bash

#
# Set your argument here
#
CONFIG=configs/voc_resnet38.yaml
DATASET=pascal_voc
#FILELIST=../1sw/data/val_voc.txt
FILELIST=../1sw/data/train_augvoc.txt

## You values here (see below how they're used)
#
OUTPUT_DIR=../1sw/output
EXP=v1013
RUN_ID=cam_casa_wgap_v5
SNAPSHOT=e015Xs0.904
EXTRA_ARGS=
SAVE_ID=cam_casa_wgap_v5
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

python cam.py --dataset $DATASET \
                         --cfg $CONFIG \
                         --exp $EXP \
                         --run $RUN_ID \
                         --resume $SNAPSHOT \
                         --infer-list $FILELIST \
                         --workers $NUM_THREADS \
                         --mask-output-dir $SAVE_DIR \
                         --image-path /home/lwq/sdb1/xiaoxin/data/VOC2012/VOCdevkit/VOC2012/JPEGImages/2007_007881.jpg
                         $EXTRA_ARGS

