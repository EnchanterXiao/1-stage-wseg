#!/bin/bash

## Your values here:
#
DS=pascal_voc
EXP=v1106
RUN_ID=cam_casa_wgap_pcm
#
##

#
# Script
#

LOG_DIR=../1sw/logs/${DS}/${EXP}
CMD="python train.py --dataset $DS --cfg configs/voc_resnet38.yaml --exp $EXP --run $RUN_ID"
#CMD="python train.py --dataset $DS --cfg configs/voc_resnet38.yaml --exp $EXP --run $RUN_ID --isattention True"
LOG_FILE=$LOG_DIR/${RUN_ID}.log

if [ ! -d "$LOG_DIR" ]; then
  echo "Creating directory $LOG_DIR"
  mkdir -p $LOG_DIR
fi

echo $CMD
echo "LOG: $LOG_FILE"

nohup $CMD > $LOG_FILE 2>&1 &
sleep 1
tail -f $LOG_FILE
