#!/bin/bash

## Your values here:
DS=pascal_voc
EXP=v1124
RUN_ID=cam_casa_wgap_tf
#热加载参数
SNAPSHOT=e005Xs0.914
start_epoch=5

#
# Script命令
#
#冷启动
CMD="python train.py --dataset $DS --cfg configs/voc_resnet38.yaml --exp $EXP --run $RUN_ID"

#热启动
#CMD="python train.py --dataset $DS --cfg configs/voc_resnet38.yaml --exp $EXP --run $RUN_ID --resume $SNAPSHOT --start_epoch $start_epoch"

#SEAM训练脚本
#CMD="python train_SEAM.py --dataset $DS --cfg configs/voc_resnet38.yaml --exp $EXP --run $RUN_ID"

#使用attn loss的训练
#CMD="python train.py --dataset $DS --cfg configs/voc_resnet38.yaml --exp $EXP --run $RUN_ID --isattention True"

#训练日志
LOG_DIR=../1sw/logs/${DS}/${EXP}
LOG_FILE=$LOG_DIR/${RUN_ID}.log
if [ ! -d "$LOG_DIR" ]; then
  echo "Creating directory $LOG_DIR"
  mkdir -p $LOG_DIR
fi

echo $CMD
echo "LOG: $LOG_FILE"

#执行命令
nohup $CMD > $LOG_FILE 2>&1 &
sleep 1
tail -f $LOG_FILE
