MODEL="resnet50"
BATCH_SIZE=16
MAX_EPOCH=200
EARLY_STOP=10
LR=0.00001
SEED=2021

python train.py \
--model=$MODEL  \
--batch_size=$BATCH_SIZE \
--max_epoch=$MAX_EPOCH \
--early_stop=$EARLY_STOP \
--lr=$LR \
--seed=$SEED
