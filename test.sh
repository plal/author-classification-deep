BATCH_SIZE=16
CKPT="runs/initial_exp_resnet34/checkpoints/WordsModel_25_0.9164.pt"

python test.py \
-ckpt=$CKPT \
-b=$BATCH_SIZE
