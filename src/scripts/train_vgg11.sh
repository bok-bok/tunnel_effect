
LR=1e-3
WD=1e-4

GPU=0
SIZES=(128 224)

for SIZE in ${SIZES[@]}
    do
        python trainer.py \
        --size $SIZE \
        --batch_size 256 \
        --lr $LR \
        --wd $WD \
        --epochs 70 \
        --gpu $GPU

    done