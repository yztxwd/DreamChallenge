#!/bin/bash

# train
parallel --jobs 1 --ungroup -k python3 tpu_ddp.py --logdir loggers/hybrid_conv_num{1}_dropout{2} --num_hybrid_conv {1} --dropout {2} --batch_size 512 --num_workers 32 --num_cores 8 --patience 10 ::: 1 2 3 ::: 0.2 0.5

# predict
python tpu_predict.py --num_workers 1 --num_hybrid_conv 3 --dropout 0.5 --ckpt loggers/hybrid_conv_num3_dropout0.5/checkpoint-best.ckpt

# cp to bucket
gsutil cp loggers/hybrid_conv_num3_dropout0.5/hybrid_conv_num3_dropout0.5_submission.json gs://dream-challenge/data/

# test
for i in `seq 2 9`; do
    python tpu_predict.py --num_workers 1 --num_hybrid_conv 1 --dropout 0.5 --ckpt loggers/hybrid_conv_num1_dropout0.5/checkpoint-epoch${i}.ckpt --name epoch${i}_
done

parallel --dry-run --jobs 1 --ungroup python tpu_predict.py --num_workers 1 --num_hybrid_conv {1} --dropout {2} --ckpt loggers/hybrid_conv_num{1}_dropout{2}/checkpoint-{3}.ckpt --name checkpoint-{3}_ ::: 2 3 ::: 0.5 0.8 ::: best last