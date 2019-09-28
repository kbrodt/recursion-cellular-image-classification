#!/bin/bash

for exp in 'HEPG2' 'HUVEC' 'RPE' 'U2OS'
do
  python ./src/train.py --cell-type $exp --gpus 0 --fp16 --with-plates
done

python ./src/predict.py --gpus 0 --fp16 --with-plates
