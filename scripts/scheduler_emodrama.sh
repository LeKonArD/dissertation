#!/bin/bash
export TRANSFORMERS_CACHE=/home/ext/konle/diss
python3 /home/ext/konle/diss/code/pretrainer.py -output "/home/ext/konle/diss/logs_emo_drama.tsv" -modelname "deepset/gbert-large" -pretrainlr 0.0005 -pretrainbsize 10 -pretrainsteps 10000 -evalinterval 1000 -pretrainmethod "bert" -pretrainfile "/home/ext/konle/diss/data/cls/textresources/emodrama_text.txt" -evalbsize 30 -clsbsize 20 -clslr 1e-5 -clsepochs 10
