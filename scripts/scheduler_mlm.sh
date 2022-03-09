#!/bin/bash
export TRANSFORMERS_CACHE=/home/ext/konle/diss
python3 /home/ext/konle/diss/code/mlm_eval.py -output "/home/ext/konle/diss/lyrik_run.tsv" -modelname "deepset/gbert-large" -pretrainlr 1e-4 -pretrainbsize 5 -pretrainsteps 100000 -evalinterval 100 -pretrainmethod "bert" -pretrainfile "/home/ext/konle/diss/data/textresources/lyrik.txt" -evalbsize 5 -clsbsize 20 -clslr 1e-5 -clsepochs 10
