from transformers import BertTokenizerFast
from transformers import BertForMaskedLM
from transformers import get_cosine_schedule_with_warmup
from transformers import AdamW
from transformers import BertForSequenceClassification

from pretrain_helper import create_batches
from cls_helper import create_cls_batches, classify, load_checkpoint_cls
from eval_helper import eval_on_batches, create_mlm_evalsets

import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import torch
import argparse

parser = argparse.ArgumentParser(description='Start pretraining with intern and extern evaluation.')
parser.add_argument('-output', type=str, help='file to store tsv-reports')
parser.add_argument('-modelname', type=str, help='identifier from huggingface')

parser.add_argument('-pretrainlr', type=float, help='Learning rate for pretraining')
parser.add_argument('-pretrainbsize', type=int, help='Batchsize for pretraining')
parser.add_argument('-pretrainsteps', type=int, help='Steps for pretraining')
parser.add_argument('-evalinterval', type=int, help='Evaluation Intervall')
parser.add_argument('-pretrainmethod', type=str, help='Masking algorithm')
parser.add_argument('-pretrainfile', type=str, help='Text File for pretraining')


parser.add_argument('-evalbsize', type=int, help='MLM Evaluation Batchsize')
parser.add_argument('-clsbsize', type=int, help='Classification Evaluation Batchsize')
parser.add_argument('-clslr', type=float, help='Classification Evaluation Learningrate')
parser.add_argument('-clsepochs', type=float, help='Classification Evaluation Trainging Epochs')

#parser.add_argument('-mlmevaldata', type=str, help='MLM Evaluation Sets (List of dicts containing keys: [name,file]) as pickle #obj')
#parser.add_argument('-clsevaldata', type=str, help='Classification Evaluation Sets (List of dicts containing keys: [name,file]) as pickle obj')

print("Parse Agruments")
args = parser.parse_args()

# args
output_path = args.output
model_name = args.modelname

# pretrain_args
pretrain_lr = args.pretrainlr
pretrain_bsize = args.pretrainbsize
pretrain_steps = args.pretrainsteps
pretrain_eval_steps = args.evalinterval
pretrain_method = args.pretrainmethod
pretrain_file = args.pretrainfile

# eval_args
eval_bsize = args.evalbsize
cls_bsize = args.clsbsize
cls_lr = args.clslr
cls_epochs = args.clsepochs

mlm_evaldata = [{"name":"oscar",
                 "file":"/home/ext/konle/diss/data/oscar/pre_oscar.txt"},
                {"name":"mlm_2010", 
                 "file":"/home/ext/konle/diss/data/zeit/pre_2010.txt"},
                {"name":"mlm_2000", 
                 "file":"/home/ext/konle/diss/data/zeit/pre_2000.txt"},
                {"name":"mlm_1990", 
                 "file":"/home/ext/konle/diss/data/zeit/pre_1990.txt"},
                {"name":"mlm_1980", 
                 "file":"/home/ext/konle/diss/data/zeit/pre_1980.txt"},
                {"name":"mlm_1970", 
                 "file":"/home/ext/konle/diss/data/zeit/pre_1970.txt"},
                {"name":"mlm_1960", 
                 "file":"/home/ext/konle/diss/data/zeit/pre_1960.txt"}
               ]

print("Load Model")
model = BertForMaskedLM.from_pretrained(model_name)
tokenizer = BertTokenizerFast.from_pretrained(model_name)
print("MLM Evaluation Batches")
batches = None

if mlm_evaldata != None:
    
    eval_sets = create_mlm_evalsets(batches,
                                    tokenizer,
                                    model.config.max_position_embeddings,
                                    mlm_evaldata,
                                    pretrain_method,
                                    pretrain_bsize)
model.cuda()
device = "cuda"
re = dict()

for eval_set in eval_sets:

    eval_loss = eval_on_batches(eval_set["batches"], model, device)
    re[eval_set["name"]] = eval_loss

output = pd.DataFrame.from_dict(re, orient="index")
output.to_csv("/home/ext/konle/diss/mlm_startpoint.tsv", sep="\t")
