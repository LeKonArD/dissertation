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
parser.add_argument('-clsmodel', type=str, help='Classification Evaluation Trainging Epochs')
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



cls_evaldata = [
                {"name":"cls_2010", 
                 "file":"/home/ext/konle/diss/data/zeit/cls_2010.tsv"},
                {"name":"cls_2000", 
                 "file":"/home/ext/konle/diss/data/zeit/cls_2000.tsv"},
                {"name":"cls_1990", 
                 "file":"/home/ext/konle/diss/data/zeit/cls_1990.tsv"},
                {"name":"cls_1980", 
                 "file":"/home/ext/konle/diss/data/zeit/cls_1980.tsv"},
                {"name":"cls_1970", 
                 "file":"/home/ext/konle/diss/data/zeit/cls_1970.tsv"},
                {"name":"cls_1960", 
                 "file":"/home/ext/konle/diss/data/zeit/cls_1960.tsv"}
                ]

print("Load Model")
model = BertForMaskedLM.from_pretrained(model_name)
state_dict = torch.load(args.clsmodel)
model.load_state_dict(state_dict)


tokenizer = BertTokenizerFast.from_pretrained(model_name)


print("Create CLS Evaluation Batches")
cls_eval_sets = []
if cls_evaldata != None:
    
    for cls_evalset in cls_evaldata:
        print(cls_evalset)
        train, test = create_cls_batches(tokenizer, model.config.max_position_embeddings, cls_evalset["file"], cls_bsize)
        print(len(train)) 
        print(len(test))

        cls_eval_sets.append({"name":cls_evalset["name"], "train":train, "test":test})


device = "cuda"
re = dict()

for eval_set in cls_eval_sets:

                print(len(eval_set["train"]))
                print(len(eval_set["test"]))
                model = load_checkpoint_cls(model_name, args.clsmodel)
                scores = classify(model, eval_set["train"][:500], eval_set["test"], cls_epochs, cls_lr, device)
                re[eval_set["name"]] = scores

                del model
                torch.cuda.empty_cache()

output = pd.DataFrame(re)
output.to_csv(args.clsmodel+"_cls.tsv", sep="\t")
