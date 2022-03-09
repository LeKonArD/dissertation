from transformers import BertTokenizerFast
from transformers import BertForMaskedLM
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
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


mlm_evaldata = [{"name":"lyrik", "file":"/home/ext/konle/diss/data/cls/textresources/gerdracor.txt"}]

#mlm_evaldata = [{"name":"oscar",
#                 "file":"/home/ext/konle/diss/data/oscar/pre_oscar.txt"},
#                {"name":"mlm_2010", 
#                 "file":"/home/ext/konle/diss/data/zeit/pre_2010.txt"},
#                {"name":"mlm_2000", 
#                 "file":"/home/ext/konle/diss/data/zeit/pre_2000.txt"},
#                {"name":"mlm_1990", 
#                 "file":"/home/ext/konle/diss/data/zeit/pre_1990.txt"},
#                {"name":"mlm_1980", 
#                 "file":"/home/ext/konle/diss/data/zeit/pre_1980.txt"},
#                {"name":"mlm_1970", 
#                 "file":"/home/ext/konle/diss/data/zeit/pre_1970.txt"},
#                {"name":"mlm_1960", 
#                 "file":"/home/ext/konle/diss/data/zeit/pre_1960.txt"}
#               ]
cls_evaldata = None
#cls_evaldata = [{"name":"cls_2010", 
#                 "file":"/home/ext/konle/diss/data/zeit/cls_2010.tsv"},
#                {"name":"cls_2000", 
#                 "file":"/home/ext/konle/diss/data/zeit/cls_2000.tsv"},
#                {"name":"cls_1990", 
#                 "file":"/home/ext/konle/diss/data/zeit/cls_1990.tsv"},
#                {"name":"cls_1980", 
#                 "file":"/home/ext/konle/diss/data/zeit/cls_1980.tsv"},
#                {"name":"cls_1970", 
#                 "file":"/home/ext/konle/diss/data/zeit/cls_1970.tsv"},
#                {"name":"cls_1960", 
#                 "file":"/home/ext/konle/diss/data/zeit/cls_1960.tsv"}
#                ]

print("Load Model")
model = BertForMaskedLM.from_pretrained(model_name)
state_dict = torch.load("/home/ext/konle/diss/eval_drama_checkpoint_10000.pt")
model.load_state_dict(state_dict)


tokenizer = BertTokenizerFast.from_pretrained(model_name)

print("Create Pretrain Batches")
batches = create_batches(tokenizer,  
                         pretrain_file, 
                         pretrain_bsize, 
                         pretrain_method,
                         model.config.max_position_embeddings)

print(str(len(batches))+" pretrain batches")
print("MLM Evaluation Batches")

if mlm_evaldata != None:
    
    eval_sets = create_mlm_evalsets(batches, 
                                    tokenizer, 
                                    model.config.max_position_embeddings, 
                                    mlm_evaldata, 
                                    pretrain_method,
                                    pretrain_bsize)
    
print("Create CLS Evaluation Batches")
cls_eval_sets = []
if cls_evaldata != None:
    
    for cls_evalset in cls_evaldata:
        print(cls_evalset)
        train, test = create_cls_batches(tokenizer, model.config.max_position_embeddings, cls_evalset["file"], cls_bsize)
        
        
        cls_eval_sets.append({"name":cls_evalset["name"], "train":train, "test":test})

pre_optimizer = AdamW(model.parameters(),lr = pretrain_lr, )
pre_scheduler = get_linear_schedule_with_warmup(pre_optimizer, num_warmup_steps = int(pretrain_steps*0.06), num_training_steps = pretrain_steps)

model.train()
model.cuda()
device = "cuda"


i = 0
total_loss = 0
report = []
epoch_num = 0
for step in list(range(pretrain_steps)):
      
    x = torch.tensor(batches[i][0]).to(device)
    m = torch.tensor(batches[i][1]).to(device)
    y = torch.tensor(batches[i][2]).to(device)
    
    model.zero_grad()
    
    outputs = model(x, token_type_ids=None, attention_mask=m, labels=y)
    loss = outputs[0].sum()
    total_loss += loss.item()
    print(loss.item())

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.9)

    pre_optimizer.step()
    pre_scheduler.step()
    
    re = {"train_loss":loss.item(),
          "train_step":step, 
          "num_batches":len(batches), 
          "pretrain_file":pretrain_file,
          "pre_train_lr":pretrain_lr,
          "pretrain_bsize":pretrain_bsize,
          "epoch":epoch_num}
    
    for eval_set in eval_sets:
        re[eval_set["name"]] = np.nan
    
    for eval_set in cls_eval_sets:
        re[eval_set["name"]] = np.nan
    report.append(re)
    output = pd.DataFrame(report)
    output.to_csv(output_path, sep="\t")
    
    if i % pretrain_eval_steps == 0 and i != 0:
        print("eval")
        print("mlm")
        for eval_set in eval_sets:
        
            eval_loss = eval_on_batches(eval_set["batches"], model, device)
            re[eval_set["name"]] = eval_loss
            torch.save(model.state_dict(), "/home/ext/konle/diss/eval_emodrama_checkpoint_"+str(step)+".pt")

        print("cls")
        # classification evaluation
        if len(cls_eval_sets) > 0:
        
            torch.save(model.state_dict(), "/home/ext/konle/diss/eval_emodrama_checkpoint_"+str(step)+".pt")
            torch.save(pre_optimizer.state_dict(), "/home/ext/konle/diss/emodrama_optimizer.pt")
            torch.save(pre_scheduler.state_dict(), "/home/ext/konle/diss/emodrama_scheduler.pt")
            model.to("cpu")
            
            del pre_optimizer
            del pre_scheduler
            del model
            del x
            del m
            del y
            
            torch.cuda.empty_cache()
            
            for eval_set in cls_eval_sets:

                print(len(eval_set["train"]))
                print(len(eval_set["test"]))            
                model = load_checkpoint_cls(model_name, "/home/ext/konle/diss/eval_emodrama_checkpoint_"+str(step)+".pt")
                scores = classify(model, eval_set["train"], eval_set["test"], cls_epochs, cls_lr, device)
                re[eval_set["name"]] = scores
                
                del model
                torch.cuda.empty_cache()
         
        output = pd.DataFrame(report)
        output.to_csv(output_path, sep="\t")
        # load model to continue pretraining
        #model = BertForMaskedLM.from_pretrained(model_name)
        #state_dict = torch.load("/home/ext/konle/diss/eval_lyrik_checkpoint_"+str(step)+".pt")
        #model.load_state_dict(state_dict)
        #model.cuda()
        
        #pre_optimizer = AdamW(model.parameters(),lr = pretrain_lr)
        #pre_scheduler = get_cosine_schedule_with_warmup(pre_optimizer, num_warmup_steps = 0, num_training_steps = pretrain_steps)
        
        #pre_optimizer.load_state_dict(torch.load("/home/ext/konle/diss/lyrik_optimizer.pt"))
        #pre_scheduler.load_state_dict(torch.load("/home/ext/konle/diss/lyrik_scheduler.pt"))
        
    report.append(re)
    
    i+=1
    if i == len(batches):
        epoch_num+=1
        i = 0
        
        
output = pd.DataFrame(report)
output.to_csv(output_path, sep="\t")
