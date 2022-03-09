from pretrain_helper import pack_batches

from transformers import BertForSequenceClassification
from transformers import get_cosine_schedule_with_warmup
from transformers import AdamW

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from tqdm import tqdm

import pandas as pd
import numpy as np
import torch
import re

def create_cls_examples(tokenizer, max_len, X, Y):
    
    
    inputs = []
    masks = []
    trues = []
    
    i = 0
    for text in tqdm(X):
        
        try:
            tokenizer_output = tokenizer.encode_plus(text, max_length=max_len, padding="max_length", truncation=True)
        except:
            continue
        inputs.append(tokenizer_output["input_ids"])
        masks.append(tokenizer_output["attention_mask"])
        trues.append(Y[i])
        i+=1
    print(len(inputs))
    return inputs, masks, trues

def create_cls_batches(tokenizer, maxlen, filepath, bsize):
    
    
    data = pd.read_csv(filepath, sep="\t", index_col=0)
    data.columns = ["X","Y","SOURCE"]
    
    X_train, X_test, y_train, y_test = train_test_split(list(data["X"]), list(data["Y"]), test_size=0.1, random_state=42)
    
    inputs_train, masks_train, trues_train = create_cls_examples(tokenizer, 
                                                                 maxlen, 
                                                                 X_train, 
                                                                 y_train)
    
    inputs_test, masks_test, trues_test = create_cls_examples(tokenizer, 
                                                              maxlen, 
                                                              X_test, 
                                                              y_test)
    
    
    cls_train_batches = pack_batches(inputs_train, masks_train, trues_train, bsize)
    cls_eval_batches = pack_batches(inputs_test, masks_test, trues_test, bsize)
    
    
    return cls_train_batches, cls_eval_batches


def load_checkpoint_cls(model_name, path):
    
    model = BertForSequenceClassification.from_pretrained(model_name)
    state_dict = torch.load(path)

    mkeys = list(model.state_dict().keys())
    skeys = list(state_dict.keys())


    for k in skeys:

        if k not in mkeys:
            del state_dict[k]

    mkeys = list(model.state_dict().keys())
    skeys = list(state_dict.keys())

    for k in mkeys:

        if k not in skeys:
            state_dict[k] = model.state_dict()[k]


    model.load_state_dict(state_dict)
    
    
    return model


def classify(model, train_batches, test_batches, epochs, lr, device):

    scores = []
    cls_optimizer = AdamW(model.parameters(),lr = lr)
    cls_scheduler = get_cosine_schedule_with_warmup(cls_optimizer, 
                                                    num_warmup_steps = 0, 
                                                    num_training_steps = epochs*len(train_batches))
    model.cuda()
    for epoch in tqdm(list(range(int(epochs)))):
        
        model.train()
        i=0
        for batch in train_batches:
            
            x = torch.tensor(batch[0]).to(device)
            m = torch.tensor(batch[1]).to(device)
            y = torch.tensor(batch[2]).to(device)

            model.zero_grad()

            outputs = model(x, token_type_ids=None, attention_mask=m, labels=y)
        
            loss = outputs[0].sum()
            print("CLS Loss: "+str(loss.item()))

            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.9)

            cls_optimizer.step()
            cls_scheduler.step()
            i+=1         
            if i % 30 == 0:
                model.eval()
        
        
                rep_t = []
                rep_p = []
        
                for batch in test_batches:    
            
                    x = torch.tensor(batch[0]).to(device)
                    m = torch.tensor(batch[1]).to(device)
                    y = torch.tensor(batch[2]).to(device)


                    with torch.no_grad():
                        outputs = model(x, token_type_ids=None, attention_mask=m)
                        logits = outputs[0]
                        logits = logits.detach().cpu().numpy()
                        label_ids = y.to('cpu').numpy()

                        trues = label_ids.flatten()
                        pred = np.argmax(logits, axis=1)

                        trues[trues == -100] = 0
                        pred[pred == -100] = 0

                        rep_t.append(trues)
                        rep_p.append(pred)

                rep = classification_report(np.stack(rep_t).flatten(), np.stack(rep_p).flatten())
                print(rep)
                f1_epoch = rep.split("\n")[-2]
                f1_epoch = float(re.sub("\s+"," ",f1_epoch).split(" ")[-2])
                print(f1_epoch)
                scores.append(f1_epoch)
    
    
            
    return scores















