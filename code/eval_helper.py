import torch
from pretrain_helper import create_batches
from tqdm import tqdm

def eval_on_batches(batches, model, device):
    
    model.eval()
        
    with torch.no_grad():
            
        self_eval_loss = 0
        for batch in batches:
                
            x = torch.tensor(batch[0]).to(device)
            m = torch.tensor(batch[1]).to(device)
            y = torch.tensor(batch[2]).to(device)
            outputs = model(x, token_type_ids=None, attention_mask=m, labels=y)
            self_eval_loss += outputs[0].sum().item()
            
        self_eval_loss = self_eval_loss/len(batches)
    
    return self_eval_loss


def create_mlm_evalsets(batches, tokenizer, maxlen, mlm_evaldata, pretrain_method, pretrain_bsize):

    eval_sets = []
    for evalcase in tqdm(mlm_evaldata):

        if evalcase["name"] != "self":

            eval_batches = create_batches(tokenizer,  
                                          evalcase["file"], 
                                          pretrain_bsize, 
                                          pretrain_method,
                                          maxlen)

            eval_sets.append({"name":evalcase["name"], "batches":eval_batches})

        else:

            eval_sets.append({"name":evalcase["name"], "batches":batches})

    return eval_sets

