import numpy as np
from tqdm import tqdm

def mask_bertlike(input_seq):
    
    seq_len = len([x for x in input_seq if x != 0]) # sequence length without padding
    num_mask_token = int(seq_len/100*15) # 15% of all token in sequence
    mask_indices = np.random.choice(range(1,seq_len-1), num_mask_token, replace=False)
    
    for mask_index in mask_indices:
        input_seq[mask_index] = 104
    
    return input_seq

def create_pretrain_examples(tokenizer, max_len, pretrain_text, pretrain_method):
    
    
    inputs = []
    masks = []
    trues = []
    
    for line in pretrain_text:
        try:
            line = line.decode("utf-8")
        except:
            continue
        
        tokenizer_output = tokenizer.encode_plus(line, max_length=max_len, padding="max_length", truncation=True)
        
        
        if pretrain_method == "bert":
            
            
            masked_input = mask_bertlike(tokenizer_output["input_ids"])
            
            
        if len([ x for x in masked_input if x != 0]) < 15:
            continue
        
        i = 0
        y = []
        true_token = tokenizer.encode(line)
        for token in masked_input:
            
            if token == 104:
                y.append(true_token[i])
            else:
                y.append(-100)
            i+=1
            
        inputs.append(masked_input)    
        masks.append(tokenizer_output["attention_mask"])
        trues.append(y)
        
        
    return inputs, masks, trues


def pack_batches(inputs, masks, trues, bsize):
    
    
    num = 0
    c = 0
    
    batches = []
    
    ins = []
    msk = []
    tru = []
    
    while num < len(trues) and num < len(inputs) and num < len(masks):
        
        ins.append(inputs[num])
        msk.append(masks[num])
        tru.append(trues[num])
        
        num+=1
        c+=1
        
        if c == bsize:
            
            batches.append([ins,msk,tru])
            
            ins = []
            msk = []
            tru = []
            
            c=0
            
            
    return batches


def create_batches(tokenizer, input_file, bsize, pretrain_method, max_len):


    pretrain_text = open(input_file,"rb")
    inputs, masks, trues = create_pretrain_examples(tokenizer, max_len, pretrain_text, pretrain_method)
    batches = pack_batches(inputs, masks, trues, bsize)
    
    return batches
    
    
    
