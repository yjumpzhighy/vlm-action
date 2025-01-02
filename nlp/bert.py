import torch
import torch.nn as nn
import tensorflow as tf

from transformer import Encoder
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss
import time
import numpy as np
from tqdm import tqdm

import itertools
from utils import AvgMeter
from datasets.MRPC import MRPCData, MRPCWithPretrainedTokenizer
from GPT import GPT

class BERT(GPT):
    def __init__(self, model_dim, max_len, n_layer, n_head, n_vocab,
                 max_seg, drop_rate, padding_idx):
        super().__init__(model_dim, max_len, n_layer, n_head, n_vocab,
                 max_seg, drop_rate, padding_idx)

    def forward(self, seqs, segs):
        return super().forward(seqs, segs, training=True)
        
    def _look_ahead_mask(self, seqs):
        # In bert, able to look front and rear
        return (seqs == self.padding_idx)[:,None,None,:]

def _get_loss_mask(len_arange, seqs, pad_id=0):
    rand_id = np.random.choice(
        len_arange, size=max(2,int(0.15*len(len_arange))),
        replace=False)
    loss_mask = np.full_like(seqs, pad_id, dtype=np.bool)
    loss_mask[rand_id] = True
    return loss_mask[None, :], rand_id

def do_mask(seq, len_arange, pad_id, mask_id):
    loss_mask, rand_id = _get_loss_mask(len_arange, seq, pad_id)
    seq[rand_id] = mask_id
    return loss_mask

def do_replace(seq, len_arange, pad_id, word_ids):
    loss_mask, rand_id = _get_loss_mask(len_arange, seq, pad_id)
    seq[rand_id] = np.random.choice(word_ids, size=len(rand_id))
    return loss_mask

def do_nothing(seq, len_arange, pad_id):
    loss_mask, _ = _get_loss_mask(len_arange, seq, pad_id)
    return loss_mask

def random_mask_or_replace(db, arange):
    seqs, segs, xlen, nsp_labels = db.sample(16)
    seqs_ = seqs.copy()
    p = np.random.random()
    
    if p<0.7: #do mask
        loss_mask = np.concatenate(
            [do_mask(seqs[i],
                     np.concatenate((arange[:xlen[i,0]], arange[xlen[i,0]+1:xlen[i].sum()+1])),
                     db.PAD_ID,
                     db.v2i["<MASK>"]) for i in range(len(seqs))], axis=0)
    elif p<0.85: #do nothing
        loss_mask = np.concatenate(
            [do_nothing(seqs[i],
                       np.concatenate((arange[:xlen[i,0]], arange[xlen[i,0]+1:xlen[i].sum()+1])),
                       db.PAD_ID) for i in range(len(seqs))], axis=0)
    else: #do replace
        loss_mask = np.concatenate(
            [do_replace(seqs[i],
                        np.concatenate((arange[:xlen[i,0]], arange[xlen[i,0]+1:xlen[i].sum()+1])),
                        db.PAD_ID,
                        db.word_ids) for i in range(len(seqs))], axis=0)
    return seqs, segs, seqs_, loss_mask, xlen, nsp_labels


class PretrainedEncoder(nn.Module):
    def __init__(self,
                 model_name='distilbert-base-uncased'):
        super().__init__()
        self.model = DistilBertModel.from_pretrained(model_name)
       
        #fine tune
        for p in self.model.parameters():
            p.requires_grad = True
            
        self.target_token_idx = 0
        
    def forward(self,input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        #[batch,max_len,model_dim]
        return last_hidden_state
        #return last_hidden_state[:,self.target_token_idx,:]
  
class PretrainedMlmHeads(nn.Module):
    def __init__(self, model_dim, n_vocab):
        super().__init__()
        self.task_mlm = nn.Linear(model_dim, n_vocab)
    def forward(self, x):
        # [batch_size,max_len,n_vocab]
        mlm_logits = self.task_mlm(x)
        return mlm_logits

class PretrainedNspHeads(nn.Module):
    def __init__(self, model_dim, max_len):
        super().__init__()
        self.task_nsp = nn.Linear(model_dim * max_len, 2)
    def forward(self, x):
        # [batch_size,2]
        nsp_logits = self.task_nsp(x.view(x.shape[0],-1))
        return nsp_logits

class Pretrained(nn.Module):
    def __init__(self, model_dim, n_vocab, max_len):
        super().__init__()
        self.encoder = PretrainedEncoder()
        self.mlm_head = PretrainedMlmHeads(model_dim, n_vocab)
        self.nsp_head = PretrainedNspHeads(model_dim, max_len)
    def forward(self, input_ids, attention_mask):
        #[batch,max_len,model_dim]
        embeddings = self.encoder(input_ids, attention_mask)
        #[batch,max_len,n_vocab]
        mlm_logits = self.mlm_head(embeddings)
        #[batch,2]
        nsp_logits = self.nsp_head(embeddings)
        return mlm_logits, nsp_logits
                
def train():   
    MODEL_DIM = 256
    N_LAYER = 4
    N_HEADS = 4
    LEARNING_RATE = 1e-3
    PAD_ID = 0
    STEP = 10000

    db = MRPCData("/home/zuyuanyang/Data/MRPC/", 2000)
    print("num word: ", db.num_word)

    model = BERT(
        model_dim=MODEL_DIM, max_len=db.max_len - 1, n_layer=N_LAYER, n_head=N_HEADS,
        n_vocab=db.num_word, max_seg=db.NUM_SEG, drop_rate=0.2, 
        padding_idx=db.PAD_ID)
    criterior = CrossEntropyLoss(reduction='none')
    #optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.001)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0001)
    
    for step in range(STEP):
        t0 = time.time()
        # seqs: train token index of <GO>+s1+<SEP>+s2+<SEP>
        # segs: s1 token pos labels 0, s2 token pos labels 1, pading token pos labels 2
        # xlen: token length of [s1,s2] in train
        # nsp_labels: s1 and s2 is_same or not
        seqs, segs, seqs_original, loss_mask, xlen, nsp_labels = random_mask_or_replace(db, np.arange(0,db.max_len,16))
        
        seqs, segs, seqs_original, loss_mask, xlen, nsp_labels = torch.tensor(seqs), torch.tensor(segs),\
                                                                 torch.tensor(seqs_original), torch.tensor(loss_mask),\
                                                                 torch.tensor(xlen),torch.tensor(nsp_labels)
        mlm_logits, nsp_logits = model(seqs[:, :-1], segs[:, :-1])
        pred_loss = torch.mean(criterior(mlm_logits.permute(0,2,1), seqs_original[:,1:].long())[loss_mask[:,1:]])
        nsp_loss = torch.mean(criterior(nsp_logits, nsp_labels))
        loss = pred_loss + 0.2 * nsp_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 100 == 0 or step==STEP-1:
            pred = mlm_logits[0].detach().numpy().argmax(axis=1)
            t1 = time.time()
            print(
                "\n\nstep: ", step,
                "| time: %.2f" % (t1 - t0),
                "| loss: %.3f" % loss.detach().numpy(),
                "\n| tgt: ", " ".join([db.i2v[i] for i in seqs.detach().numpy()[0][:xlen[0].sum()+1]]),
                "\n| prd: ", " ".join([db.i2v[i] for i in pred[:xlen[0].sum()+1]]),
                )
            t0 = t1
    
        
    # test masked language model
    seqs, segs, xlen, nsp_labels = db.sample(5)
    mlm_logits, nsp_logits = model(seqs[:, :-1], segs[:, :-1])
    mlm_probs = nn.functional.softmax(mlm_logits, -1)
    mlm_idx = torch.argmax(mlm_probs, axis=-1).numpy()
    
    seqs = seqs.numpy()
    
    mlm_label = ["".join([(db.i2v[i] + " ") for i in seqs[j]]) for j in range(len(seqs))]
    mlm_rslt = ["".join([(db.i2v[i] + " ") for i in mlm_idx[j]]) for j in range(len(mlm_idx))]
    for sample_idx in range(5):
        print("=>", mlm_label[sample_idx]) 
        print("=>", mlm_rslt[sample_idx])
        print("-----------------")
    
def train_finetune():   
    MODEL_DIM = 768
    N_LAYER = 4
    N_HEADS = 4
    LEARNING_RATE = 1e-4
    PAD_ID = 0
    STEP = 10000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    db = MRPCWithPretrainedTokenizer()
    dataloader = torch.utils.data.DataLoader(
        db,
        batch_size=32,
        num_workers=4,
        shuffle=True,
    )

    model = Pretrained(
        model_dim=MODEL_DIM, n_vocab=db.vocab_len, max_len=db.max_len-1).to(device)
    criterior = CrossEntropyLoss(reduction='none')
    params = [
        {"params": model.encoder.parameters(), "lr": 1e-5},
        {"params": itertools.chain(
            model.mlm_head.parameters(), model.nsp_head.parameters()
        ), "lr": 1e-3, "weight_decay": 1e-3}
    ]
    optimizer = torch.optim.AdamW(params, weight_decay = 0.)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for epoch in range(30):
        model.train()
        loss_meter = AvgMeter()
        
        mlm_pred = None
        for batch in tqdm(dataloader, desc="train"):
            batch = {k: v.to(device) for k,v in batch.items()}
            
            mlm_logits, nsp_logits = model(batch['input_ids'][:, :-1], batch['attention_mask'][:, :-1])
            mlm_loss = torch.mean(criterior(mlm_logits.permute(0,2,1), batch['input_ids'][:,1:].long())[batch['attention_mask'][:, 1:]])
            nsp_loss = torch.mean(criterior(nsp_logits, batch['nsp_y']))
            loss = mlm_loss + 0.2 * nsp_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            count = batch['input_ids'].shape[0]
            loss_meter.update(loss.item(), count)
            mlm_pred = mlm_logits

        pred = mlm_pred[0].detach().cpu().numpy().argmax(axis=1)
        out = db.tokenizer.decode(pred)
        src = db.tokenizer.decode(batch['input_ids'][0])
        print('-------------------')
        print("->", src)
        print("->", out)
        print("loss=", loss_meter.avg)
        torch.save(model.state_dict(), "best_bert_1.pt")    
   
if __name__ == "__main__":
    #train()
    train_finetune()