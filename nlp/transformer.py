import torch
import torch.nn as nn
import numpy as np
import os
import time
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer

from datasets.DateData import DateData, DateDataPretrainBertTokenizer
from utils import set_soft_gpu



class MultiHead(nn.Module):
    def __init__(self, n_head, model_dim, drop_rate):
        super().__init__()
        self.hidden_dim = model_dim
        self.head_dim = model_dim // n_head
        self.n_head = n_head
        
        # expand ratio?
        self.wq = nn.Linear(model_dim, model_dim)
        self.wk = nn.Linear(model_dim, model_dim)
        self.wv = nn.Linear(model_dim, model_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.proj = nn.Linear(model_dim, model_dim)
        self.drop = nn.Dropout(drop_rate)

    def split_heads(self, x):
        # [batch_size, max_len, model_dim] -> [batch_size, max_len, n_head, head_dim]
        x = x.view(x.shape[0], x.shape[1], self.n_head, self.head_dim)
        # [batch_size, n_head, max_len, head_dim]
        x = x.permute(0, 2, 1, 3)
        return x

    def forward(self, q, k, v, mask, training):
        #[batch_size, max_len, model_dim]
        _q, _k, _v = self.wq(q), self.wk(k), self.wv(v)
        #[batch_size, n_head, max_len, head_dim]
        _q, _k, _v = self.split_heads(_q), self.split_heads(_k), self.split_heads(_v)
        #[batch_size, n_head, max_len, max_len]
        score = torch.matmul(_q, _k.transpose(-2, -1)) / (torch.sqrt(torch.tensor(_k.shape[-1]).float()) + 1e-8)

        if mask is not None:
            score += mask * -1e9  #block masked token attention

        #[batch_size, n_head, max_len, max_len]
        score = self.softmax(score)
        #[batch_size, n_head, max_len, head_dim]
        context = torch.matmul(score, _v)
        #[batch_size, max_len, head_dim*n_heads]
        context = context.permute(0,2,1,3)
        context = context.reshape(context.shape[0], context.shape[1], -1)
        #[batch_size, max_len, model_dim]
        context = self.proj(context)
        context = self.drop(context)
        return context

class PositionWiseFFN(nn.Module):
    def __init__(self, model_dim, expand_ratio=4.0):
        super().__init__()
        hidden_dim = int(model_dim * expand_ratio)
        self.f1 = nn.Linear(model_dim, hidden_dim)
        self.norm1 = nn.BatchNorm1d(hidden_dim)
        self.act1 = nn.ReLU()
        self.f2 = nn.Linear(hidden_dim, model_dim)
    def forward(self, x):
        x = self.f2(self.act1(self.f1(x)))
        return x
           
class PositionEmbedding(nn.Module):
    def __init__(self, max_len, model_dim, n_vocab):
        super().__init__()
        
        pos = np.arange(max_len)[:, None]
        pe = pos / np.power(10000, 2. * np.arange(model_dim)[None,:]/model_dim)
        # sin(i/10000^2k/model_dim)
        pe[:, 0::2] = np.sin(pe[:, 0::2])
        pe[:, 1::2] = np.cos(pe[:, 1::2])
        pe = pe[None,:,:] 
        self.pe = torch.tensor(pe, dtype=torch.float32, requires_grad=False).cuda() #[1, max_len, model_dim]
        self.embeddings = nn.Embedding(n_vocab, model_dim) #[n_vocab, model_dim]
        self.embeddings.weight.data.normal_(mean=0, std=0.01)
        #self.embeddings.weight.data.uniform_(0, 0.01)
    def forward(self, x):
        # x:[batch_size, max_len], sequence are padded to max_len tokens
        # [batch_size, max_len, model_dim], each token gets its embedding vector
        
        x_embedding = self.embeddings(x) + self.pe
        return x_embedding
        
class EncoderLayer(nn.Module):
    def __init__(self, n_head, model_dim, drop_rate):
        super().__init__()
        self.attn = MultiHead(n_head, model_dim, drop_rate)
        self.norm1 = nn.LayerNorm(model_dim)
        self.ffn = PositionWiseFFN(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.drop = nn.Dropout(drop_rate)
    def forward(self, x, training=True, mask=None):
        #[batch_size, max_len, model_dim]
        attn = self.attn(x, x, x, mask, training)
        attn = self.norm1(attn + x)
        ffn = self.drop(self.ffn(attn))
        out = self.norm2(ffn + attn)
        return out
               
class Encoder(nn.Module):
    def __init__(self, n_head, model_dim, drop_rate, n_layer):
        super().__init__()
        self.layers = nn.ModuleList(
            [EncoderLayer(n_head, model_dim, drop_rate) for _ in range(n_layer)])
    def forward(self, x, training, mask):
        for layer in self.layers:
            x = layer(x, training, mask)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, n_head, model_dim, drop_rate):
        super().__init__()
        self.attn1 = MultiHead(n_head, model_dim, drop_rate)
        self.attn2 = MultiHead(n_head, model_dim, drop_rate)
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.norm3 = nn.LayerNorm(model_dim)
        self.ffn = PositionWiseFFN(model_dim)
        self.drop = nn.Dropout(drop_rate)
    def forward(self, x1, x2, training, x1_look_ahead_mask, x2_pad_mask):
        attn = self.attn1(x1, x1, x1, x1_look_ahead_mask, training)
        out1 = self.norm1(attn + x1)
        # encoder-decoder cross attention. 
        # calculate the cross attention beween x_emd and y_emd, then get info from x_emd
        attn = self.attn2(out1, x2, x2, x2_pad_mask, training)
        out2 = self.norm2(attn + out1)
        ffn = self.drop(self.ffn(out2))
        out3 = self.norm3(ffn + out2)
        return out3

class Decoder(nn.Module):
    def __init__(self, n_head, model_dim, drop_rate, n_layer):
        super().__init__()
        self.layers = nn.ModuleList(
            [DecoderLayer(n_head, model_dim, drop_rate) for _ in range(n_layer)])
    def forward(self, x1, x2, training, x1_look_ahead_mask, x2_pad_mask):
        for layer in self.layers:
            x1 = layer(x1, x2, training, x1_look_ahead_mask, x2_pad_mask)
        return x1
    
class Transformer(nn.Module):
    N_LAYER = 3
    N_HEAD = 4
    DROP_RATE = 0.1
    MODEL_DIM = 32
    
    def __init__(self, max_len, n_vocab, padding_idx=0):
        super().__init__()
        
        self.max_len = max_len
        self.padding_idx = torch.tensor(padding_idx, dtype=torch.int).cuda()
        self.embed = PositionEmbedding(max_len, self.MODEL_DIM, n_vocab)
        self.encoder = Encoder(self.N_HEAD, self.MODEL_DIM, self.DROP_RATE, self.N_LAYER)
        self.decoder = Decoder(self.N_HEAD, self.MODEL_DIM, self.DROP_RATE, self.N_LAYER)
        self.out = nn.Linear(self.MODEL_DIM, n_vocab)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, batch, training=True):
        # x:[batch_size, max_len], y:[batch_size, max_len]
  
        pad_mask = ~batch['attention_mask'][:, None, None, :] #[batch_size, 1, 1, max_len]
        look_ahead_mask = self._look_ahead_mask(batch['gt_ids']) #[batch_size, 1, max_len, max_len]
        
        x_embed = self.embed(batch['input_ids'])  #[batch_size, max_len, model_dim]
        y_embed = self.embed(batch['gt_ids'])  #[batch_size, max_len, model_dim]

        encoded = self.encoder(x_embed, training, mask=pad_mask)
        decoded = self.decoder(y_embed, encoded, training,
                               look_ahead_mask, pad_mask)

        out = self.out(decoded) #[batch_size, max_len, n_vocab]
        return out
     
    def _look_ahead_mask(self, seqs):
        mask = torch.triu(torch.ones((self.max_len, self.max_len),dtype=torch.long),1).cuda()
        mask = torch.where((seqs == self.padding_idx)[:,None,None,:],
                        1, mask[None,None,:,:])
        return mask
    
    def translate(self, src, v2i, i2v, pad_id):
        src_pad = pad_zero_keras(src, self.max_len, pad_id)
        tgt = pad_zero_keras(torch.tensor([[v2i["<GO>"],] for _ in range(len(src))]),
                       self.max_len+1, pad_id)
        
        src_pad, tgt = torch.tensor(src_pad,dtype=torch.int), torch.tensor(tgt,dtype=torch.int)
        encoded = self.encoder(self.embed(src_pad), False, 
                               mask=self._pad_mask(src_pad))
        
        # decode results
        tgti = 0
        while True:
            y = tgt[:, :-1]
            y_embed = self.embed(y)
            decoded = self.decoder(y_embed, encoded, False, 
                                   self._look_ahead_mask(y), self._pad_mask(src_pad))
            logits = self.out(decoded)[:, tgti, :]
            probs = self.softmax(logits)
            idx = torch.argmax(probs, axis = 1)
            tgti += 1
            tgt[:, tgti] = idx
            if tgti >= self.max_len:
                break
        
        tgt = tgt.detach().numpy()
        return ["".join([i2v[i] for i in tgt[j, 1:tgti]]) for j in range(len(src))]        
           

# Use pretrained distilbert         
class PretrainedBertEncoder(nn.Module):
    def __init__(self, model_name='distilbert-base-uncased', pretrained=True):
        super().__init__()

        if pretrained:
            self.model = DistilBertModel.from_pretrained(model_name)
        else:
            self.model = DistilBertModel(config=DistilBertConfig())
       
        self.output_states_dim = 768
        
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

class PretrainedBertHead(nn.Module):
    def __init__(self, model_dim, n_vocab):
        super().__init__()
        self.project = nn.Linear(model_dim, n_vocab)
    def forward(self, x):
        # [batch_size,max_len,n_vocab]
        logits = self.project(x)
        return logits

class PretrainedBertTransformer(nn.Module): 
    def __init__(self, n_vocab):
        super().__init__()
        self.encoder = PretrainedBertEncoder()
        self.head = PretrainedBertHead(self.encoder.output_states_dim, n_vocab)
    def forward(self, batch):
        #[batch,max_len,model_dim]
        embeddings = self.encoder(batch['input_ids'], batch['attention_mask'])
        #[batch,max_len,n_vocab]
        logits = self.head(embeddings)
        return logits
        
              
if __name__ == "__main__":
    MAX_LEN = 20
    BATCH_SIZE = 64
    EPOCH = 15
    FINETUNE = False
    set_soft_gpu(True)
    
    
    if (not FINETUNE):
        db = DateData(MAX_LEN, 4000)
        dataloader = torch.utils.data.DataLoader(
            db,
            batch_size=BATCH_SIZE,
            num_workers=16,
            shuffle=True)
        
        model = Transformer(MAX_LEN, db.vocab_size, db.PAD_INDEX).to(torch.device("cuda"))
        criterior = CrossEntropyLoss(reduction='none')
        #optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.001)
        optimizer = optim.Adam(model.parameters(), lr=0.002)
        model.train()

        for epoch in range(20):
            epoch_loss = 0.0
            pred = None
            goal = None
            for batch in dataloader:
                batch = {k: v.cuda() for k,v in batch.items()}
                # [batch_size, max_len, vocab_len]
                logits = model(batch)
                
                loss = torch.mean(
                    criterior(logits[:,:-1,:].permute(0,2,1), 
                              batch['gt_ids'][:,1:].long())[batch['gt_mask'][:, 1:]])
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss = loss.item()
                pred = logits[0].detach().cpu().numpy().argmax(axis=1)
                goal = batch['gt_ids'][0].detach().cpu().numpy()

            print(
                "| epoch {}, loss = {}".format(epoch, epoch_loss),
                "| inference:", "".join([db.i2v[i] for i in pred if i not in db.SPECIAL_INDEX]),
                "| goal:", "".join([db.i2v[i] for i in goal if i not in db.SPECIAL_INDEX]),
            )
       
    else:    
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased") 
        dataset = DateDataPretrainBertTokenizer(tokenizer, 4000)    
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            num_workers=16,
            shuffle=True)
        model = PretrainedBertTransformer(dataset.vocab_len).cuda()

        params = [
            {"params": model.encoder.parameters(), "lr": 1e-5},
            {"params": model.head.parameters(), "lr": 1e-3},
            ]
        optimizer = torch.optim.AdamW(params, weight_decay = 0.)
        criterior = CrossEntropyLoss(reduction='none')
        model.train()

        for epoch in range(20):
            epoch_loss = 0.0
            pred = None
            goal = None
            for batch in dataloader:
                batch = {k: v.cuda() for k,v in batch.items()}
                # [batch_size, max_len, vocab_len]
                logits = model(batch)
                
                loss = torch.mean(
                    criterior(logits[:,:-1,:].permute(0,2,1), 
                              batch['gt_ids'][:,1:].long())[batch['gt_mask'][:, 1:]])
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss = loss.item()
                pred = logits[0].detach().cpu().numpy().argmax(axis=1)
                goal = batch['gt_ids'][0].detach().cpu().numpy()

            print(
                "| epoch {}, loss = {}".format(epoch, epoch_loss),
                "| inference:", tokenizer.decode(pred.tolist(), skip_special_tokens=True),
                "| goal:", tokenizer.decode(goal.tolist(), skip_special_tokens=True),
            )
            
        
            
            
        
        
        

    
    
    