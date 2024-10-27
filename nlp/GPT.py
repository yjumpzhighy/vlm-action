import torch
import torch.nn as nn
import tensorflow as tf
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss
import time
import functools

from transformer import Encoder, PositionEmbedding
from datasets.DateData import DateData
from utils import GeneratorWithBeamSearch

class GPT(nn.Module):
    # GPT use transformer encoder (no following decoder), but use look ahead mask.
    
    # Due to the look ahead mask, GPT model usually perform worse in first-half sentence,
    # as it missed lots of imformation.
    # From simple experiments, GPT performance worse than transformer.
    
    MODEL_DIM = 32
    N_LAYER = 4
    N_HEADS = 4
    DROP_RATE = 0.2

    def __init__(self, max_len, n_vocab, padding_idx, eos_indx):
        super().__init__()
        self.padding_idx = padding_idx
        self.eos_idx = eos_indx
        self.n_vocab = n_vocab
        self.max_len = max_len
        
        # self.word_emb = nn.Embedding(n_vocab, self.MODEL_DIM)
        # self.word_emb.weight.data.normal_(mean=0, std=0.01)
        # self.segment_emb = nn.Embedding(max_seg, self.MODEL_DIM)
        # self.segment_emb.weight.data.normal_(mean=0, std=0.01)
        # self.register_parameter("position_emb", 
        #     nn.Parameter(torch.rand((1, max_len, self.MODEL_DIM),dtype=torch.float32)))
        # self.task_mlm = nn.Linear(self.MODEL_DIM, n_vocab)
        # self.task_nsp = nn.Linear(self.MODEL_DIM * self.max_len, 2)
       
        self.input_emb = PositionEmbedding(max_len, self.MODEL_DIM, n_vocab)
        self.encoder = Encoder(self.N_HEADS, self.MODEL_DIM, self.DROP_RATE, self.N_LAYER)
        self.out = nn.Linear(self.MODEL_DIM, n_vocab)
        
        
    def forward(self, batch, training=True):     
            pad_mask = ~batch['attention_mask'][:, None, None, :] #[batch_size, 1, 1, max_len]
            look_ahead_mask = self._look_ahead_mask(batch['input_ids']) #[batch_size, 1, max_len, max_len]

            x_embed = self.input_emb(batch['input_ids'])  #[batch_size, max_len, model_dim]

            encoded = self.encoder(x_embed, training, mask=look_ahead_mask)
            
            out = self.out(encoded) #[batch_size, max_len, n_vocab]
    
            return out
        
    def _look_ahead_mask(self, seqs):
        # #lower triangular part matrix
        # mask = 1 - tf.linalg.band_part(
        #        tf.ones((self.max_len, self.max_len)), -1, 0)
        # mask = tf.where((seqs == self.padding_idx)[:,None,None,:],
        #                 1, mask[None,None,:,:])
        # #TODO, use torch tensor directly
        # mask = torch.tensor(mask.numpy())
        
        mask = torch.triu(torch.ones((self.max_len, self.max_len),dtype=torch.long),1).cuda()
        mask = torch.where((seqs == self.padding_idx)[:,None,None,:],
                        1, mask[None,None,:,:])
        return mask

    
    def decode(self, seqs):
        #seqs: [batch*beam, cur_len]
        batch_size = seqs.shape[0]
        beam_size = int(seqs.shape[0] / batch_size)
    
        x_embed = self.input_emb(batch['input_ids'])  #[batch_size, max_len, model_dim]
        encoded = self.encoder(x_embed, False, mask=None)
        out = self.out(encoded) #[batch_size, max_len, n_vocab]
        
        return out


if __name__ == "__main__":
    MAX_LEN = 20
    BATCH_SIZE = 64

    dataset = DateData(MAX_LEN, 4000)
    dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            num_workers=16,
            shuffle=True)

    model = GPT(
        max_len=MAX_LEN, n_vocab=dataset.vocab_size, padding_idx=dataset.PAD_INDEX,
        eos_indx=dataset.END_INDEX).cuda()
    criterior = CrossEntropyLoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=0.002, weight_decay=0.0001)
    model.train()
    
    for epoch in range(20):
            epoch_loss = 0.0
            pred = None
            goal = None
            for batch in dataloader:
                batch = {k: v.cuda() for k,v in batch.items()}
                # [batch_size, max_len, vocab_len]
                logits = model(batch, training=True)
                
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
                "| inference:", "".join([dataset.i2v[i] for i in pred if i not in dataset.SPECIAL_INDEX]),
                "| goal:", "".join([dataset.i2v[i] for i in goal if i not in dataset.SPECIAL_INDEX]),
            )
    
    
    