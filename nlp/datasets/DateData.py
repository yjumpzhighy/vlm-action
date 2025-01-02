import numpy as np
import datetime
import os
import torch
import requests
import pandas as pd
import re
import sys


# With raw tokenizer
class DateData(torch.utils.data.Dataset):
    def __init__(self, MAX_LEN, N):
        np.random.seed(1)
        self.date_cn = []
        self.date_en = []
        
        # For example, date_cn:'31-04-25' -> date_en:'25/Apr/2031'
        for timestamp in np.random.randint(143835585, 2043835585, N):
            date = datetime.datetime.fromtimestamp(timestamp)
            self.date_cn.append(date.strftime("%y-%m-%d"))
            self.date_en.append(date.strftime("%d/%b/%Y"))
        # build vocabulary dict in date_en: 0~9,-,/,<GO>,<EOS>,Jan~Dec
        self.vocab = set(
            [str(i) for i in range(0, 10)] + ["-", "/", "<GO>", "<EOS>"] + 
            ['Aug', 'Nov', 'Dec', 'Sep', 'Jul', 'Apr', 'Feb', 'May', 'Mar', 'Oct', 'Jun', 'Jan'])

        # word - index mapping
        self.v2i = {v: i for i, v in enumerate(sorted(list(self.vocab)), start=1)}
        self.v2i["<PAD>"] = 0
        self.vocab.add("<PAD>")
        # index - word mapping
        self.i2v = {i:v for v, i in self.v2i.items()}

        x, y = [], []
        for cn, en in zip(self.date_cn, self.date_en):
            # convert words to correponding indexs
            x.append(
                [self.v2i["<GO>"], ] +
                [self.v2i[v] for v in cn] + [self.v2i["<EOS>"], ])
            # convert words to correponding indexs, with GO and EOS
            y.append(
                [self.v2i["<GO>"], ] + [self.v2i[v] for v in en[:3]] +
                [self.v2i[en[3:6]], ] + [self.v2i[v] for v in en[6:]] +
                [self.v2i["<EOS>"], ])

        self.x = np.full((len(x), MAX_LEN), fill_value=self.v2i["<PAD>"], 
                           dtype=np.long)
        for i, seq in enumerate(x):
            self.x[i, :len(seq)] = seq
        self.y = np.full((len(y), MAX_LEN), fill_value=self.v2i["<PAD>"], 
                           dtype=np.long)
        for i, seq in enumerate(y):
            self.y[i, :len(seq)] = seq
        
        self.x = torch.tensor(self.x, dtype=torch.int32, requires_grad=False)
        self.y = torch.tensor(self.y, dtype=torch.int32, requires_grad=False)

        self.vocab_size = len(self.vocab)
        self.START_INDEX = self.v2i["<GO>"]
        self.END_INDEX = self.v2i["<EOS>"]
        self.PAD_INDEX = self.v2i["<PAD>"]
        self.SPECIAL_INDEX = [self.START_INDEX, self.END_INDEX, self.PAD_INDEX]
        
    def __getitem__(self, idx):
        item = {
            'input_ids': self.x[idx],
            'attention_mask': self.x[idx] != self.v2i["<PAD>"],
            'gt_ids': self.y[idx],
            'gt_mask': self.y[idx] != self.v2i["<PAD>"],
        }
        return item

    def __len__(self):
        return len(self.x)    

# With pretrained distilbert tokenizer
class DateDataPretrainBertTokenizer(torch.utils.data.Dataset):
    def __init__(self, tokenizer, N=4000):
        np.random.seed(1)
        self.date_cn = []
        self.date_en = []
        # For example, date_cn:'31-04-25' , date_en:'25/Apr/2031'
        for timestamp in np.random.randint(143835585, 2043835585, N):
            date = datetime.datetime.fromtimestamp(timestamp)
            self.date_cn.append(date.strftime("%y-%m-%d"))
            self.date_en.append(date.strftime("%d/%b/%Y"))

        self.encoded_captions = tokenizer(list(self.date_cn),
                                          padding='max_length',
                                          truncation=True,
                                          max_length=20,
                                          return_tensors='pt')
        
        self.ground_truth = tokenizer(list(self.date_en),
                                          padding='max_length',
                                          truncation=True,
                                          max_length=20,
                                          return_tensors='pt')
        
        
        self.vocab_len = tokenizer.vocab_size
        
    def __getitem__(self, idx):
        # item = {
        #     key: torch.tensor(values[idx]) 
        #     for key, values in self.encoded_captions.items()
        # }
        item = {
            'input_ids': self.encoded_captions['input_ids'][idx],
            'attention_mask': self.encoded_captions['attention_mask'][idx],
            'gt_ids': self.ground_truth['input_ids'][idx],
            'gt_mask': self.ground_truth['attention_mask'][idx],
        }

        return item

    def __len__(self):
        return len(self.date_cn)
        

