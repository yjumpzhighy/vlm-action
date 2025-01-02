import numpy as np
import datetime
import os
import torch
import requests
import pandas as pd
import re
import sys
from transformers import DistilBertTokenizer




class MRPCData(object):
    NUM_SEG = 3
    PAD_ID = 0
    
    def __init__(self, data_dir="/home/zuyuanyang/Data/MRPC/", nrows=None):
        self.download_mrpc(save_dir = data_dir)
        # data['train']['is_same'][i]: data['train']['s1'][i] and data['train']['s2'][i] related or not, 
        data, self.v2i, self.i2v = self._process(data_dir, nrows)
        self.data = data
        # max token length of s1+s2 in train+test
        self.max_len = max(
            [len(s1) + len(s2) + 3 for s1, s2 in zip(
                data["train"]["s1id"] + data["test"]["s1id"], data["train"]["s2id"] + data["test"]["s2id"])])
        # token length of [s1,s2] in train
        self.xlen = np.array([
            [len(data["train"]["s1id"][i]), len(data["train"]["s2id"][i])
             ] for i in range(len(data["train"]["s1id"]))], dtype=np.int)
        # train token index of <GO>+s1+<SEP>+s2+<SEP>  
        x = [
            [self.v2i["<GO>"]] + data["train"]["s1id"][i] + [self.v2i["<SEP>"]] + data["train"]["s2id"][i] + [self.v2i["<SEP>"]]
            for i in range(len(self.xlen))
        ]
        # pad x to max_len
        self.x = pad_zero(x, self.max_len, self.PAD_ID)
        # nsp gt
        self.y = np.array(data["train"]["is_same"])

        # s1 token pos labels 0, s2 token pos labels 1, pading token pos labels 2
        self.seg = np.full(self.x.shape, fill_value=self.NUM_SEG-1, dtype=np.int32)
        for i in range(len(x)):
            si = self.xlen[i][0] + 2
            self.seg[i, :si] = 0
            si_ = si + self.xlen[i][1] + 1
            self.seg[i, si:si_] = 1
        # all index without pad/mask/sep
        self.word_ids = np.array(list(set(self.i2v.keys()).difference(
            [self.v2i[v] for v in ["<PAD>", "<MASK>", "<SEP>"]])))
        
    def _process(self, data_dir="", nrows=None):
        data = {"train": None, "test": None}
        files = os.listdir(data_dir)
        for f in files:
            df = pd.read_csv(os.path.join(data_dir, f), sep='\t', nrows=nrows)
            k = "train" if "train" in f else "test"
            data[k] = {"is_same": df.iloc[:, 0].values, "s1": df["#1 String"].values, "s2": df["#2 String"].values}
        vocab = set()
        for n in ["train", "test"]:
            for m in ["s1", "s2"]:
                for i in range(len(data[n][m])):
                    data[n][m][i] = self._text_standardize(data[n][m][i].lower())
                    cs = data[n][m][i].split(" ")
                    vocab.update(set(cs))
        v2i = {v: i for i, v in enumerate(sorted(vocab), start=1)}
        v2i["<PAD>"] = MRPCData.PAD_ID
        v2i["<MASK>"] = len(v2i)
        v2i["<SEP>"] = len(v2i)
        v2i["<GO>"] = len(v2i)
        i2v = {i: v for v, i in v2i.items()}
        for n in ["train", "test"]:
            for m in ["s1", "s2"]:
                data[n][m+"id"] = [[v2i[v] for v in c.split(" ")] for c in data[n][m]]
        return data, v2i, i2v
        
    def _text_standardize(self, text):
        text = re.sub(r'—', '-', text)
        text = re.sub(r'—', '-', text)
        text = re.sub(r'―', '-', text)
        text = re.sub(r" \d+(,\d+)?(\.\d+)? ", " <NUM> ", text)
        text = re.sub(r" \d+-+?\d*", " <NUM>-", text)
        return text.strip()   
              
    def download_mrpc(self, save_dir, proxy=None):
        train_url = 'https://mofanpy.com/static/files/MRPC/msr_paraphrase_train.txt'
        test_url = 'https://mofanpy.com/static/files/MRPC/msr_paraphrase_test.txt'
        os.makedirs(save_dir, exist_ok=True)
        proxies = {"http": proxy, "https": proxy}
        for url in [train_url, test_url]:
            raw_path = os.path.join(save_dir, url.split("/")[-1])
            if not os.path.isfile(raw_path):
                print("downloading from %s" % url)
                r = requests.get(url, proxies=proxies)
                with open(raw_path, "w", encoding="utf-8") as f:
                    f.write(r.text.replace('"', "<QUOTE>"))
                    print("completed")

    def sample(self, n):
        bi = np.random.randint(0, self.x.shape[0], size=n)
        bx, bs, bl, by = self.x[bi], self.seg[bi], self.xlen[bi], self.y[bi]
        return bx, bs, bl, by

    @property
    def num_word(self):
        return len(self.v2i)

    @property
    def mask_id(self):
        return self.v2i["<MASK>"]


class MRPCWithPretrainedTokenizer(torch.utils.data.Dataset):
    def __init__(self, MAX_LEN, data_dir="/home/zuyuanyang/Data/MRPC/", mode="train"):
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

        self.download_mrpc(save_dir = data_dir)
        # data['train']['is_same'][i]: data['train']['s1'][i] and data['train']['s2'][i] related or not, 
        data, self.v2i, self.i2v = self._process(data_dir, nrows=None)
        
        if mode=='train':
            self.context = [
                "<GO> " + data["train"]["s1"][i] + " <SEP> " + data["train"]["s2"][i] + " <SEP>"
                for i in range(len(data["train"]["s1"]))
            ]
            self.nsp = np.array(data["train"]["is_same"])
        else:
            self.context = [
                "<GO> " + data["test"]["s1"][i] + " <SEP> " + data["test"]["s2"][i] + " <SEP>"
                for i in range(len(data["test"]["s1"]))
            ]
            self.nsp = np.array(data["train"]["is_same"])
        
        self.encoded_context = self.tokenizer(list(self.context),
                                          padding='max_length',
                                          truncation=True,
                                          max_length=MAX_LEN,
                                          return_tensors='pt')

        self.vocab_len = self.tokenizer.vocab_size #30522
 
    def _process(self, data_dir="", nrows=None):
        data = {"train": None, "test": None}
        files = os.listdir(data_dir)
        for f in files:
            df = pd.read_csv(os.path.join(data_dir, f), sep='\t', nrows=nrows)
            k = "train" if "train" in f else "test"
            data[k] = {"is_same": df.iloc[:, 0].values, "s1": df["#1 String"].values, "s2": df["#2 String"].values}
        vocab = set()
        for n in ["train", "test"]:
            for m in ["s1", "s2"]:
                for i in range(len(data[n][m])):
                    data[n][m][i] = self._text_standardize(data[n][m][i].lower())
                    cs = data[n][m][i].split(" ")
                    vocab.update(set(cs))
        v2i = {v: i for i, v in enumerate(sorted(vocab), start=1)}
        v2i["<PAD>"] = MRPCData.PAD_ID
        v2i["<MASK>"] = len(v2i)
        v2i["<SEP>"] = len(v2i)
        v2i["<GO>"] = len(v2i)
        i2v = {i: v for v, i in v2i.items()}
        for n in ["train", "test"]:
            for m in ["s1", "s2"]:
                data[n][m+"id"] = [[v2i[v] for v in c.split(" ")] for c in data[n][m]]
        return data, v2i, i2v
        
    def _text_standardize(self, text):
        text = re.sub(r'—', '-', text)
        text = re.sub(r'—', '-', text)
        text = re.sub(r'―', '-', text)
        text = re.sub(r" \d+(,\d+)?(\.\d+)? ", " <NUM> ", text)
        text = re.sub(r" \d+-+?\d*", " <NUM>-", text)
        return text.strip()   
              
    def download_mrpc(self, save_dir, proxy=None):
        train_url = 'https://mofanpy.com/static/files/MRPC/msr_paraphrase_train.txt'
        test_url = 'https://mofanpy.com/static/files/MRPC/msr_paraphrase_test.txt'
        os.makedirs(save_dir, exist_ok=True)
        proxies = {"http": proxy, "https": proxy}
        for url in [train_url, test_url]:
            raw_path = os.path.join(save_dir, url.split("/")[-1])
            if not os.path.isfile(raw_path):
                print("downloading from %s" % url)
                r = requests.get(url, proxies=proxies)
                with open(raw_path, "w", encoding="utf-8") as f:
                    f.write(r.text.replace('"', "<QUOTE>"))
                    print("completed")
             
    def __getitem__(self, idx):
        item = {
            key: torch.tensor(values[idx]) 
            for key, values in self.encoded_context.items()
        }
        
        item['nsp_y'] = self.nsp[idx]
        return item
    
    def __len__(self):
        return len(self.context)
    