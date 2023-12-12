import torch
import numpy as np
from datasets import load_dataset

class MlmBaseDataset(torch.utils.data.Dataset):   
    def __init__(self,tokenizer,file_name=None,random=False,mode='train'):
        self.random = random
        self.tokenizer = tokenizer
        self.mode = mode
        self.data, self.label = self.load_data(file_name)

        self.special_tokens = [tokenizer.pad_token_id, tokenizer.bos_token_id,
                               tokenizer.eos_token_id, tokenizer.unk_token_id,
                               tokenizer.mask_token_id]
        self.mask_token_id = tokenizer.mask_token_id
        self.pad_token_id = tokenizer.pad_token_id 

        self.inputs_id, self.attention_mask, self.label_id = self._preprocess(self.data, self.label)
  
    def load_data(self, filename):
        raise NotImplementedError("loda data not implemented")
  
    def random_masking(self, token_ids, gt_ids, mask_token_id):
        rands = np.random.random(len(token_ids))
        src, tgt, mask = [], [], []
        for r, t, g in zip(rands, token_ids, gt_ids):
            if r < 0.15 and t not in self.special_tokens:
                src.append(mask_token_id)
                tgt.append(g)
                mask.append(1)
            else:
                src.append(t)
                tgt.append(g)
                mask.append(0)
                
        return src, tgt, mask
    
    def _preprocess(self, inputs, labels):
        # return (inputs_id, attention_mask, label_id, label_mask)
        assert len(inputs) == len(labels)
        
        max_len = 64
        out_inputs_id = []
        out_attention_mask = []
        out_label_id = []
        for i in range(len(inputs)):
            input = f"{inputs[i]} : "
            label = labels[i]
            input_id = self.tokenizer.encode(input,
                                add_special_tokens=False,
                                return_tensors='np'
                              )[0]
            label_id = self.tokenizer.encode(label,
                                add_special_tokens=False,
                                return_tensors='np'
                              )[0]
            label_id = np.concatenate((label_id, [self.tokenizer.pad_token_id]))
            
            input_label_id = np.concatenate((input_id, label_id))
            label_id = np.concatenate(([-100] * len(input_id), label_id))
            attention_mask = [1] * len(input_label_id)
            
            assert len(input_label_id)==len(attention_mask) and\
                   len(attention_mask)==len(label_id) 
        
            # max_len = max(max_len, len(input_label_id))
            out_inputs_id.append(input_label_id)
            out_attention_mask.append(attention_mask)
            out_label_id.append(label_id)

        # padding to same len 
        for i in range(len(out_inputs_id)):
            if self.tokenizer.padding_side == 'right':
                out_inputs_id[i] = np.concatenate((out_inputs_id[i],
                                                [self.pad_token_id] * (max_len-len(out_inputs_id[i]))))
                out_attention_mask[i] = np.concatenate((out_attention_mask[i],
                                                [0] * (max_len-len(out_attention_mask[i]))))
                out_label_id[i] = np.concatenate((out_label_id[i],
                                                [-100] * (max_len-len(out_label_id[i]))))
            else:
                out_inputs_id[i] = np.concatenate(([self.pad_token_id] * (max_len-len(out_inputs_id[i])),
                                                   out_inputs_id[i]))
                out_attention_mask[i] = np.concatenate(([0] * (max_len-len(out_attention_mask[i])),
                                                        out_attention_mask[i]))
                out_label_id[i] = np.concatenate(([-100] * (max_len-len(out_label_id[i])),
                                                  out_label_id[i]))
            assert len(out_inputs_id[i])==len(out_attention_mask[i]) and\
                   len(out_inputs_id[i])==len(out_label_id[i])
        
        return out_inputs_id,out_attention_mask,out_label_id
    
    def __getitem__(self, idx):
        item = {
            'input_ids': torch.tensor(self.inputs_id[idx],dtype=torch.int),
            'attention_mask': torch.tensor(self.attention_mask[idx],dtype=torch.int),
            'labels': torch.tensor(self.label_id[idx],dtype=torch.int),
        }   
        return item 
        
    def __len__(self):
        return len(self.inputs_id)    

class TwitterComplaintDataset(MlmBaseDataset):
    def __init__(self,tokenizer,file_name=None,random=False,mode='train'):
        super().__init__(tokenizer,file_name,random,mode)
        
    def load_data(self, filepath=None):
        dataset = load_dataset("ought/raft", "twitter_complaints")
        classes = [k.replace("_", " ") for k in dataset[self.mode].features["Label"].names]

        D = []
        T = []
        for i in range(len(dataset[self.mode])):
            text = dataset[self.mode][i]['Tweet text']
            label = classes[dataset[self.mode][i]['Label']]
            D.append(text)
            T.append(label)

        return D, T 
