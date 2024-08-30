import os
import torch
import numpy as np
import h5py
from scipy import ndimage
import datasets
import pandas as pd
import cv2
from datasets import load_dataset, DownloadConfig

from torchvision.datasets import MNIST, CIFAR10, ImageFolder

def image_zoom(x, raw_h, tgt_h, raw_w, tgt_w, order=0):
    if tgt_h != raw_h or tgt_w != raw_w:
            x = ndimage.zoom(x, (float(tgt_h)/raw_h, float(tgt_w)/raw_w), order=order)
    return x

class Tokenizer(torch.utils.data.Dataset):
    def __init__(self, tokenizer, max_len=512):
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        if hasattr(tokenizer,'eos_token') and hasattr(tokenizer,'pad_token'):
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.pad_token_id = tokenizer.pad_token_id if hasattr(tokenizer,'pad_token') else None
        self.bos_token_id = tokenizer.bos_token_id if hasattr(tokenizer,'bos_token') else None
        self.eos_token_id = tokenizer.eos_token_id if hasattr(tokenizer,'eos_token') else None
        self.unk_token_id = tokenizer.unk_token_id if hasattr(tokenizer,'unk_token') else None
        self.mask_token_id = tokenizer.mask_token_id if hasattr(tokenizer,'mask_token') else None
            
    def tokenize_text_with_label(self, inputs, labels):
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
                                                [self.pad_token_id] * (self.max_len-len(out_inputs_id[i]))))
                out_attention_mask[i] = np.concatenate((out_attention_mask[i],
                                                [0] * (self.max_len-len(out_attention_mask[i]))))
                out_label_id[i] = np.concatenate((out_label_id[i],
                                                [-100] * (self.max_len-len(out_label_id[i]))))
            else:
                out_inputs_id[i] = np.concatenate(([self.pad_token_id] * (self.max_len-len(out_inputs_id[i])),
                                                   out_inputs_id[i]))
                out_attention_mask[i] = np.concatenate(([0] * (self.max_len-len(out_attention_mask[i])),
                                                        out_attention_mask[i]))
                out_label_id[i] = np.concatenate(([-100] * (self.max_len-len(out_label_id[i])),
                                                  out_label_id[i]))
            assert len(out_inputs_id[i])==len(out_attention_mask[i]) and\
                   len(out_inputs_id[i])==len(out_label_id[i])
        
        return out_inputs_id,out_attention_mask,out_label_id

    def tokenize_text(self, input):
        # input: string
        
        out_inputs_id = []
        out_attention_mask = []


        input_id = self.tokenizer.encode(input,
                            add_special_tokens=False,
                            return_tensors='np'
                            )[0]

        input_id = np.concatenate((input_id, [self.pad_token_id]))

        attention_mask = [1] * len(input_id)

        if self.tokenizer.padding_side == 'right':
            input_id = np.concatenate((input_id,
                                            [self.pad_token_id] * (self.max_len-len(input_id))))
            attention_mask = np.concatenate((attention_mask,
                                            [0] * (self.max_len-len(attention_mask))))
        else:
            input_id = np.concatenate(([self.pad_token_id] * (self.max_len-len(input_id)),
                                                input_id))
            attention_mask = np.concatenate(([0] * (self.max_len-len(attention_mask)),
                                                    attention_mask))
        return input_id, attention_mask

class SynapseDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, list_dir, split, img_h, img_w):
        self.data_dir = data_dir
        self.list_dir = list_dir
        self.split = split
        self.img_h = img_h
        self.img_w = img_w
        self.sample_list = open(os.path.join(self.list_dir, self.split+'.txt')).readlines()
        
    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name+'.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']
            
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            data = h5py.File(filepath)
            image, label = data['image'][:], data['label'][:]
   
        image = image_zoom(image, image.shape[0], self.img_h, image.shape[1], self.img_w, 3)
        label = image_zoom(label, label.shape[0], self.img_h, label.shape[1], self.img_w, 0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0).repeat(3,1,1)      
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label}
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample
    
class ImageNetDataset(Tokenizer):
    def __init__(self, rootdir, image_size=224, transform=None, mode='train', tokenizer=None, 
                 max_token_len=512):
        self.image_size = image_size
        self.transform = transform
        #self.dataset = datasets.load_from_disk(rootdir)[mode]
        
        if rootdir is None:
            self.dataset = load_dataset('imagenet-1k', split=mode, download_config=DownloadConfig(resume_download=True))
        else:
            self.dataset = datasets.load_from_disk(rootdir) #.select(range(10000))
        self.dataset.set_transform(self.transform_img)
        self.classes = self.dataset.features['label']

        super().__init__(tokenizer, max_token_len)
 
    def transform_img(self, data):
        data['image'] = [self.transform(img) for img in data['image']]
        return data
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]

        image = data['image']
        label = data['label']
        label_str = self.classes.int2str(label) if label != -1 else 'invalid'
        image_mode = image.mode


        if self.image_size != image.shape[1] or self.image_size != image.shape[2]:
            # image = image.resize((self.image_size, self.image_size)) 
            raise ValueError("image shape not resized correctly.")    
                  
        if image.shape[0]==1: #L
            image = image.repeat(3,1,1)
        elif image.shape[0]==4: #RGBA
            image = image[:3,:,:]
            
        # tokenize label text
        text_id, attention_mask = self.tokenize_text(label_str)    
            
        item = {}
        item['image'] = image.float()
        item['label'] = label
        item['text_id'] = text_id
        item['attention_mask'] = attention_mask
        return item
    
class FlickrDataset(torch.utils.data.Dataset):
    def __init__(self, captions_path, images_path, tokenizer=None, token_max_len=256, 
                 image_size=224, transforms=None, mode='train'):
        self.images_path = images_path
        self.captions_path = captions_path
        
        dataframe = pd.read_csv(captions_path)
        max_id = dataframe["id"].max() + 1
        image_ids = np.arange(0, max_id)
        # 2-8 eval/train split
        valid_ids = np.random.choice(
            image_ids, size=int(0.2 * len(image_ids)), replace=False
        )
        train_ids = [id_ for id_ in image_ids if id_ not in valid_ids]
        
        if mode=='train':
            dataframe = dataframe[dataframe["id"].isin(train_ids)].reset_index(drop=True)
        elif mode=='valid':
            dataframe = dataframe[dataframe["id"].isin(valid_ids)].reset_index(drop=True)
        else:
            raise ValueError("mode not supported")

        self.images = list(dataframe['image'].values)
        self.captions = list(dataframe['caption'].values)
        
        self.encoded_captions = tokenizer(list(self.captions),
                                          padding=True,
                                          truncation=True,
                                          max_length=token_max_len,
                                          return_tensors='pt')
        self.token_max_length = len(self.encoded_captions['input_ids'][0])
 
        self.transforms = transforms
        self.image_size = image_size
    
    def __getitem__(self, idx):
        item = {
            key: values[idx]
            for key, values in self.encoded_captions.items()
        }
        
        image = cv2.imread(f"{self.images_path}/{self.images[idx]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #[h,w,3]
        
        if self.image_size != image.shape[0] or self.image_size != image.shape[1]:
            # image = ndimage.zoom(image, (float(self.image_size)/image.shape[0], 
            #                      float(self.image_size)/image.shape[1], 1))
            image = cv2.resize(image, (self.image_size,self.image_size),
                               interpolation=cv2.INTER_LINEAR)

        # iamge normalize to [0,1]
        image = image / 255.

        item['image'] = torch.tensor(image).permute(2,0,1).float()
        item['caption'] = self.captions[idx]
        
        return item
    def __len__(self):
        return len(self.captions)
    
class MnistDataset(torch.utils.data.Dataset):
    def __init__(self, rootdir, image_size=32, transform=None, mode='train'):
        self.image_size = image_size
        self.dataset = MNIST(root=rootdir, train=(mode=='train'), download=False, transform=transform)
    
    def __getitem__(self, idx):
        data = self.dataset[idx]
        
        image = np.array(data[0])
        if self.image_size != image.shape[1] or self.image_size != image.shape[2]:
            # image = ndimage.zoom(image, (1, float(self.image_size)/image.shape[1], 
            #                      float(self.image_size)/image.shape[2]))
            raise ValueError("image shape not resized correctly.")
        
        item = {}
        item['image'] = torch.tensor(image).float()
        item['label'] = data[1]
        return item
        
    def __len__(self):
        return len(self.dataset)
    
class CifarDataset(Tokenizer):
    def __init__(self, rootdir, image_size=32, transform=None, mode='train', tokenizer=None, 
                 max_token_len=512):
        self.image_size = image_size
        self.dataset = CIFAR10(root=rootdir, train=(mode=='train'), 
                               transform=transform, download=False)
        self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
        super().__init__(tokenizer, max_token_len)
    
    def __getitem__(self, idx):
        data = self.dataset[idx]
        
        image = np.array(data[0])
        if self.image_size != image.shape[1] or self.image_size != image.shape[2]:
            # image = ndimage.zoom(image, (1, float(self.image_size)/image.shape[1], 
            #                      float(self.image_size)/image.shape[2]))
            raise ValueError("image shape not resized correctly.")
        
        # tokenize label text
        text_id, attention_mask = self.tokenize_text(self.classes[data[1]])
        
        item = {}
        item['image'] = torch.tensor(image).float()
        item['label'] = data[1]
        item['text_id'] = text_id
        item['attention_mask'] = attention_mask
        
        return item
        
    def __len__(self):
        return len(self.dataset)

class ButterfliesDataset(Tokenizer):
    def __init__(self, rootdir, image_size=224, transform=None, mode='train', tokenizer=None, 
                 max_token_len=512):
        self.image_size = image_size
        self.transform = transform
        #self.dataset = datasets.load_from_disk(rootdir)[mode]
        
        self.dataset = load_dataset('huggan/smithsonian_butterflies_subset', split=mode)
        self.dataset.set_transform(self.transform_img)
        
        super().__init__(tokenizer, max_token_len)
 
    def transform_img(self, data):
        data['image'] = [self.transform(img) for img in data['image']]
        return data
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]

        image = data['image']
        label = -1
        label_str = 'invalid'
        image_mode = image.mode

        if self.image_size != image.shape[1] or self.image_size != image.shape[2]:
            # image = image.resize((self.image_size, self.image_size)) 
            raise ValueError("image shape not resized correctly.")    
                  
        if image.shape[0]==1: #L
            image = image.repeat(3,1,1)
        elif image.shape[0]==4: #RGBA
            image = image[:3,:,:]
            
        # tokenize label text
        text_id, attention_mask = self.tokenize_text(label_str)    
            
        item = {}
        item['image'] = image.float()
        item['label'] = label
        item['text_id'] = text_id
        item['attention_mask'] = attention_mask
        return item
