import os
import torch
import numpy as np
import h5py
from scipy import ndimage
import torchvision.datasets as datasets
import pandas as pd
import cv2
from torchvision.datasets import MNIST, CIFAR10

def image_zoom(x, raw_h, tgt_h, raw_w, tgt_w, order=0):
    if tgt_h != raw_h or tgt_w != raw_w:
            x = ndimage.zoom(x, (float(tgt_h)/raw_h, float(tgt_w)/raw_w), order=order)
    return x

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
    
class ImageNetDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.train_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
        
    def __len__(self):
        return self.train_dataset.__len__()

    def __getitem__(self, idx):
        return self.train_dataset[idx]
    
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
            image = ndimage.zoom(image, (1, float(self.image_size)/image.shape[1], 
                                 float(self.image_size)/image.shape[2]))
        
        item = {}
        item['image'] = torch.tensor(image).float()
        item['label'] = data[1]
        return item
        
    def __len__(self):
        return len(self.dataset)
    
class CifarDataset(torch.utils.data.Dataset):
    def __init__(self, rootdir, image_size=32, transform=None, mode='train'):
        self.image_size = image_size
        self.dataset = CIFAR10(root=rootdir, train=(mode=='train'), 
                               transform=transform, download=False)
    
    def __getitem__(self, idx):
        data = self.dataset[idx]
        
        image = np.array(data[0])
        if self.image_size != image.shape[1] or self.image_size != image.shape[2]:
            image = ndimage.zoom(image, (1, float(self.image_size)/image.shape[1], 
                                 float(self.image_size)/image.shape[2]))
        
        item = {}
        item['image'] = torch.tensor(image).float()
        item['label'] = data[1]
        return item
        
    def __len__(self):
        return len(self.dataset)
