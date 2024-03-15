import os
import torch
import numpy as np
import h5py
from scipy import ndimage
import torchvision.datasets as datasets


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
