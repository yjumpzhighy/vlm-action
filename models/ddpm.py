import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
import os
from common.gaussian_diffusion import GaussianDiffusion
from common.conditional_unet import ConditionalUNet
from common.utils import inception_score, frechet_inception_distance
from common.datasets import FlickrDataset, MnistDataset, CifarDataset


if __name__ == "__main__":
    TIMESTEPS = 1000
    BATCHSIZE = 16
    IMAGE_SIZE = 200
    IMAGE_C = 3
    NUM_EPOCHS = 100
    SAVE_MODEL_PATH = 'data/ddpm/best.pth'
    # data_train = CifarDataset('/home/zuyuanyang/Projects/vlm/data/cifar', IMAGE_SIZE, 
    #                           transforms.Compose([transforms.ToTensor()]), 'train')
 
    from transformers import DistilBertTokenizer
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    data_train = FlickrDataset("/home/zuyuanyang/Data/flickr30k/captions_fixed.csv",
                     "/home/zuyuanyang/Data/flickr30k/flickr30k_images/",
                     tokenizer,
                     256,
                     IMAGE_SIZE)
    train_loader = torch.utils.data.DataLoader(
        data_train,
        batch_size=BATCHSIZE,
        num_workers=4,
        shuffle=True)

    diffuser = GaussianDiffusion(ConditionalUNet(IMAGE_SIZE,
                                                IMAGE_SIZE,
                                                IMAGE_C,
                                                base_channel=64,
                                                output_channel=IMAGE_C),
                                 IMAGE_SIZE, IMAGE_C, TIMESTEPS, 'pred_noise').cuda()
    optimizer = torch.optim.Adam(diffuser.parameters(), lr=0.001)

    total_step = len(train_loader)
    cur_loss = np.finfo(np.float32).max
    for epoch in range(NUM_EPOCHS):
        for idx, batch in enumerate(train_loader):
            x = batch['image']
            if 'label' in batch.keys():
                mask = (batch['label'] == 5)
                x = x[mask]
                if x.shape[0]==0:
                    continue
            x = x.cuda()
                
            loss = diffuser(x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                    epoch + 1, NUM_EPOCHS, idx + 1, total_step, loss.item()))

        # save best model
        if loss.item() < cur_loss:
            cur_loss = loss.item()
            torch.save(diffuser.state_dict(), SAVE_MODEL_PATH)

    with torch.no_grad():
        test_model = GaussianDiffusion(ConditionalUNet(IMAGE_SIZE,
                                                       IMAGE_SIZE,
                                                       IMAGE_C,
                                                       base_channel=64,
                                                       output_channel=3),
                                       IMAGE_SIZE, IMAGE_C, TIMESTEPS, 
                                       objective='pred_noise', device=torch.device('cuda')).cuda()
        test_model.load_state_dict(torch.load(SAVE_MODEL_PATH))

        sample_batch = 9
        # ddpm sampling
        samples = test_model.sample(sample_batch)
        samples = samples.permute(0, 2, 3, 1).contiguous().detach().cpu() #[b,h,w,c]
        # display images
        for k in range(sample_batch):
            img = samples[k].numpy()
            plt.imshow(img)
            plt.show()
 
        # # metrics
        # samples = samples.view(-1, 3)
        # print(inception_score(samples))
        # print(frechet_inception_distance(samples, samples))
        
    
        
        
        
