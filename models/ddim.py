import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
import os
from common.gaussian_diffusion import GaussianDiffusion
from common.conditional_unet import ConditionalUNet
from common.utils import inception_score, frechet_inception_distance

if __name__ == "__main__":
    
    TIMESTEPS = 1000
    BATCHSIZE = 128
    IMAGE_SIZE = 32
    IMAGE_C = 3
    NUM_EPOCHS = 0
    SAVE_MODEL_PATH = 'data/ddpm/best.pth'

    transform = transforms.Compose([
        # transforms.Pad(4),
        # transforms.RandomHorizontalFlip(),
        #transforms.RandomCrop(IMAGE_SIZE),
        transforms.ToTensor(),
    ])

    train_dataset = torchvision.datasets.CIFAR10(root='./data/cifar',
                                                 train=True,
                                                 transform=transform,
                                                 download=False)

    # 测试数据集
    test_dataset = torchvision.datasets.CIFAR10(root='./data/cifar',
                                                train=False,
                                                transform=transform,
                                                download=False)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=BATCHSIZE,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=BATCHSIZE,
                                              shuffle=False)

    diffuser = GaussianDiffusion(ConditionalUNet(IMAGE_SIZE,
                                                IMAGE_SIZE,
                                                IMAGE_C,
                                                base_channel=64,
                                                output_channel=3),
                                 IMAGE_SIZE, IMAGE_C, TIMESTEPS, 'pred_noise').cuda()
    optimizer = torch.optim.Adam(diffuser.parameters(), lr=0.001)

    total_step = len(train_loader)
    cur_loss = np.finfo(np.float32).max
    for epoch in range(NUM_EPOCHS):
        for i, (images, labels) in enumerate(train_loader):
            images = images.cuda()

            mask = (labels == 5)
            images = images[mask]
            if images.shape[0]==0:
                continue


            loss = diffuser(images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                    epoch + 1, NUM_EPOCHS, i + 1, total_step, loss.item()))

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
                                       IMAGE_SIZE, IMAGE_C, TIMESTEPS, 100, 0., 
                                       'pred_noise', torch.device('cuda')).cuda()
        test_model.load_state_dict(torch.load(SAVE_MODEL_PATH))

        sample_batch = 3
        # ddim sampl
        samples = test_model.sample_ddim(sample_batch) 
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
