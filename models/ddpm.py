import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms

from common.gaussian_diffusion import GaussianDiffusion
from models.common.conditional_unet import ConditionalUNet

if __name__ == "__main__":
    TIMESTEPS = 1000
    BATCHSIZE = 32
    IMAGE_SIZE = 32
    IMAGE_C = 3

    transform = transforms.Compose([
        # transforms.Pad(4),
        # transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(BATCHSIZE),
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
    # 测试数据加载器
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=BATCHSIZE,
                                              shuffle=False)

    model = ConditionalUNet(IMAGE_SIZE,
                            IMAGE_SIZE,
                            IMAGE_C,
                            base_channel=64,
                            output_channel=3)
    diffuser = GaussianDiffusion(model, IMAGE_SIZE, IMAGE_C, TIMESTEPS).cuda()
    optimizer = torch.optim.Adam(diffuser.parameters(), lr=0.001)

    total_step = len(train_loader)
    num_epochs = 1
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.cuda()

            loss = diffuser(images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 500 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                    epoch + 1, num_epochs, i + 1, total_step, loss.item()))

    with torch.no_grad():
        sample_batch = 1
        samples = diffuser.sample(sample_batch)

        samples = samples.permute(0, 2, 3, 1).detach().cpu().numpy()
        # Display images
        plt.imshow(samples[0])
        plt.axis('off')
        plt.show()
