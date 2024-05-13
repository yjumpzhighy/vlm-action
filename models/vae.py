import torch
from torch import nn
import torch.nn.functional as F
from torch import optim

import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.stats import norm
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

from common.variational_auto_encoder import VariationalAutoEncoder
from common.datasets import FlickrDataset, MnistDataset, CifarDataset

if __name__ == "__main__":
    EPOCHS = 100
    BATCH_SIZE = 512
    IMAGE_SIZE = 32
    IMAGE_C = 3
    EMBEDDING_DIM = 64
    LATENT_DIM = 4

    from transformers import CLIPTokenizer
    tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')
    data_train = CifarDataset(
        os.path.abspath(os.path.join(os.getcwd(), '../../Data/cifar10')),
        IMAGE_SIZE,
        transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor()
        ]), 'train', tokenizer)

    train_loader = torch.utils.data.DataLoader(data_train,
                                               batch_size=BATCH_SIZE,
                                               num_workers=4,
                                               shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VariationalAutoEncoder(IMAGE_SIZE,
                                   IMAGE_C,
                                   EMBEDDING_DIM,
                                   output_chann=IMAGE_C,
                                   latent_chann=LATENT_DIM).to(device)
    optimizer = model.configure_optimziers(lr=1e-4)[0]

    for epoch in range(EPOCHS):
        model.train()
        train_num = len(train_loader.dataset)
        for idx, batch in enumerate(train_loader):
            x = batch['image'].to(device)

            *_, loss, log = model(x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Training loss {loss: .3f} in Epoch {epoch}")
    torch.save(model.state_dict(), 'data/vae/best.pth')

    n = 3
    linspace = 10
    grids = [
        norm.ppf(np.linspace(0.05, 0.95, linspace))
        for i in range(model.encoded_image_h * model.encoded_image_w *
                       LATENT_DIM)
    ]

    figure = np.zeros((IMAGE_SIZE * n, IMAGE_SIZE * n, IMAGE_C))
    with torch.no_grad():
        for i in range(n * n):
            z_sampled = np.array([
                np.random.choice(grids[i])
                for i in range(model.encoded_image_h * model.encoded_image_w *
                               LATENT_DIM)
            ]).reshape(1, LATENT_DIM, model.encoded_image_h,
                       model.encoded_image_w)
            z_sampled = torch.FloatTensor(z_sampled).cuda()
            decode = model(z_sampled, mode='decode')[0]
            digit = decode[0].permute(1, 2,
                                      0).contiguous().detach().cpu().numpy()

            fig_row_idx = int(i / n)
            fig_col_idx = int(i % n)
            figure[fig_row_idx * IMAGE_SIZE:(fig_row_idx + 1) * IMAGE_SIZE,
                   fig_col_idx * IMAGE_SIZE:(fig_col_idx + 1) *
                   IMAGE_SIZE, :] = digit

    figure = (figure * 255).astype('uint8')
    if IMAGE_C == 1:
        figure = np.squeeze(figure, axis=-1)
        plt.imsave('vae_image.png', figure, cmap='gray', format='png')
    elif IMAGE_C == 3:
        plt.imsave('vae_image.png', figure, format='png')
    else:
        raise ValueError("image channels not supported.")
