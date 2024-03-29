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
from common.datasets import FlickrDataset, MnistDataset

if __name__ == "__main__":
    EPOCHS = 10
    BATCH_SIZE = 128
    IMAGE_SIZE = 32
    IMAGE_C = 1
    EMBEDDING_DIM = 8
    LATENT_DIM = 2
    
    data_train = MnistDataset(os.path.abspath(os.path.join(os.getcwd(), '../../Data')),
                              IMAGE_SIZE, 
                              transforms.Compose([transforms.ToTensor()]), 'train')    
    train_loader = torch.utils.data.DataLoader(
        data_train,
        batch_size=BATCH_SIZE,
        num_workers=4,
        shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VariationalAutoEncoder(IMAGE_SIZE,IMAGE_C,EMBEDDING_DIM,IMAGE_C,LATENT_DIM)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(EPOCHS):
        model.train()
        train_num = len(train_loader.dataset)
        for idx, batch in enumerate(train_loader):
            x = batch['image'].to(device)

            recon_x, mu, logvar = model(x)
            kl_loss = model.get_kl_loss(mu, logvar)
            recon_loss = model.get_recon_loss(x, recon_x)
            loss = kl_loss + recon_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()  
        print(f"Training loss {loss: .3f} \t Recon {recon_loss : .3f} \t KL {kl_loss : .3f} in Epoch {epoch}")

    torch.save(model.state_dict(), 'data/vae/best.pth')

    # plot generated images from latent z
    n = 10
    grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
    grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

    test_model = VariationalAutoEncoder(IMAGE_SIZE,IMAGE_C,EMBEDDING_DIM,IMAGE_C,LATENT_DIM)
    test_model.load_state_dict(torch.load('data/vae/best.pth'))
    test_model = test_model.to(torch.device('cpu'))
    test_model.eval()
    figure = np.zeros((IMAGE_SIZE * n, IMAGE_SIZE * n, IMAGE_C))
    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            t = [[xi, yi]]
            z_sampled = torch.FloatTensor(t)
            with torch.no_grad():
                decode = test_model.decode_latent(z_sampled)
                digit = decode.view(IMAGE_SIZE, IMAGE_SIZE, IMAGE_C)
                figure[
                    i * IMAGE_SIZE: (i + 1) * IMAGE_SIZE,
                    j * IMAGE_SIZE: (j + 1) * IMAGE_SIZE,
                    :
                ] = digit
    plt.figure(figsize=(10,10))
    plt.imshow(figure, cmap="Greys_r")
    plt.xticks([])
    plt.yticks([])
    plt.axis('off');
    plt.show()
