import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.stats import norm


class VAE(nn.Module):
    def __init__(self,input_dim=28,embedding_dim=256,latent_dim=2):
        super(VAE, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, latent_dim * 2)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, input_dim),
            nn.Sigmoid()
        )
        
    def reparameter(self,mean,logvar):
        eps = torch.randn_like(mean)  #random sample from (0,1)
        return mean + torch.exp(logvar * 0.5) * eps  # u + sqrt(var)*eps ~ N(u, var)


    def forward(self,x):
        ori_size = x.size()
        batch_size = ori_size[0]
        
        x = x.view(batch_size, -1)
        x = self.encoder(x)
        mean, logvar = x.chunk(2, dim=1)
        z = self.reparameter(mean, logvar)
        
        x = self.decoder(z).view(ori_size)
        return x, mean, logvar
                
    def get_kl_loss(self, mean, logvar):
        kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp()) 
        return kl_loss
    
    def get_recon_loss(self, x, recon_x,):
        recon_loss = F.binary_cross_entropy(recon_x,x,size_average=False)
        return recon_loss
        

# VAE unit test 
if __name__ == "__main__":        
    EPOCHS = 1000
    BATCH_SIZE = 128
    INPUT_DIM = 28 * 28
    EMBEDDING_DIM = 128
    LATENT_DIM = 2

    transform = transforms.Compose([transforms.ToTensor()])
    data_train = MNIST(root='./data', train=True, download=False, transform=transform)
    data_valid = MNIST(root='./data', train=False, download=False, transform=transform)

    train_loader = DataLoader(data_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(data_valid, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = VAE(INPUT_DIM, EMBEDDING_DIM, LATENT_DIM)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-2)

    best_loss = 1e9
    best_epoch = 0

    valid_losses = []
    train_losses = []

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.
        train_num = len(train_loader.dataset)

        d = None
        for idx, (x, _) in enumerate(train_loader):
            batch = x.size(0)
            x = x.to(device)
            recon_x, mu, logvar = model(x)
            d = logvar
            
            kl_loss = model.get_kl_loss(mu, logvar)
            recon_loss = model.get_recon_loss(x, recon_x)
            loss = recon_loss + kl_loss 
            train_loss += loss.item()
            loss = loss / batch

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Training loss {loss: .3f} \t Recon {recon_loss / batch: .3f} \t KL {kl_loss / batch: .3f} in Epoch {epoch}")

        train_losses.append(train_loss / train_num)

        valid_loss = 0.
        valid_recon = 0.
        valid_kl = 0.
        valid_num = len(test_loader.dataset)
        model.eval()
        with torch.no_grad():
            for idx, (x, _) in enumerate(test_loader):
                x = x.to(device)
                recon_x, mu, logvar = model(x)
                
                kl_loss = model.get_kl_loss(mu, logvar)
                recon_loss = model.get_recon_loss(x, recon_x)
                loss = recon_loss + kl_loss 
                valid_loss += loss.item()

            valid_losses.append(valid_loss / valid_num)
            print(f"Valid loss {valid_loss / valid_num: .3f} \t Recon {recon_loss / valid_num: .3f} \t KL {kl_loss / valid_num: .3f} in epoch {epoch}")

                
    plt.plot(train_losses, label='Train')
    plt.plot(valid_losses, label='Valid')
    plt.legend()
    plt.title('Learning Curve');
    plt.show()


    # plot generated images from latent z
    n = 20
    digit_size = 28

    grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
    grid_y = norm.ppf(np.linspace(0.05, 0.95, n))


    model.to(torch.device('cpu'))
    model.eval()
    figure = np.zeros((digit_size * n, digit_size * n))
    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            t = [xi, yi]
            z_sampled = torch.FloatTensor(t)
            with torch.no_grad():
                decode = model.decoder(z_sampled)
                digit = decode.view((digit_size, digit_size))
                figure[
                    i * digit_size: (i + 1) * digit_size,
                    j * digit_size: (j + 1) * digit_size
                ] = digit

    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap="Greys_r")
    plt.xticks([])
    plt.yticks([])
    plt.axis('off');
    plt.show()
