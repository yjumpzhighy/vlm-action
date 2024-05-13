import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
import os
from common.gaussian_diffusion import GaussianDiffusion
from common.conditional_unet import ConditionalUNet
from common.utils import inception_score, frechet_inception_distance, cosine_with_warmup_scheduler
from common.datasets import FlickrDataset, MnistDataset, CifarDataset

if __name__ == "__main__":
    TIMESTEPS = 1000
    BATCHSIZE = 128
    IMAGE_SIZE = 32
    IMAGE_C = 3
    NUM_EPOCHS = 10
    SAVE_MODEL_PATH = 'data/ddpm/cifar_1000epoch.pth'
    TRAIN_DDPM = True

    if TRAIN_DDPM:
        from transformers import CLIPTokenizer
        tokenizer = CLIPTokenizer.from_pretrained(
            'openai/clip-vit-base-patch32')
        data_train = CifarDataset(
            os.path.abspath(os.path.join(os.getcwd(), '../../Data/cifar10')),
            IMAGE_SIZE,
            transforms.Compose([
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor()
            ]), 'train', tokenizer)

        # from transformers import DistilBertTokenizer
        # tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        # data_train = FlickrDataset(
        #     os.path.abspath(os.path.join(os.getcwd(), '../../Data/flickr30k/captions_fixed.csv')),
        #     os.path.abspath(os.path.join(os.getcwd(), '../../Data/flickr30k/flickr30k_images/')),
        #     tokenizer, 256, IMAGE_SIZE)
        train_loader = torch.utils.data.DataLoader(data_train,
                                                   batch_size=BATCHSIZE,
                                                   num_workers=4,
                                                   shuffle=True)

        diffuser = GaussianDiffusion(
            ConditionalUNet(IMAGE_SIZE,
                            IMAGE_SIZE,
                            IMAGE_C,
                            base_channel=64,
                            output_channel=IMAGE_C), IMAGE_SIZE, IMAGE_C,
            TIMESTEPS, 'pred_noise').cuda()
        optimizer = torch.optim.Adam(diffuser.parameters(), lr=0.001)
        scheduler = cosine_with_warmup_scheduler(optimizer, NUM_EPOCHS,
                                                BATCHSIZE,
                                                len(train_loader.dataset),
                                                1)

        total_step = len(train_loader)
        cur_loss = np.finfo(np.float32).max
        for epoch in range(NUM_EPOCHS):
            for idx, batch in enumerate(train_loader):
                x = batch['image']
                if 'label' in batch.keys():
                    mask = (batch['label'] == 5)
                    x = x[mask]
                    if x.shape[0] == 0:
                        continue
                x = x.cuda()

                loss = diffuser(x)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

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
                                       IMAGE_SIZE,
                                       IMAGE_C,
                                       TIMESTEPS,
                                       objective='pred_noise',
                                       device=torch.device('cuda')).cuda()
        test_model.load_state_dict(torch.load(SAVE_MODEL_PATH))

        figure = np.zeros((IMAGE_SIZE * 3, IMAGE_SIZE * 3, IMAGE_C))
        samples = test_model.sample([9, IMAGE_C, IMAGE_SIZE, IMAGE_SIZE])
        samples = samples.permute(
            0, 2, 3, 1).contiguous().detach().cpu().numpy()  #[b,h,w,c]
        
        for k in range(9):
            img = samples[k]
            plt.imshow(img)
            plt.show()
        
        for i in range(9):
            fig_row_idx = int(i / 3)
            fig_col_idx = int(i % 3)
            figure[fig_row_idx * IMAGE_SIZE:(fig_row_idx + 1) * IMAGE_SIZE,
                   fig_col_idx * IMAGE_SIZE:(fig_col_idx + 1) *
                   IMAGE_SIZE, :] = samples[i]

        figure = (figure * 255).astype('uint8')
        if IMAGE_C == 1:
            figure = np.squeeze(figure, axis=-1)
            plt.imsave('output_image4.png', figure, cmap='gray', format='png')
        elif IMAGE_C == 3:
            plt.imsave('output_image4.png', figure, format='png')
        else:
            raise ValueError("image channels not supported.")
