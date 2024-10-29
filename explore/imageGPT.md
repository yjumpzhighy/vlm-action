#Generative Pretraining from Pixels

sequence Transformer to auto-regressively predict pixels, withoutincorporating knowledge of the 2D input structure.

![image](https://github.com/user-attachments/assets/90a6c6fb-b5ec-4983-9e46-cf3bc51d87cd)

1) downsample to 32x32 low resolution image, then flatten to 1-d by rows.
2) use gpt2 model, to predict next pixel based on generated pixels
3) use bert model, to predict masked pixels
4) finetune with linear prob, to test the model feature extraction ablility
5) all operations on pixel spaces directly
   
