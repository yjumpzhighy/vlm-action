# Segment Anything

![image](https://github.com/user-attachments/assets/c8924a90-157f-459c-873f-570a99a1ec32)



    #1. Image encoder
    x = Conv2d(kernel_size=16, stride=16)(img)  #embedding to patches [B,H/16,W/16,C]
    pos_embed = Parameter(zeros(1, H // 16, W // 16, C)  #init absolute positional embedding
    #classic 12 multihead self-attention blocks and mlp neck, [B,H/16,W/16,C]
    x = ViT(x + pos_embed)  

    #2. Prompt encoder
    
    
