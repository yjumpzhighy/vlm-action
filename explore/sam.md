# Segment Anything

![image](https://github.com/user-attachments/assets/c8924a90-157f-459c-873f-570a99a1ec32)



    #1. Image encoder
    x = Conv2d(kernel_size=16, stride=16)(img)  #embedding to patches [B,H/16,W/16,C]
    pos_embed = Parameter(zeros(1, H // 16, W // 16, C)  #init absolute positional embedding
    #classic 12 multihead self-attention blocks and mlp neck, [B,H/16,W/16,C]
    x = ViT(x + pos_embed)  

    #2. Prompt encoder
    PositionEmbeddingRandom(embed_dim // 2)

    ## 2.1 encode points 
    # points[bs,1,2], where (x,y) get unnormalized to [1024,1024],
    # bs is the number of labels, each label include 1 point
    pos_embed_gaussian_matrix = randn(2,embed_dim // 2)
    points = (points + 0.5) / 1024  # Shift to center of pixel and normalize to [0,1]
    points = points * 2 - 1 # Shift to [-1,1]
    points_embed = points @ pos_embed_gaussian_matrix #[bs,1,embed_dim/2]
    points_embed = cat([sin(2*pi*points_embed), cos(2*pi*points_embed)],
                       dim=-1) #[bs,1,embed_dim]
