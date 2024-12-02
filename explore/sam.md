# Segment Anything

![image](https://github.com/user-attachments/assets/c8924a90-157f-459c-873f-570a99a1ec32)



    #1. Image encoder
    x = Conv2d(kernel_size=16, stride=16)(img)  #embedding to patches [B,H/16,W/16,C]
    pos_embed = Parameter(zeros(1, H // 16, W // 16, C)  #init absolute positional embedding
    #classic 12 multihead self-attention blocks and mlp neck, [B,H/16,W/16,C]
    x = ViT(x + pos_embed)  

    #2. Prompt encoder
    point_embeddings = [Embedding(1, embed_dim),Embedding(1, embed_dim),
                        Embedding(1, embed_dim),Embedding(1, embed_dim)]
    pos_embed_gaussian_matrix = randn(2, embed_dim/2)

    ## 2.1 encode points 
    # points[bs,2,2], format like [[[x, y],[0,0]]] unnormalized within [1024,1024]
    # bs is the number of labels, each label include 1 point
    # labels[bs,2], format like [[1,-1]]
    points = (points + 0.5) / 1024  # Shift to center of pixel and normalize to [0,1]
    points = points * 2 - 1 # Shift to [-1,1]
    points_embed = points @ pos_embed_gaussian_matrix #[bs,2,embed_dim/2]
    points_embed = cat([sin(2*pi*points_embed), cos(2*pi*points_embed)],
                       dim=-1) #[bs,2,embed_dim]
    points_embed[labels==-1] = 0.0 #reset padding point to zero
    points_embed[labels==1] += point_embeddings[1].weight #[bs,2,embed_dim]
    
    ## 2.2 encode box
    # box[bs,2,2], format like [[[x, y],[x',y']]] unnormalized within [1024,1024]
    # bs is the number of labels
    box = (box + 0.5) / 1024  # Shift to center of pixel and normalize to [0,1]
    box = box * 2 - 1 # Shift to [-1,1]
    box_embed = box @ pos_embed_gaussian_matrix #[bs,2,embed_dim/2]
    box_embed = cat([sin(2*pi*box_embed), cos(2*pi*box_embed)],
                       dim=-1) #[bs,2,embed_dim]
    box_embed[:,0,:] += point_embeddings[2].weight
    box_embed[:,1,:] += point_embeddings[3].weight #[bs,2,embed_dim]

    ## 2.3 encode mask

    #[bs,n,embed_dim], where n=0,2,4, depends on points or box avaliablity
    sparse_embed = cat([points_embed, box_embed], dim=1) 
    
