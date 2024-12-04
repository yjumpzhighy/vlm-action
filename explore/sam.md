# Segment Anything

![image](https://github.com/user-attachments/assets/c8924a90-157f-459c-873f-570a99a1ec32)



    #1. Image encoder
    x = Conv2d(kernel_size=16, stride=16)(img)  #embedding to patches [B,H/16,W/16,C]
    pos_embed = Parameter(zeros(1, H // 16, W // 16, C)  #init absolute positional embedding
    #image_embed: final image embedding [B,embed,H/16,W/16]
    #high_res_feats: intermediate layers embeddings [[B,embed,H/8,W/8],[[B,embed,H/4,W/4]]
    [image_embed,high_res_feats] = ViT(x + pos_embed)  
    

    #2. Prompt encoder
    point_embeddings = [Embedding(1, embed_dim),Embedding(1, embed_dim),
                        Embedding(1, embed_dim),Embedding(1, embed_dim)]
    pos_embed_gaussian_matrix = randn(2, embed_dim/2)

    ## 2.1 encode points 
    # points[B,cls,2,2], format like [[[[x, y],[0,0]]]] unnormalized within [1024,1024]
    # cls is the number of labels, each label include 1 point
    # labels[B,cls,2], format like [[[1,-1]]]
    points = (points + 0.5) / 1024  # Shift to center of pixel and normalize to [0,1]
    points = points * 2 - 1 # Shift to [-1,1]
    points_embed = points @ pos_embed_gaussian_matrix #[cls,2,embed_dim/2]
    points_embed = cat([sin(2*pi*points_embed), cos(2*pi*points_embed)],
                       dim=-1) #[B,cls,2,embed_dim]
    points_embed[labels==-1] = 0.0 #reset padding point to zero
    points_embed[labels==1] += point_embeddings[1] #[B,cls,2,embed_dim]
    
    ## 2.2 encode box
    # box[B,cls,2,2], format like [[[[x, y],[x',y']]]] unnormalized within [1024,1024]
    # cls is the number of labels
    box = (box + 0.5) / 1024  # Shift to center of pixel and normalize to [0,1]
    box = box * 2 - 1 # Shift to [-1,1]
    box_embed = box @ pos_embed_gaussian_matrix #[B,cls,2,embed_dim/2]
    box_embed = cat([sin(2*pi*box_embed), cos(2*pi*box_embed)],
                       dim=-1) #[B,cls,2,embed_dim]
    box_embed[:,0,:] += point_embeddings[2]
    box_embed[:,1,:] += point_embeddings[3] #[B,cls,2,embed_dim]

    ## 2.3 encode mask

    #[B,cls,n,embed_dim], where where n=0,2,4, depends on points/box/mask avaliablity
    sparse_prompt_embed = cat([points_embed, box_embed], dim=1) 
    #[B,cls,embed_dim,H/16,W/16]
    dense_prompt_embed =     #

    # 3. Mask decoder
    #sparse_prompt_embed with shape [B,cls,n,embed_dim], n=0/2/4
    #dense_prompt_embed with shape [B,cls,embed_dim,H/16,W/16]
    #image_embed with shape [B,embed_dim,H/16,W/16]
    #cls is the number of labels
    
    num_multimask_outputs = 3  #each point predict 3 to resolve ambigurity
    iou_token = Embedding(1, embed_dim)
    mask_tokens = Embedding(num_multimask_outputs + 1, embed_dim)
    obj_score_token = Embedding(1, embed_dim)
    output_tokens = cat([obj_score_token,iou_token,mask_tokens], 
                        dim=0).reshape(..) #[B,cls, 6, embed_dim]         
    tokens = cat((output_tokens, sparse_prompt_embed), dim=1) #[B,cls,6+n,embed_dim]  
    
    # Expand per-image to be per-mask, [B,embed_dim,H/16,W/16]->[B*cls,embed_dim,
    # H/16,W/16]. thus in each image, it has [cls,embed_dim,H/16,W/16]
    src = repeat_interleave(image_embed,cls,dim=0)  #[B*cls,embed_dim,H/16,W/16]
    src = src + dense_prompt_embed
