Masked Autoencoders Are Scalable Vision Learners

![image](https://github.com/user-attachments/assets/26360490-b467-4f13-9a2d-eebd06498c52)

    # 1. encoder
    # 1.1 patch embedding, img[b,3,H,W]->x[b,H/p*W/p,C]
    x = Conv2d(3, C, kernel_size=patch_size, stride=patch_size)(img).flatten(2).transpose(1, 2)
    # 1.2 patch 2-D pos embedding
    #sin(x_idx / 10000^(i/(C//4))), cos(y_idx / 10000^(i/(C//4))), i~[0, 4//2], x_idx~[0,h/p], y_idx~[0,w/p]
    omega = 1. / 10000**(range(C//4)/(C/4))  # (C/4,)
    pos = meshgrid(arange(W/p), arange(H/p)) # (2,H/p,C/p)
    #pos be like [[[0,0,0],[1,1,1],[2,2,2]],[[0,1,2],[0,1,2],[0,1,2]]
    out_h = np.einsum('m,d->md', pos[0].reshape(-1), omega)  #(H/p*W/p, C/4)
    embed_h = concat([sin(out_h), cos(out_h)], dim=-1) #(H/p*W/p, C/2)
    out_w = np.einsum('m,d->md', pos[1].reshape(-1), omega)  #(H/p*W/p, C/4)
    embed_w = concat([sin(out_w), cos(out_w)], dim=-1) #(H/p*W/p, C/2)
    embed_pos = concat([embed_h, embed_w],dim=-1) # (H/p*W/p, C)
    embed_pos = concat([np.zeros([1, C]), embed_pos], axis=0) #if cls token prefix, # (H/p*W/p+1, C)
    x = x + embed_pos[:,1:,:]
    # 1.3 random masking
    ids_selected = argsort(rand(x.shape[0],x.shape[1])
    mask = argsort(rand(x.shape[0],x.shape[1])
    ids_restore =argsort(ids_selected)  #recover the original order, (b,H/p*W/p) 
    ids_selected = ids_selected[:, :(ids_selected.shape[1]*0.75)]  #select 25% , (b,0.25*H/p*W/p)
    x = gather(x, ids_selected) #(b,0.25*H/p*W/p,C)
    mask = gather(mask, ids_restore)
    # 1.4 add cls token
    cls_tokens = parameter(1,1,C) + embed_pos[:,:1,:] #init
    x = concat([cls_tokens, x]) #(b,0.25*H/p*W/p+1,C)
    # 1.5 vit 
    x = vit(x) #(b,0.25*H/p*W/p+1,C)

    # 2. decoder
    # 2.1 add mask tokens, sum up masked tokens and recover to full patches with original order
    mask_token = parameter(1,1,C)  #init
    mask_tokens = mask_token.repeat(b,0.75*H/p*W/p,C) 
    x_ = cat([x[:,1:,:], mask_tokens], dim=1) 
    x_ = gather(x_, ids_restore) #(b,H/p*W/p,C)
    # 2.2 add cls token
    x = cat([x[:,:1,:], x_]) #(b,1+H/p*W/p,C)
    # 2.3 pos embedding
    x = x + embed_pos
    # 2.4 vit
    pred = vit(x)[:,1:,:] #(b,H/p*W/p,p*p*3)
    
    # 3. loss
    tgt = reshape(img) #(b,3,H,W) -> (b,H/p*W/p,p*p*3)
    loss = ((tgt - pred) ** 2)[mask]
 
