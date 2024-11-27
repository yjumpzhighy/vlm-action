# MViTv2: Improved Multiscale Vision Transformers for Classification and Detection

![image](https://github.com/user-attachments/assets/f60f79f0-6db5-4e53-8698-a54b8e6a0e6c)

### 1.decomposed relative position embedding
##### Shift-invariance in vision: 
objects can be at various positions in image. shift-invariant will correctly 
recognize objects regardless where it is located in the image. 
For example, [0,0,1,1,0,0,1,1] perform maxpool(k=2,s=2) get [0,1,0,1]. but if 
shift left one col [0,1,1,0,0,1,1,0] would get [1,1,1,1], which is not desired.

##### Position embedding issue
ViT divide into non-overlapping patches as tokens, introduces absolute position of 
patches (in a sequence), rather than their relative spatial relationships, which
ingores shift-invariance principle in vision. (BTW, CNN uses pooling and kernel 
inherently provide translation invariance, since it is relative, not absolute pos).

Thus, two patches interaction would change depending on their absolute position in images even if their relative positions stay unchanged.

    # pooled q,k,v = [b,head,l,head_dim],where head_dim=c/head, l==h*w,
    #  h==H/p/s, w==W/p/s (s downsample stride)
    attn = q @ k' #[b,head,l,l]
    hq,hk = h
    wq,wk = w
    
    #[h,h], with value [[h-1,   h-2, ..., 1, 0]
    #                   [h,     h-1, ..., 2, 1]
    #                   [...,   ..., ..., ., .]
    #                   [2(h-1),2h-1,..., ., h-1]]
    dist_h = (arange(hq)[:, None] - arange(hk)[None, :] + (hk-1) 
    #[w,w], with value [[w-1,   w-2, ..., 1, 0]
    #                   [w,     w-1, ..., 2, 1]
    #                   [...,   ..., ..., ., .]
    #                   [2(w-1),2w-1,..., ., w-1]]
    dist_w = (arange(wq)[:, None] - arange(wk)[None, :] + (wk-1)
    # what does 2(w-1) mean here? 
    # for two patches i and j, their w bias ranges [-(w-1),(w-1)]. after
    # offset to non-negative, it ranges [0, 2(w-1)]
    
        
    rel_pos_h = Parameter(zeros(2*max(wq,wk)-1, head_dim)) #[2w-1,head_dim]
    rel_pos_w = Parameter(zeros(2*max(wq,wk)-1, head_dim)) #[2w-1,head_dim]
    
    Rh = rel_pos_h[dist_h.long()] #[h,h,head_dim]
    Rw = rel_pos_w[dist_w.long()] #[w,w,head_dim] 
    Rq = q.reshape(b, head, hq, wq, head_dim) #[b,head,h,w,head_dim]
    rel_h = Rq @ Rh'  #[b,head,h,w,h]
    rel_w = Rq @ Rw'  #[b,head,h,w,w]

    attn = (attn.view(b,head,h,w,h,w)
        + rel_h[:, :, :, :, :, None]
        + rel_w[:, :, :, :, None, :]).view(b,head,l,l)
    

    
