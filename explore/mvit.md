# Multiscale Vision Transformers


![image](https://github.com/user-attachments/assets/b857f861-759a-4837-937b-73f0d4d532c5)
1. MHA -> MHPA, with pooling added
Compared to MHA, it performed 1) pooling on q,k,v with channel mixing 2) residual skip x+q

        # after patch embed, input x[b,l,c], l==h*w, h==H/p/s,
        # w==W/p/s(s downsample stride)
   
        q,k,v = Linear(c,3c')(x).reshape(b,l,3,heads,c'//head).permute(2,0,3,1,4).chunk(3)
        # q pooling
        q_pool = Conv2d(c'//head,c'//head,kernel=(1,1))
        q = q.reshape(b*head,h,w,c'//head).permute(0,3,1,2) #[b*head,c'/head,h,w]
        q = q_pool(q)   #1x1 channel mixing only
        q = q.rearrange(b,head,h*w,c'/head) #[b,head,l,c'/head]
        # k pooling
        k_pool = Conv2d(c'//head,c'//head,kernel=(1,1))
        k = k.reshape(b*head,h,w,c'//head).permute(0,3,1,2) #[b*head,c'/head,h,w]
        k = k_pool(k)   #1x1 channel mixing only
        k = k.rearrange(b,head,h*w,c'/head) #[b,head,l,c'/head]
        # v pooling
        v_pool = Conv2d(c'//head,c'//head,kernel=(1,1))
        v = v.reshape(b*head,h,w,c'//head).permute(0,3,1,2) #[b*head,c'/head,h,w]
        v = v_pool(v)   #1x1 channel mixing only
        v = v.rearrange(b,head,h*w,c'/head) #[b,head,l,c'/head]
    
        attn = q @ k'
        x = attn.softmax(dim=-1) @ v [b,head,l,c'/head]
        x = x.rearrage(b,l,c')
        
