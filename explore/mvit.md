# Multiscale Vision Transformers


![image](https://github.com/user-attachments/assets/b857f861-759a-4837-937b-73f0d4d532c5)
1. MHA -> MHPA, with pooling added

        # after patch embed, input x[b,l,c], l==h*w, h==H/p, w==W/p
        q,k,v = Linear(c,3c')(x).reshape(b,l,3,heads,c'//head).permute(2,0,3,1,4).chunk(3)
        # q pooling
        q_pool = Conv2d(c'//head,c'//head,kernel=(1,1))
        q = q.reshape(b*head,h,w,c'//head).permute(0, 3, 1, 2)
    
    
