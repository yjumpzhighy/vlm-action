
# 1. Deformable convolution
Compared to classic convolution, which the conv kernel usually with fixed shape, like 3x3 or 7x7. 
On contrast, the deformable convolution would shift the kernel's each point a little bit on x&y.
![image](https://github.com/user-attachments/assets/bebd41e0-82d9-4190-8c25-caf80028e511)

Overall deformable-conv process:
![image](https://github.com/user-attachments/assets/7a3ed141-0883-46d0-a8b5-f4aa67136c93)
1) input (B,H,W,C)
2) conv on input feats, get xy shift (B,H,W,N*2), N is the kernel volume size, like 3*3
3) on input feats's each channel, apply the same shift. At each channel's each location, it get
   N variations. thus, the totoal shifted feats (B,C,H,W,N)
4) reshape shifted feats to (B,C,H*k,W*k), where N=k*k
5) conv(C,C',stride=k) finall get (B,C',H,W)



# 2. Deformable attention
## 2.1 single scale



## 2.2 multi scales

