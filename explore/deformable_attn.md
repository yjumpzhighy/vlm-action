
# 1. Deformable convolution  
<img src="https://github.com/user-attachments/assets/bebd41e0-82d9-4190-8c25-caf80028e511" width="400" height="200">  

Compared to classic convolution, which the conv kernel usually with fixed shape, like 3x3 or 7x7.  
On contrast, the deformable convolution would shift the kernel's each point a little bit on x&y.  

Overall deformable-conv process:  
<img src="https://github.com/user-attachments/assets/7a3ed141-0883-46d0-a8b5-f4aa67136c93" width="300" height="300">  
1) input (B,H,W,C)  
2) conv on input feats, get xy shift (B,H,W,N*2), N is the kernel volume size, like 3*3  
3) on input feats's each channel, apply the same shift. At each channel's each location, it get  
   N variations. thus, the totoal shifted feats (B,C,H,W,N)  
4) reshape shifted feats to (B,C,H*k,W*k), where N=k*k  
5) conv(C,C',stride=k) finall get (B,C',H,W)  



# 2. Deformable attention  
<img src="https://github.com/user-attachments/assets/6d66cfce-0cd2-41fb-a84e-bc67c55dafa6" width="300" height="300">       
Each token not necessary to get attention with all tokens, as most of them are actually unrelated.  
Instead, on each directions, only focus on 4 points to gather most important info.  

## 2.1 single scale  
<img src="https://github.com/user-attachments/assets/87281018-b892-45f8-a398-8e45b2c91bea" width="300" height="300">     

1) input tokens [C,h,w], flatten to [h*w,C]. each token refers to a location in backbone output feats map.   
2) for each token:   
       a. [1,C] split to 8 heads, each head feat [1,C/8], 8 indicates surrounding 8 directions.      
       b. each head feat generate offest [1,4*2] on 8 directions, which presents 4 locations xy offset on input tokens.     
          with bilinear-interpolation, get values from input tokens and project to [1,4,C/8]    
       c. each head feat generate learn attn weights [1,4]    
       d. weights * values, and sum to [1,C/8]    
       e. all heads feats concat to [1,C]    



## 2.2 multi scales  
<img src="https://github.com/user-attachments/assets/2835815d-d885-4c2e-946d-805239559fab" width="300" height="300">  

