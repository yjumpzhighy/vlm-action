
# Absolute pos embedding
1.对图片mask(b,h,w), 在x,y 累加，得到x_embed, y_embed (b,h,w)     

<img src="https://github.com/user-attachments/assets/f01fb555-5f66-4e4b-8952-e405e5207c36" width="400" height="600">

2. 对x, 第i维(0~C/2)，偶数sin(x_embed/10000^(2i/128))，奇数cos(x_embed/10000^(2i/128)), 叠加后有pos_x (b,h,w,C/2)   
3. 对y, 第i维(0~C/2)，偶数sin(y_embed/10000^(2i/128))，奇数cos(y_embed/10000^(2i/128)), 叠加后有pos_y (b,h,w,C/2)

<img src="https://github.com/user-attachments/assets/d592da8f-3745-4b66-8db6-9b32828e79e3" width="400" height="600">

4. 最后在叠加到一起pos, (b,h,w,C)     
5. q=q+pos, k=k+pos, attn = sm(q@k')    

# Relative pos embedding
若对q旋转角度m, 对k旋转角度n, 则其内积<q',k'>只与q,k,(m-n)相关,也即是和m和n的相对位置无关,只和相对位置相关.
                         
q: [B, L, Heads, C]
                        
freqs = 10000^(-2 * j / C)  #j:range(0, 2/C-1)                      
position_indices = range(0, L)                     
                 
cos_values = cos(rotation_angles)  # Shape [L, C/2]                   
sin_values = sin(rotation_angles)  # Shape [L, C/2]                  
                 
// Split C into even and odd indices                             
q_even = q[..., 0::2]  # Shape [B,L,Heas,C/2]: c0, c2, c4, ...                     
q_odd = q[..., 1::2]   # Shape [B,L,Heas,C/2]: c1, c3, c5, ...                              
                  
// Apply 2D rotation matrix to each pair:                                    
// [cos  -sin] [q_even]                     
// [sin   cos] [q_odd ]                      
q_even_rot = q_even * cos_values - q_odd * sin_values   # Shape[B,L,Heads,C/2]                
q_odd_rot = q_even * sin_values + q_odd * cos_values   # Shape[B,L,Heads,C/2]                
               
q = torch.stack([q_even_rot, q_odd_rot], dim=-1)                



    
