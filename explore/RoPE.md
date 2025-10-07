
# Absolute pos embedding
1.对图片mask(b,h,w), 在x,y 累加，得到x_embed, y_embed (b,h,w)     

<img src="https://github.com/user-attachments/assets/f01fb555-5f66-4e4b-8952-e405e5207c36" width="400" height="600">

2. 对x, 第i维(0~C/2)，偶数sin(x_embed/10000^(2i/128))，奇数cos(x_embed/10000^(2i/128)), 叠加后有pos_x (b,h,w,C/2)   
3. 对y, 第i维(0~C/2)，偶数sin(y_embed/10000^(2i/128))，奇数cos(y_embed/10000^(2i/128)), 叠加后有pos_y (b,h,w,C/2)

<img src="https://github.com/user-attachments/assets/d592da8f-3745-4b66-8db6-9b32828e79e3" width="400" height="600">

4. 最后在叠加到一起pos, (b,h,w,C)     
5. q=q+pos, k=k+pos, attn = sm(q@k')    

# Relative pos embedding
