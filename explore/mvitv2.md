# MViTv2: Improved Multiscale Vision Transformers for Classification and Detection

![image](https://github.com/user-attachments/assets/f60f79f0-6db5-4e53-8698-a54b8e6a0e6c)

### 1.decomposed relative position embedding
#### shift-invariance in vision:
objects can be at various positions in image. shift-invariant will correctly 
recognize objects regardless where it is located in the image. 
For example, [0,0,1,1,0,0,1,1] perform maxpool(k=2,s=2) get [0,1,0,1]. but if 
shift left one col [0,1,1,0,0,1,1,0] would get [1,1,1,1], which is not desired.

ViT divide into non-overlapping patches as tokens, introduces absolute position of 
patches (in a sequence), rather than their relative spatial relationships, which
ingores shift-invariance principle in vision. (BTW, CNN uses pooling and kernel 
inherently provide translation invariance)



