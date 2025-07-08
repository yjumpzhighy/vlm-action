# Grounded Language-Image Pre-training
CLIP, relates the image content and image caption, i.e, treat the image as one object.
GLIP, relates image parts with phase grounding, i.e, treat the image as multi objects.

<img src="https://github.com/user-attachments/assets/98d7685a-9bf7-46c8-b5e9-8e9ea3764737" width="300" height="300">

Overview:
1.fusion module:
image feature and text feature cross-attn (below),text prompt would impact image encoder in early stage.
     q = image_feats * Wq
     k = text_feats * Wk
     attn = q * k'
     image_feats = softmax(attn) * (image_feats * Wv)
     text_feats = softmax(attn') * (text_feats * Wv)

2.word-region allignment score
image features [M,C], text features [N,C]

  
  
  

