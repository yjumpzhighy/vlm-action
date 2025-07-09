# Grounded Language-Image Pre-training
CLIP, relates the image content and image caption, i.e, treat the image as one object.
GLIP, relates image parts with phase grounding, i.e, treat the image as multi objects.

<img src="https://github.com/user-attachments/assets/4f52df76-7f62-41c1-b487-d75df8836963" width="400" height="600">



Overview:  
- fusion module:    
  - image feature and text feature cross-attn (below),text prompt would impact image encoder in early stage.     
    q = image_feats * Wq     
    k = text_feats * Wk      
    attn = q * k'     
    image_feats = softmax(attn) * (image_feats * Wv)      
    ext_feats = softmax(attn') * (text_feats * Wv)      
      
- word-region allignment score      
  - image features [M,C], text features [N,C]     
    S = dot(image features, text features), get [M,N]     
  - in each row, do softmat and convert to probabilities. thus S(i,j) refers the probability of image i-th region
    belongs to class j-th text token.
    for example, text = "Detect the person, the car, and the traffic light.". thus calcualte the probability of
    image region with tokens "person", "car", "traffic light".

- bbox regression
  - image features [h,w,c] and pre-defined anchors proposal
  - regression bbox offset
  
  

