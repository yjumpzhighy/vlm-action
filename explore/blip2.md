# Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models

## Conclude     
1. Use uni modality pretrained models into multi-modality training leading to catastrophic         
   forgetting. Thus frozen uni modality models may solve this issue.  
2. Use Q-former to align frozen pretrained models         


## Q-Former       
<img src="https://github.com/user-attachments/assets/08af54b3-b920-4d5e-9023-6032e2c73bbc" width="400" height="600">   

1. query encoder uses self-attn on learnable query embeddings ([max_num_query_tokens, hidden_dim])     
2. query encoder cross-attn on image embedding (from image encoder) with query embedding         
3. text encoder share the same self-attn with query encoder, apply strong interact between query&text encoders
```
clss QFormer:
   self.cross_attn = CrossAttn(..)
   self.query_ffn = FFN(..)
   self.text_ffn = FFN(..)
   self.shared_self_attn = SelfAttn(..)

   # in forward(query_embedding, text_embedding, image_embedding):
   task_mask = task_mask(..) #details see below
   embedding = concat(query_embedding, text_embedding)
   embedding = shared_self_attn(embedding, mask=task_mask)
   query_embedding, text_embedding = embedding[:max_num_query_tokens,], embedding[max_num_query_tokens:,]
   query_embedding = self.cross_attn(q=query_embedding,k=image_embedding,v=image_embedding)
   query_embedding = self.query_ffn(query_embedding)
   text_embedding = self.text_ffn(text_embedding)
```
   
5. Image-Text-Matching
   
   
7. Image-Text-Constrstive Learning
8. Image-Grounded Text Generation

## Train
1. In first stage, froze Image Encoder and train Q-Former，so that learnable Queries embedding able to extrac image
   feature, and convert to text-feature-grounded image feature. i.e, converted image features and text features
   would be strong aligned.
3. 
   

