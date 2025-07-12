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



