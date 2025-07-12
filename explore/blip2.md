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
   query_embedding = self.query_ffn(query_embedding)  #query encoder output
   text_embedding = self.text_ffn(text_embedding)  #text encoder output
```
   
5. Image-Text-Matching     
   In ITM, during the self-attention layer, the query and text can interact without any limit. Used to          
   match image and text info.        
   ```
   task_mask(..):
      return None
   ```
   Project output query_embedding to binary score, and calcualte loss
   ```
   logits = nn.Linear(hidden_dim,2)(query_embedding)
   loss = bianryentropy(logits, labels)
   ```
   
7. Image-Text-Constrstive Learning          
   Align query and text representations, with contrastive learning, to pull positive (query,text)       
   pairs closer together and pushing negative pairs apart.           
   As is contrastive, peeking between query and text not allowed         
   ```
   task_mask(..):
      # query_embedding to query_embedding no limit
      # query_embedding to text_embedding disabled
      # text_embedding to text_embedding no limit
      # text_embedding to query_embedding disabled           
   ```
   a) generate M*N score matrix S, S[i,j] is cos-similarity of query_embedding[i] and text_embedding[j].          
   b) S[i,i] refers to the correctly matched postive pair, while S[i,j] referst to negative pair.      
   c) Query-to-Text cross entropy              
      labels = torch.arange(batchsize)         
      ce = cross_entropy(S,labels)             
   d) Text-to-Query cross entropy           
      S' = S.T #transpose to text-to-image perspective              
      labels = torch.arange(batchsize)             
      ce = cross_entropy(S', labels)            
   e) self-attn mask                 
   ```
   task_mask(..):
      # query_embedding to query_embedding no limit
      # query_embedding to text_embedding disabled
      # text_embedding to text_embedding no limit
      # text_embedding to query_embedding disabled
9. Image-Grounded Text Generation        
   Given image conditioning, force to generate related text      
   ```
   task_mask(..):
      # query embedding part, no mask
      # text embedding part, use casual mask to avoid front peeking backwards
   loss = cross-entropy (text_embedding, text_input_labels)  #general LM loss   
   ```

## Train
1. In first stage, froze Image Encoder, Train Q-Former (ITG+ITC+ITM). so that learnable Queries embedding able to     
   extrac image feature, and convert to text-feature-grounded image feature. i.e, converted image features and text features      
   would be strong aligned.          
2. In second stage     
   <img src="https://github.com/user-attachments/assets/ea06c7e3-5d36-4785-8d8e-6c486815e543" width="400" height="600">

   Froze LLM Decoder, Train Q-Former and FC. Here, must use the ITC self-attn mask, to avoid output include text info.     
   then with Q-Former output, the learned query embedding, used as LLM Decoder input and generate text, and force LM loss.    
   

   

