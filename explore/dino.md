# DETR with Improved deNoising anchOr boxes

<img src="https://github.com/user-attachments/assets/144eff43-44fb-4257-b90f-adcff3df9963" width="400" height="600">   

## Model          
1. Stem        
  a. ResNet output multi-scale features, [[c,h1,w1],[c,h2,w2],...]        
  b. fixed sinusoidal positional embedding for feature maps, get[[c,h1,w1],[c,h2,w2],...]           
  c. learne scale positional embedding, indicating which scale level for feature maps, get              
     [[c],[c],...]. It is internally nn.Embedding lookup table.               
  d. (scale positional embedding) + (multi scale features), flatten to features[c[[c,h1*w1],[c,h2*w2],...]                  
     and concat to [c,h1xw1+h2xw2+..]             
  e. sinusoidal positional embedding flatten and concat to positional embedding[c,h1xw1+h2xw2+..]                 
2. Encoder          
   ```
   crossattn(q=features+positional embedding,
             k=features+positional embedding,
             v=features)
   ```
   encoded features [c,N], N is num of tokens             
3. Mixed Query Selection             
   a. ffn() to convert encoded features to (5, N). 5 is [cx,cy,w,h,objectness_score]               
   b. select topk indexs based on scores, and gather get position query[c, K]. which represent anchor position info              
   c. learnable content query with [c,K], adding with position query, get mixed query[c,K]             
4. Decoder                 
   ```
   crossattn(q=mixed query, k=encoded features, v=encoded features)
   ```
   get decoded query [c,K]        


   
