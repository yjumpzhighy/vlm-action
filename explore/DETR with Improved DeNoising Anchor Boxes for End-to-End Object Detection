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
   encoded features [N,c], N is num of tokens             
3. Mixed Query Selection             
   a. ffn() to convert encoded features to (N, 5). 5 is [cx,cy,w,h,objectness_score]               
   b. select topk indexs based on scores, and gather get position query[K,c]. which represent anchor position info              
   c. learnable content query [K,c], adding with position query, get mixed query[K,c]

4. Contrstive Denoising query
   a.add small noise on gt_bbox, get pos_bbox (small noise on box and keep same cls)
     add heavy noise on gt_bbox, get neg_bbox (heavy noise on bbox but change cls)
   b.postional encoding and label embedding to convert to pos_embedding/neg_embedding
   c.concat(mixed_query, pos_embedding, neg_embedding), get  all query[K+P+N,c]
5. Decoder                 
   ```
   crossattn(q=all query, k=encoded features, v=encoded features)         
   ```
   get decoded query [K+P+N,c]       
        
## Loss
```
detection = decoded_query[:K]      
pos_query_output = decoded_query[K:K+P]       
neg_query_output = decoded_query[:K+P] 

#refer to DETR
match_loss = hungarian_loss(detection, gt)

#contrastive loss, positive queries should match correctly
for i in zip(pos_bbox, pos_query_output):
    pred_box = pos_query_output[i][:4]
    pred_cls = pos_query_output[i][5]
    gt_box = pos_bbox[i][:4]
    gt_cls = pos_bbox[i][5]
    denoise_loss += cross_entropy(pred_cls,gt_cls) + l1(pred_box,gt_box)

#contrastive loss, neg queries should be corrected
for i in zip(neg_bbox, neg_query_output):
    pred_box = neg_query_output[i][:4]
    pred_cls = negs_query_output[i][5]
    noise_box = neg_bbox[i][:4]
    noise_cls = neg_bbox[i][5]
    # find correct in gt for negtive query
    noise_corrected_box, noise_corrected_cls = find_closest_gt(noise_box)
    denoise_loss += cross_entropy(pred_cls,noise_corrected_cls) + l1(pred_box,noise_corrected_box)
```
   
## Look forward twice
Generate twice noise query and run twice decoder accordingly. then compare the loss with           
gt and choose the better once for actual loss.            

```
 noisy_queries_A = Contrstive Denoising (gt_boxes, gt_labels, pattern='A')       
 predictions_A = decoder(encoded_features, noisy_queries_A)   
 noisy_queries_B = Contrstive Denoising (gt_boxes, gt_labels, pattern='B') 
 predictions_B = decoder(encoded_features, noisy_queries_B)   
 selected_prediction = select_better_of_two(predictions_A, predictions_B, gt_boxes)    
```


 
## Conclude:        
Instead of use gt embedding directly, construct positive (small noise) and negative (large noise) for             
constrastive learning.            
noise -> faster coverage             
constrstive -> distinguish similar objects                 
