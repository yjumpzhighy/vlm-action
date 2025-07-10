# Bootstrapping language-image pre-training for unified vision-language understanding and generation

<img src="https://github.com/user-attachments/assets/ff6bdc5c-bfdb-45b5-ac23-a49d686329ce" width="400" height="600">

## MED
1. Image Encoder
   vit based image encoder, similar with CLIP image encoder. specially, add [CLS] token
   to represent as global image features
2. Text Encoder
   based on BERT, bi-self attn, similar with CLIP text encoder. specially, add [CLS] token
   before text tokens.
3. Image grounded text encoder
   shared FFN and Bi-Self-Attn layers with text encoder, but also include cross attention
   layers to learn the image features (from image encoder) with text features. Specially,
   add [Encoder] token before text tokens to annotate a new task.
5. Image grounded text decoder
   shared Cross-Attn and FFN layers with image-grounded-text-encoder, but used a Causal-Self-
   Attn layers. Specially, add [Decoder] token before text tokens to annotate a new task.

```python pseudo code
class SharedEmbedding:
   self.shared_embedding = Embedding(..)
class SharedFFN:
   self.shared_ffn = [FFN(..) for i in range(n)]
class SharedBiSelfAttn:
   self.shared_bi_self_attn = [SelfAttn(..) for i in range(n)]
class SharedCrossAttn:
   self.shared_cross_attn = [CrossAttn(..)  for i in range(n)]

# unified and shared layers
shared_embedding = SharedEmbedding
shared_ffn = SharedFFN
shared_bi_self_attn = SharedBiSelfAttn
shared_cross_self_attn = SharedCrossAttn

class TextEncoder:
   self.embedding = shared_embedding
   self.bi_self_attns = shared_bi_self_attn
   self.ffn = shared_ffn

class ImageGroundedTextEncoder:
   self.embedding = shared_embedding
   self.bi_self_attns = shared_bi_self_attn
   self.cross_attns = shared_cross_self_attn
   self.ffn = shared_ffn

class ImageGroundedTextDecoder:
   self.embedding = shared_embedding
   self.causal_self_attns = [CausalSelfAttn(..) for i in range(n)]
   self.cross_attns = shared_cross_self_attn
   self.ffn = shared_ffn
```

## Tasks       
1. Image Text Contrastive Loss     
   Align image and text representations, with contrastive learning, to pull positive (image,text)      
   pairs closer together in image-text embedding space and pushing negative pairs apart.        
   a) generate M*N score matrix S, S[i,j] is cos-similarity of image feats[i] and text feats[j].       
   b) S[i,i] refers to the correctly matched postive pair, while S[i,j] referst to negative pair.        
   c) Image-to-Text cross entropy       
      ```
      labels = torch.arange(batchsize)
      ce = cross_entropy(S,labels)
      ```
   d) Text-to-Image cross entropy     
      ```
      S' = S.T #transpose to text-to-image perspective
      labels = torch.arange(batchsize)
      ce = cross_entropy(S', labels)
      ```
2. Image Text Matching Loss
   Unlike the Image-Text Contrastive which focuses on aligning global representations, it determines
   capture fine-grained alignment whether image and text pair actually matches.
   a) explicitly constructs negative pairs. from the ITC calculated S map, select texts that incorrectly    
      predicted as highly similar to that image (but are actually mismatched).
   b) feedinto binary classifier to predict match/dismatch
   c) get binary entropy loss
3. Language Modeling Loss
   cross-entropy on input text [[Decode],t0,t1,t2..,tn] with prediction [p1,p2,p3,...tn+1]


## Captioner-Filter
<img src="https://github.com/user-attachments/assets/c6ac7c5e-7b15-4cbc-9308-4e9471a4d6b2" width="400" height="600">
1. MED fintune training on COCO
2. captioner (based image-grouned text decoder), generate new captions for web images
3. filter (based image-grounded text encoder), use ITC and ITM to filter-out dismatching captions
4. new dataset + COCO dataset, retrain the MED.


## Inference     
   1. input image feedinto image-encoder, get image feature tokens       
   2. image feature tokens , text input start with [[Decoder]], feedinto image-grounded-text-decoder      
      to predict next token      
   3. in max_iterations, add previously predicted tokens to text input like [[Decode],[..],[..]],     
      and feedinto together with image feature tokens, to image-grouded-text-decoder     



## Limitations           
1. each image treated as one object           
2. inconstant info flow due to shared layers within modules        
   
