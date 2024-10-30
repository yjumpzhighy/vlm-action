MaskGIT: Masked Generative Image Transformer

![image](https://github.com/user-attachments/assets/04723d00-0575-4d59-ac5d-1d4ee4f19b3e)


use image tokenizer on image latents, have partial tokens masked and predict the missing tokens.
- encode raw image pixels into latents of patches
- use VectorQuantizer(codebook) to find minimul vector distance, each patch would be assigned a codebook index, which
  is image tokens
- image tokens get partial masked, and feed into Embeding and get tokens embedding
- token embeddings input into vit, and predict tokens id of each patch
- get cross entropy loss of on-hot image tokens and predicted tokens
- iteration decode. to follow the training task, still use predict masked patches method to gradually generate images
  in this way, it is much faster than token-by-token generative
- inpainting
- class conditional edition



      #1. tokenizer
      x = VQModel.encoder(x) #[b,3,H,W]->[b,C,h,w]
      #we don't need the quanize embedding, instead we only need its token ids. same like llm,
      #the token ids will be feedinto embedding later. However, the tokenizer codebook would
      #be trainable as well. 
      img_token = VQModel.quantizer(x) #[b,h,w]
    
      #2. mask 
      r = rand(img_token.size(0))
      mask = rand(size=img_token.size()) < r.view(img_token.size(0), 1, 1) #Sample the amount of tokens + localization to mask
      img_token[mask] = -1 #mask value
  
      #3. vit
      #concat visual tokens and class tokens
      #+1 for the mask of the viz token, +1 for mask of the class
      E = Embedding(codebook_size+1+nclass+1, C)  #init
      cls_token = labels.view(b, -1) + codebook_size + 1 #labels is classification gt, shift the class token by the amount of codebook
      input = torch.cat([img_token.view(img_token.size(0), -1), cls_token.view(img_token.size(0), -1)], -1)   # [b, h*w+1]
      token_emb = E(input) #[b, h*w+1, C]
      pos_emb = trunc_normal_(Parameter(zeros(1, h*w+1, C)), 0., 0.02) #use trainable parameters as pos embeding
      x = token_emb + pos_emb #[b, h*w+1, C]
      x = vit(x) #[b, h*w+1, C]
      logit = matmul(x, token_emb.weight.T)[:, :h*w, :codebook_size + 1] #[b, h*w, codebook_size + 1]
      
      #4. loss
      l = ce(logit.reshape(-1, codebook_size + 1), img_token.view(-1))

      #5. decode/sampling
      code = full((b, h, w), -1)
      labels = [...] #intend generated class with size b
      mask = ones(b, h*w)
      steps = 12  #total steps in generation process
      sche = (1 - linspace(1, 0, step)) * h*w  #valid patch tokens number after mask applied in each step
      sche[sche == 0] = 1   #in first step, at least predict one token. and last step predict h*w tokens
      for indice, t in enumerate(sche):
          pred_code = vit(code, labels)    
          f_mask = ... #with unpredicted tokens, select top t patches max confidence token id, keep for next step
          code[f_mask] = pred_code[f_mask]
        
  
  

  
