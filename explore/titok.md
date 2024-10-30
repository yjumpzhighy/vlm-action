# An Image is Worth 32 Tokens for Reconstruction and Generation
![image](https://github.com/user-attachments/assets/c3ced2f4-69f3-445b-8f2c-8a306025fe51)

2D tokenizations have inherent redundancies present in images, where adjacent regions frequently display similarities


    #1. encode
    class_embedding_encode = Parameter(1, C) #init
    pos_embedding_encode = Parameter(h*w+1, C) #init
    latent_tokens = Parameter(128, C) #init, 128 tokens tokenizer
    latent_pos_embedding_encode = Parameter(128, C) #init
    
    x = Conv2d(kernel=patch_size).reshape  #[b,3,H,W] -> [b,h*w,C]
    x = concat(x, class_embedding_encode) + pos_embedding_encode #[b, h*w+1, C]
    x_latent = latent_tokens + latent_pos_embedding_encode
    x = concat(x, x_latent) #[b, h*w+1+128, C]
    x = encoder(x) #[b, h*w+1+128, C]
    x = x[:, 1+h*w:]  #[b, 128, C], only latent tokens embedding required
    
    
    #2. tokenize
    x = x.reshape(..)  #[b, C, 1, 128], to match quantizer data format
    z, z_ids = VectorQuantizer(x) #[b, C, 1, 128],[b,1,128]
    
    #3. decode
    mask_tokens = Parameter(1, 1, C) #init
    class_embedding_decode = Parameter(1, C) #init
    pos_embedding_decode = Parameter(h*w+1, C) #init
    latent_pos_embedding_decode = Parameter(128, C) #init
    
    mask = concat(class_embedding_decode.repeat(b,1,C), mask_tokens.repeat(b,h*w,C)) + pos_embedding_decode #[b,1+h*w, C]
    z = reshape(..) #[b, 128, C]
    z = z + latent_pos_embedding_decode
    z = concat(mask, z) #[b, 1+h*w+128, C]
    z = decoder(z)  #[b,1+h*w+128,C]
    z = z[:, 1:1+h*w] #[b,h*w,C], remove cls embed and latent token embed

    #4. reconstruct
    z = z.reshape(..) #[b,C,h,w]
    z = Conv2d(C,h*w*3,kernel=1)(z).rearrange(..) #[b,3,H,W]
    
    
    
