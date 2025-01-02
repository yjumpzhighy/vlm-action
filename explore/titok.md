# An Image is Worth 32 Tokens for Reconstruction and Generation
![image](https://github.com/user-attachments/assets/c3ced2f4-69f3-445b-8f2c-8a306025fe51)

Tokenizes images into 1D latent sequences:
- 2D tokenizations (like maskgit) have inherent redundancies. e.g, one patch may fall into two similar token ids in codebook
- set 1D latent embeddings as tokenzer. after concat with image patches embeddings, feedinto vit. In this way,
  the latent embeddings would learn the representation of image patches.
- Use the latent embeddings into vector quantizer codebook, to get patchs tokens embedding and tokens id.
- Concate mask tokens embedding with patchs tokens embedding and feedinto vit. In this way, the mask embeddings would be used
  as the representtion of patches, which then decode to a image directly.
- No pretrained vae needed, it uses regular encoder and decoder structure
  


      #1. encode
      class_embedding_encode = Parameter(1, C) #init
      pos_embedding_encode = Parameter(h*w+1, C) #init
      latent_tokens = Parameter(128, C) #init, 128 tokens tokenizer
      latent_pos_embedding_encode = Parameter(128, C) #init
      
      x = Conv2d(kernel=patch_size)(img).reshape  #[b,3,H,W] -> [b,h*w,C]
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
      # reconstruction = Conv2d(C,h*w*3,kernel=1)(z).rearrange(..) #[b,3,H,W], direclty predict pixels
      reconstruction = Conv2d(C,1024)(z) #[b,1024,h,w]

      #5. training
      #5.1 stage 1, focus on encoder training and 1D-128 latent tokens learning image representation. Use pretrained maskgit
      #with vqgan as teacher, to avoid train gan and other complex loss functions.
      proxy_codes = MaskGit.encoder(img) #[b,h,w], used maskgit encoder, returns patches token codebook indices
      loss = crossentropyloss(reconstruction.reshape(b,1024,-1), proxy_codes.reshape(b,-1))
      #5.2 stage 2, focus on finetune decoder with gan.
      loss = mse_loss(img, reconstructions)
      perceptual_loss = PerceptualLoss(img, reconstructions)
      NLayerDiscriminator().require_grad = false  # disable discriminator during use generator
      generator_loss = NLayerDiscriminator(reconstructions)
      backward(loss+perceptual_loss+generator_loss)

      NLayerDiscriminator().require_grad = true  # enable discriminator during training discriminator
      real = NLayerDiscriminator(img)
      fake = NLayerDiscriminator(reconstructions)
      discriminator_loss = hinge_d_loss(real, fake)
      backward(discriminator_lossl)
  
      
    

