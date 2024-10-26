
# VLM-Action
Brief tutorial vision-launguage large model related works: modeling, finetune, parallel training, quantize, deployment, etc.


## 1. Modeling
### 1.1 Diffuser
Diffusion model is usually used for high-quality image generation. The principle is by add noise during 
training and denoise in generation process. In this section, several STOA diffusion models, like vae, 
ddpm, ddim, ldm, DiT, etc.
[deepspeed overview doc] (https://github.com/yzy-jumphigh/vlm-action/blob/main/models/README.md)  

    #run autoencoder
    python models/vae.py
    #run ddpm
    python models/ddpm.py
    #run ddim
    python models/ddim.py
    #run ldm, ddp training used
    CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 models/stable_diffuser.py


## 2. Training
### 2.1 Deepspeed
Talk about how to utilize zero on large model training, and, how to apply deepspeed within your train pipeline.    
[deepspeed overview doc](https://github.com/yzy-jumphigh/vlm-action/blob/main/training/README.md)   

    #single gpu
    deepspeed --num_gpus 1 training/llama2_clm_sft_lora_with_zero.py
    #multi gpus
    deepspeed training/llama2_clm_sft_lora_with_zero.py

How effective deepspeed would be? 
[Deepspeed memory quantitative analysis](https://github.com/yzy-jumphigh/vlm-action/blob/main/training/README.md)


## 3. Finetune
Cover common large model finetune strategies prefix, prompt, p-tuning, p-tuning v2, lora.     
[finetune methods overivew](https://github.com/yzy-jumphigh/vlm-action/blob/main/finetune/README.md)  

    #lora
    python finetune/llama2_lora.py

    #prefix
    python finetune/llama2_prefix.py

    #ptune-v2
    python finetune/llama2_ptune_v2.py

    #prompt
    python finetune/llama2_prompt.py

How effective LORA would be? 
[LORA memory quantitative analysis](https://github.com/yzy-jumphigh/vlm-action/blob/main/training/README.md)
    


## 4. RAG
rag is able to alleviate LLMs hallucination issue, while benefits from data security and avoide finetune process.  
[rag overview doc](https://github.com/yzy-jumphigh/vlm-action/blob/main/rag/README.md)   

    # raw rag
    python rag/raw_rag.py

    # child-parents documents retrieval rag
    python rag/advanced_rag_child_parent_retrieval.py

    # reranker rag
    python rag/advanced_rag_rerank.py

    # contextual compression rag 
    python rag/advanced_rag_compression.py

    # merged retrievers rag
    python rag/advanced_rag_merge_retrievers.py

## 5. Explore papers
5.1 masked autoencoders are scalable vision learners

![image](https://github.com/user-attachments/assets/26360490-b467-4f13-9a2d-eebd06498c52)

    # 1. encoder
    # 1.1 patch embedding, [b,3,H,W]->[b,H/p,W/p,C]
    x = Conv2d(3, C, kernel_size=patch_size, stride=patch_size).flatten(2).transpose(1, 2)
    # 1.2 patch 2-D pos embedding
    #sin(x_idx / 10000^(i/(C//4))), cos(y_idx / 10000^(i/(C//4))), i~[0, 4//2], x_idx~[0,h/p], y_idx~[0,w/p]
    omega = 1. / 10000**(range(C//4)/(C/4))  # (C/4,)
    pos = meshgrid(arange(W/p), arange(H/p)) # (2,H/p,C/p)
    out_h = np.einsum('m,d->md', pos[0].reshape(-1), omega)  #(H/p*W/p, C/4)
    embed_h = concat([sin(out_h), cos(out_h)], dim=-1) #(H/p*W/p, C/2)
    out_w = np.einsum('m,d->md', pos[1].reshape(-1), omega)  #(H/p*W/p, C/4)
    embed_w = concat([sin(out_w), cos(out_w)], dim=-1) #(H/p*W/p, C/2)
    embed_pos = concat([embed_h, embed_w],dim=-1) # (H/p*W/p, C)
    embed_pos = concat([np.zeros([1, C]), embed_pos], axis=0) #if cls token prefix, # (H/p*W/p+1, C)
    
    
    
    




