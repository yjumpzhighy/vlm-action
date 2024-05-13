
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






