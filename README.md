
# VLM-Action
Brief tutorial vision-launguage large model related works: modeling, finetune, parallel training, quantize, deployment, etc.


## 1. Modeling
(TO BE UPDATED)

## 2. Finetune
Cover common large model finetune strategies prefix, prompt, p-tuning, p-tuning v2, lora. 

2.1 [finetune methods overivew](https://github.com/yzy-jumphigh/vlm-action/blob/main/finetune/overview.md)  
2.2 run commands
    #lora
    python finetune/llama2_lora.py
    #prefix
    python finetune/llama2_prefix.py
    #ptune-v2
    python finetune/llama2_ptune_v2.py
    #prompt
    python finetune/llama2_prompt.py
2.3 [lora memory quantitative analysis](https://github.com/yzy-jumphigh/vlm-action/blob/main/training/memory_quantitative_analysis.md)
    
## 3. Training

### 3.1 Tensor parallel


### 3.2 Deepspeed
Talk about how to utilize zero on large model training, and, how to apply deepspeed within your train pipeline.    
3.2.1 [deepspeed overview doc](https://github.com/yzy-jumphigh/vlm-action/blob/main/training/zero_overview.md)   

3.2.2 run commands
    #single gpu
    deepspeed --num_gpus 1 training/llama2_clm_sft_lora_with_zero.py
    #multi gpus
    deepspeed training/llama2_clm_sft_lora_with_zero.py

3.2.3 [deepspeed memory quantitative analysis](https://github.com/yzy-jumphigh/vlm-action/blob/main/training/memory_quantitative_analysis.md)









