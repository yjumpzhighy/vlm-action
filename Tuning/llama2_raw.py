import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
    GenerationConfig
)
from utils import inference


base_model_name = "NousResearch/Llama-2-7b-chat-hf"
base_model = AutoModelForCausalLM.from_pretrained( 
    base_model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16).cuda()
tokenizer = AutoTokenizer.from_pretrained(base_model_name, 
                                          trust_remote_code=True)

inference("how is the company amazon?", tokenizer, base_model)
