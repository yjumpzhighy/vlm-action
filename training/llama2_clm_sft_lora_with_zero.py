import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
    logging,
    get_linear_schedule_with_warmup,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    HfArgumentParser
)
import random
import numpy as np
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model, TaskType
from utils import TwitterComplaintDataset
from torchpack import distributed as dist

#"meta-llama/Llama-2-7b-chat-hf" #'bigscience/bloomz-560m' #"NousResearch/Llama-2-7b-chat-hf"
base_model_name = "meta-llama/Llama-2-7b-chat-hf"
learning_rate = 8e-4
num_train_epochs = 50
output_dir = "./results"
fp16 = False
bf16 = False #A100 support
per_device_train_batch_size = 4
per_device_eval_batch_size = 1
gradient_accumulation_steps = 1
max_grad_norm = 0.3
weight_decay = 0.001
optim = "paged_adamw_32bit"
lr_scheduler_type = "constant"
warmup_ratio = 0.03
group_by_length = True
max_steps = -1
save_steps = -1
logging_steps = -1
max_seq_length = None
packing = False
deepspeed_config = "./deepspeed_zero2_no_offload.json"


#Lora
lora_r = 8
lora_alpha = 32
lora_dropout = 0.1
 
#BitsandBytes   
use_4bit = True
bnb_4bit_compute_dtype = "float16" #4bit base model dtype
bnb_4bit_quant_type = "nf4" #fp4, nf4
use_nested_quant = False #nested double quantization

    
distributed = True

    
def main():   
    if distributed:
        dist.init()
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        torch.cuda.set_device(dist.local_rank())
    
    training_arguments = TrainingArguments(
        deepspeed = deepspeed_config,
        output_dir = output_dir,
        num_train_epochs = num_train_epochs,
        per_device_train_batch_size = per_device_train_batch_size,
        gradient_accumulation_steps = gradient_accumulation_steps,
        optim = optim,
        save_steps = save_steps,
        logging_steps = logging_steps,
        learning_rate = learning_rate,
        weight_decay = weight_decay,
        fp16 = fp16,
        bf16 = bf16,
        max_grad_norm = max_grad_norm,
        max_steps = max_steps,
        warmup_ratio = warmup_ratio,
        group_by_length = group_by_length,
        lr_scheduler_type = lr_scheduler_type,
        report_to = "tensorboard",
        save_strategy="no"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, 
                                            trust_remote_code=True)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM,
                                    inference_mode=False,
                                    r=lora_r,
                                    lora_alpha=lora_alpha,
                                    lora_dropout=lora_dropout)
                   
    base_model = AutoModelForCausalLM.from_pretrained( 
        base_model_name,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16)

    base_model.enable_input_require_grads()
    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()
    model = model.cuda()

    dataset = TwitterComplaintDataset(tokenizer,"") 
    trainer = SFTTrainer(
        model = model,
        train_dataset = dataset,
        peft_config = peft_config,
        args = training_arguments,
        packing = True
    )
    trainer.train()
 
    if distributed == False or dist.local_rank() == 0:
        model.eval()
        test_inputs = tokenizer([f'@united not happy with this delay from Newark to Manchester tonight :( only 30 mins free Wi-fi sucks ... : ', 
                                f'@JetBlue Completely understand but would prefer being on time to filling out forms.... : ',
                                f'Looks tasty! Going to share with everyone I know #FebrezeONE #sponsored https://t.co/4AQI53npei : '],
                                padding='max_length',
                                truncation=True,
                                add_special_tokens=False,
                                max_length=64,
                                return_tensors="pt")

        with torch.no_grad():
            test_inputs = {k: v.cuda() for k, v in test_inputs.items()}
            test_outputs = model.generate(
                input_ids=test_inputs["input_ids"],
                attention_mask=test_inputs["attention_mask"], 
                max_new_tokens=10,
                eos_token_id=tokenizer.eos_token_id
            )
            print(tokenizer.batch_decode(test_outputs.detach().cpu().numpy(), skip_special_tokens=True))
    
        
if __name__ == "__main__":  
    main()