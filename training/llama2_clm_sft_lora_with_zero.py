import os
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from transformers import (AutoModelForCausalLM, AutoConfig, AutoModel, AutoTokenizer,
                          BitsAndBytesConfig, pipeline, logging,
                          get_linear_schedule_with_warmup, Trainer,
                          TrainerCallback, TrainingArguments, HfArgumentParser,
                          get_scheduler)
import random
import numpy as np
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model, TaskType
from utils import TwitterComplaintDataset, parse_args, get_optimizer_grouped_parameters, convert_linear_layer_to_lora
import torch.distributed as dist
import deepspeed
from deepspeed import get_accelerator
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
import math
from tqdm import tqdm
from torch.nn.modules.loss import CrossEntropyLoss

#"meta-llama/Llama-2-7b-chat-hf" #'bigscience/bloomz-560m' #"NousResearch/Llama-2-7b-chat-hf"
base_model_name = "meta-llama/Llama-2-7b-chat-hf"
learning_rate = 8e-4
num_train_epochs = 30

#Lora
lora_r = 8
lora_alpha = 32
lora_dropout = 0.1
lora_replace_module = ["q_proj","k_proj","v_proj"] #["query_key_value"]

#Deepspeed
use_deepspeed = True


def get_ds_config(filepath):
    import json

    data = json.load(open(filepath, 'r'))
    return data
      
        

def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    get_accelerator().manual_seed_all(args.seed)

    torch.distributed.init_process_group(backend='nccl')
    local_rank = 0
    global_rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    #device = torch.device(get_accelerator().device_name(), local_rank)
    if use_deepspeed:
        get_accelerator().set_device(local_rank)
        deepspeed.init_distributed()

    ds_config = get_ds_config(
        os.path.join(os.getcwd(), "deepspeed_zero2_no_offload.json"))
    ds_config['train_batch_size'] = ds_config['train_micro_batch_size_per_gpu'] * world_size *\
                                    ds_config['gradient_accumulation_steps']
    zero_offload = ds_config['zero_optimization']['offload_optimizer'][
        'device'] != 'none'

    tokenizer = AutoTokenizer.from_pretrained(base_model_name,
                                              trust_remote_code=True,
                                              use_fast=False)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(base_model_name,
                                                 torch_dtype=torch.float16,
                                                 return_dict=True)
    model.enable_input_require_grads()

    
    # enable_lora:
    # peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM,
    #                                     target_modules = lora_replace_module,
    #                                     inference_mode=False,
    #                                     r=lora_r,
    #                                     lora_alpha=lora_alpha,
    #                                     lora_dropout=lora_dropout)
    # model = get_peft_model(model, peft_config)
    # model.enable_input_require_grads()
    # model.print_trainable_parameters()

    # use customized lora layers instead.
    model = convert_linear_layer_to_lora(
        model,
        lora_replace_module,
        lora_dim=lora_r,
        lora_droppout=lora_dropout)

    model = model.cuda()

    dataset = TwitterComplaintDataset(tokenizer, "")
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=ds_config['train_micro_batch_size_per_gpu'],
        num_workers=2,
        shuffle=True,
    )

    # optimizer
    optimizer_grouped_parameters = model.parameters()
    if use_deepspeed and zero_offload:
        optimizer = DeepSpeedCPUAdam(optimizer_grouped_parameters,
                                     lr=learning_rate,
                                     betas=(0.9, 0.95))
    elif use_deepspeed:
        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=learning_rate,
                              betas=(0.9, 0.95))
    else:
        optimizer = torch.optim.Adam(optimizer_grouped_parameters,
                                     lr=learning_rate)




    lr_scheduler = get_scheduler(name = "cosine",
                                 optimizer = optimizer,
                                 num_warmup_steps = 0,
                                 num_training_steps = num_train_epochs *\
                                     math.ceil(len(dataloader)/ds_config['gradient_accumulation_steps'])
                                )
    
    if use_deepspeed:
        model, optimizer, _, lr_scheduler = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            config=ds_config,
            lr_scheduler=lr_scheduler,
            dist_init_required=True)

    #train
    criterior = CrossEntropyLoss(reduction='none')

    # mix precision
    scaler = GradScaler()
    for epoch in range(num_train_epochs):
        epoch_loss = 0.0
        model.train()
        for batch in tqdm(dataloader, desc="train", leave=False):
            input = {k: v.cuda() for k, v in batch.items()}
            
            with autocast():
                # [batch, token_len, vocab_len]
                outputs = model(input_ids=input['input_ids'],
                                attention_mask=input['attention_mask'])


                logits = outputs.logits[:, :-1, :]
                labels = input['labels'][:, 1:]
                mask = (labels > 0).bool()

                loss = torch.mean(
                    criterior(logits.permute(0, 2, 1), labels.long())[mask])

                epoch_loss = loss.detach().float()


                if use_deepspeed:
                    model.backward(loss)
                    model.step()
                else:
                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    lr_scheduler.step()


        print("=======>epoch loss:", epoch_loss)


    #evaluate
    if global_rank == 0:
        if use_deepspeed:
            model = model.module
            
        model.eval()
        model.config.end_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = model.config.end_token_id
        generator = pipeline("text-generation",
                                model = model,
                                tokenizer = tokenizer,
                                device='cuda:0')

        with torch.no_grad():
            with autocast():
                response = generator(
                    [f'@united not happy with this delay from Newark to Manchester tonight :( only 30 mins free Wi-fi sucks ... : ',
                        f'@JetBlue Completely understand but would prefer being on time to filling out forms.... : ',
                        f'Looks tasty! Going to share with everyone I know #FebrezeONE #sponsored https://t.co/4AQI53npei : '
                    ],
                    max_new_tokens=10)
                
                print(response)
        

                
                


if __name__ == "__main__":
    main()
