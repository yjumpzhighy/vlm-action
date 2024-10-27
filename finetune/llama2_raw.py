import os
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
from utils import inference_answer, inference_sentiment_classify, generate_prompt,\
                  generate_prompt_2, MlmDataset
import torch.optim as optim
from tqdm import tqdm
from torch.nn.modules.loss import CrossEntropyLoss


base_model_name = "meta-llama/Llama-2-7b-chat-hf"
base_model = AutoModelForCausalLM.from_pretrained( 
    base_model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16).cuda()
tokenizer = AutoTokenizer.from_pretrained(base_model_name, 
                                          trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

#inference_answer("how is the company amazon?", tokenizer, base_model, generate_prompt)
#inference_sentiment_classify("Yeah I am happy, hehe.", tokenizer, base_model, generate_prompt_2)

if True:
    dataset = MlmDataset('/home/zuyuanyang/Projects/vlm/finetune/data_sentiment.txt',tokenizer)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        num_workers=2,
        shuffle=True,
    )

    for name,param in base_model.named_parameters():
        if (name != "lm_head.weight"):
            param.requires_grad=False
        
    optimizer = optim.SGD(base_model.parameters(), lr=0.001, weight_decay=0.0001)
    criterior = CrossEntropyLoss(reduction='none')
    
    for epoch in range(30):
        base_model.train()
        epoch_loss = 0.0
        for batch in tqdm(dataloader, desc="train", leave=False):
            input = {k:v.cuda() for k,v in batch.items()}
            
            #[batch, token_len, vocab_len]
            output = base_model(input_ids=input['input_ids'],
                                attention_mask=input['attention_mask']).logits 

            loss = torch.mean(
                    criterior(output.permute(0,2,1), 
                              input['target_ids'].long())[input['mlm_mask']])
            epoch_loss = loss.item()
            
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            print(base_model.lm_head.weight.grad)
            print("G:", torch.mean(base_model.lm_head.weight.grad))
            print("W:", torch.mean(base_model.lm_head.weight.data))
            
        print("loss:", epoch_loss)
            
inference_answer("how is the company amazon?", tokenizer, base_model, generate_prompt)
inference_sentiment_classify("Yeah I am happy, hehe.", tokenizer, base_model, generate_prompt_2)