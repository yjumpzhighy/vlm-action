import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
    get_linear_schedule_with_warmup,
)
from peft import PrefixTuningConfig, get_peft_model, TaskType
from tqdm import tqdm
from utils import inference_answer, generate_prompt, inference_sentiment_classify,\
                  generate_prompt_2, SentimentDataset, TwitterComplaintDataset
from torch.nn.modules.loss import CrossEntropyLoss


#"meta-llama/Llama-2-7b-chat-hf" #'bigscience/bloomz-560m'
base_model_name = "NousResearch/Llama-2-7b-chat-hf"
new_model_name = "llama-2-7b-xxx"

lr = 3e-2
num_epochs = 50
batch_size = 4
num_virtual_tokens = 30

tokenizer = AutoTokenizer.from_pretrained(base_model_name, 
                                          trust_remote_code=True)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.pad_token = tokenizer.eos_token

peft_config = PrefixTuningConfig(task_type=TaskType.CAUSAL_LM,
                                 num_virtual_tokens=num_virtual_tokens,
                                 prefix_projection=True   
                                )
                            
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
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=2,
    shuffle=True,
)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=(len(dataloader) * num_epochs),
)
criterior = CrossEntropyLoss(reduction='none')

for epoch in range(num_epochs):
    epoch_loss = 0
    model.train()
    aa = None
    bb = None
    
    for batch in tqdm(dataloader, desc="train", leave=False):
        input = {k:v.cuda() for k,v in batch.items()}

        # [batch, token_len, vocab_len]
        outputs = model(input_ids=input['input_ids'],
                                attention_mask=input['attention_mask'])
        logits = outputs.logits[:,:-1,:]
        
        labels = input['labels'][:,1:]
        # prefix_labels = torch.full((input['input_ids'].shape[0], num_virtual_tokens), -100).to(labels.device)
        # labels = torch.cat((prefix_labels, labels), dim=1)
        mask = (labels > 0).bool() 

        loss = torch.mean(
                criterior(logits.permute(0,2,1), labels.long())[mask])
        epoch_loss += loss.detach().float()
         
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

    print("===>epoch loss:", epoch_loss)


    
    
model.eval()
test_inputs = tokenizer(f'@united not happy with this delay from Newark to Manchester tonight :( only 30 mins free Wi-fi sucks ... Label : ',
                        add_special_tokens=False,   
                        return_tensors="pt")
with torch.no_grad():
    test_inputs = {k: v.cuda() for k, v in test_inputs.items()}
    test_outputs = model.generate(
        input_ids=test_inputs["input_ids"], attention_mask=test_inputs["attention_mask"], 
        max_new_tokens=10,
        eos_token_id=tokenizer.eos_token_id
    )

    print(tokenizer.batch_decode(test_outputs.detach().cpu().numpy(), skip_special_tokens=True))

                
