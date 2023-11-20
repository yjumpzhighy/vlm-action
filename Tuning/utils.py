import torch

def inference(prompt, tokenizer, model):     
    prompt = tokenizer(f"<s>[INST] {prompt} [/INST]", return_tensors="pt")
    with torch.no_grad():
        input = {k:v.cuda() for k,v in prompt.items()}
        output = model.generate(input_ids=input['input_ids'],
                                attention_mask=input['attention_mask'],
                                max_new_tokens=256)
        print(tokenizer.batch_decode(output.detach().cpu().numpy(), 
                                    skip_special_tokens = True))
