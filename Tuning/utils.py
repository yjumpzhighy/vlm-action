import torch

def inference_answer(instruct, tokenizer, model):
    def generate_prompt(instruction):
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

                ### Instruction:
                {instruction}

                ### Response:"""
    
         
    prompt = generate_prompt(instruct)
    prompt = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        input = {k:v.cuda() for k,v in prompt.items()}
        output = model.generate(input_ids=input['input_ids'],
                                attention_mask=input['attention_mask'],
                                max_new_tokens=512)
        decoded = tokenizer.batch_decode(output.detach().cpu().numpy(), 
                                    skip_special_tokens = True)[0]
        print("==>Instruction:", instruct)
        print("==>Response:", decoded.split("### Response:")[1].strip())
