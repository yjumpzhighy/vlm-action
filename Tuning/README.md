# FineTune tasks on pretrained model 

## classical model finetune
In traditional deep learning model finetune work, the most common method is to frozen the model body and finetune  
the prediction heads, or add trainable mlp layers to accommendate specific tasks. This usually works as following reasons:  
1) model usually with smaller size (relatively), easy to fine tuning training.  
2) pre-train task and finetune task usually are aligned. for example, train a resnet for classificaiton on imagenet, then finetune  
	on specific dataset for classification task as well.  
3) acceptable gap between pre-train dataset and finetune dataset.  

However, above criterials won't met in llm finetune tasks:   
1) pre-train task usually based on unsupervised MLM, while finetune supervised tasks varies, like classification, next sequence  
	prediction, text generation, relation, etc.  
2) pre-train/tuning data scale not match  
3) extreme to train a ~100B model  


## hard prompt
prompt tuning uses PET to construct inputs, to transmit finetune tasks to pre-train mlm tasks.  

For example prompt to emotionally classify "I like disney films very much. [SEP]":  
Prompt as "I like disney films very much. [SEP] It was [MASK]. [SEP]", which becomes a mlm task having the model to predict the [MASK]  
and allow cross entropy to calcuate the gradient.  
In real experiment for faster training, usally narrow the label space. In above case,can project words like "great""good""awesome" to  
"positive", and projects words like "bad""awful""terrible" to "negative".  

Prompt-tuning did great improvment for llm tuning compared with classic finetune, however, the bottleneck lies:  
1) very sentitive to pattern design, hard to design the right pattern.  
2) different tasks even sentences may perfer different patterns, hard to unify.  
3) discrete prompts corresponding word embeddings didn't involved in training.
4) manually pattern design costs a lot.

## soft prompt
Q: Does the pattern must consist of natual language tokens?  
A: No!It can be anytime once the model can recongnize, to guide the model on specific sub tasks.
Thus, it is actually not necessary to design the descrte static template.

### prompt-tuning
    # 1. What does prompt applied model look like?
    	peft_config = PromptTuningConfig(task_type=TaskType.CAUSAL_LM,
                                     prompt_tuning_init=PromptTuningInit.TEXT,
                                     prompt_tuning_init_text="",
                                     num_virtual_tokens=8,
                                     tokenizer_name_or_path=base_model_name)
    	base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
    	model = get_peft_model(base_model, peft_config)
    	print(model)

    	""" Besides base model, a new PromptEmbedding will be added into PeftModelForCausalLM
     	(prompt_encoder): ModuleDict(
      		(default): PromptEmbedding(
			(embedding): Embedding(8, 1024)
   		))
    	"""

    # 2. How does PromptEmbedding integrated with base model? trimmed related code in <perf/peft_model.py>:
		prefix_attention_mask = torch.ones(batch_size, num_virtual_tokens)
            	attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
    
		prefix_labels = torch.full((batch_size, num_virtual_tokens), -100) 
 		labels = torch.cat((prefix_labels, labels), dim=1)
   
  		prompt_tokens = torch.arange(num_virtual_tokens).long().unsqueeze(0).expand(batch_size, -1)	
 		#shape(b,num_virtual_tokens,token_dim) 
  		prompts = PromptEmbedding(prompt_tokens) 
   		inputs_embeds = torch.cat((prompts, inputs_embeds), dim=1)

		# After concat virtual tokens embedding with inputs embedding, the inputs to base model actually 
  		# changes to (b, num_virtual_tokens+num_input_tokens, token_dim).
		# The label and attention mask must pad as well.

    # 3. Conclusion:
    	Prompt tuning create a new virtual tokens embedding, and then concat virtual tokens embeddings with 
    	input tokens embeddings, and then feed into base model.
     	In training, only the new added PromptEmbedding has back grad and updated.
    
    

### prefix-tuning
Instead of descrte static template, continuous trainable vitual tokens be added as task prefix to intruct finetune tasks.

    # 1. What does prefix applied model look like?
    	base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
    	peft_config = PrefixTuningConfig(task_type="CAUSAL_LM", 
                                     num_virtual_tokens=30,
                                     prefix_projection=True)
    	model = get_peft_model(base_model, peft_config)
    	print(model)
    
    	""" Besides base model, a new PrefixEncoder will be added into PeftModelForCausalLM
    	(prompt_encoder): ModuleDict(
    		(default): PrefixEncoder(
      			(embedding): Embedding(30, 4096)
      			(transform): Sequential(
        		(0): Linear(in_features=4096, out_features=4096, bias=True)
        		(1): Tanh()
        		(2): Linear(in_features=4096, out_features=262144, bias=True)
      			)		
    		)
	     )
     	"""

     # 2. What is PrefixEncoder? trimmed related code in <perf/tuners/prefix_tuning/model.py>:
    	 def __init__(self, config):
            # Use a two-layer MLP to encode the prefix
            self.embedding = torch.nn.Embedding(num_virtual_tokens, token_dim)
            self.transform = torch.nn.Sequential(
                torch.nn.Linear(token_dim, encoder_hidden_size),
                torch.nn.Tanh(),
                torch.nn.Linear(encoder_hidden_size, num_layers * 2 * token_dim),
            )
     	def forward(self, prefix: torch.Tensor):
            prefix_tokens = self.embedding(prefix)
            past_key_values = self.transform(prefix_tokens)
     	    return past_key_values
     	# simply, PrefixEncoder take virtual tokens inputs_id and output past_key_values with shape(num_virtual_tokens,
     	# num_layers * 2 * token_dim), we will find out what num_layers and 2 means here 

     # 3. How does prefixEncoder integrated with base model? trimmed related code in <perf/peft_model.py>:
     	# expand attention mask
		prefix_attention_mask = torch.ones(batch_size, num_virtual_tokens) 
  		attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
     
		#shape(b, num_virtual_tokens).
      	prompt_tokens = torch.arange(num_virtual_tokens).long().unsqueeze(0).expand(batch_size, -1)

 		#shape(b,num_virtual_tokens,num_layers*2,num_attention_heads,token_dim//num_attention_heads)
 		past_key_values = PrefixEncoder(prompt_tokens).view(
 			batch_size, num_virtual_tokens, num_layers * 2,
			num_attention_heads, token_dim // num_attention_heads)
  
		#shape(num_layers,2,b,num_attention_heads,num_virtual_tokens,token_dim//num_attention_heads)
 		past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
 
    	#Inside basemodel(like LlamaForCausalLM), trimed related code in <transformers/models/llama/modeling_llama.py>:
		class LlamaModel(LlamaPreTrainedModel):
 			def forward():
				for idx, decoder_layer in enumerate(self.layers):
					layer_outputs = decoder_layer(hidden_states, attention_mask=attention_mask, 
							position_ids=position_ids, past_key_value= past_key_values[idx], output_attentions=output_attentions, 
	   						use_cache=use_cache, 
            		) 
		class LlamaAttention(nn.Module):
			def forward():
  				query_states = self.q_proj(hidden_states) 
            			key_states = self.k_proj(hidden_states) 
            			value_states = self.v_proj(hidden_states) 
  				if past_key_value is not None: 
            		key_states = torch.cat([past_key_value[0], key_states], dim=2) 
            		value_states = torch.cat([past_key_value[1], value_states], dim=2) 
	      
	       #Compared to noraml attention layers, the prefix model passed in past_key_value from prefix encoder into
	       #base model, and concat to k and v in each layer! Thus in the base model attention layer, the query contains
	       #input info(no prefix), the key and val contains input and prefix info, which is kind of info fuse.

     # 4. Conclusion
     Prefix tuning created a new encoder for vitual tokens, and its output past_key_values will be inserted into
	 base model's each layer! Look at the past_key_values shape(num_layers,2,b,num_attention_heads,num_virtual_tokens,
     token_dim//num_attention_heads), num_layers is the number of base model's layers, 2 is for q and v concat.
     In training, only the new added PrefixEncoder embedding has back grad and updated.
     

	

## p-tuning-v2
p-tuning-v2 is highly similar with prefix tuning, which concat virual tokens embddings into transformer q/k.

	# 1. Compared with prefix tuning, minor difference in PrefixEncoder. related code in <perf/tuners/prefix_tuning/model.py>:
    	def __init__(self, config):  
     		# Use a two-layer MLP to encode the prefix 
       		self.embedding = torch.nn.Embedding(num_virtual_tokens, num_layers * 2 * token_dim) 
     	def forward(self, prefix: torch.Tensor): 
      		prefix_tokens = self.embedding(prefix)
     		return past_key_values
	# 2.Conclusion
 	The pipeline is almost the same with prefix tuning method, while a minor difference on how virtual tokens embedding generated.
  	Obvisouly, p-turning-v2 adds much less new trainable parameters than prefix tuning.


## lora
lora add mlp (low rank) at qkv projection from input x, and keeps all dimensions the same as original model.
	
	# 1. What does lora model looks like? it inserts lora layers into transformer block 
 	(self_attn): Attention(
              (q_proj): Linear(
                in_features=4096, out_features=4096, bias=False
                (lora_dropout): ModuleDict((default): Dropout(p=0.1, inplace=False))
                (lora_A): ModuleDict((default): Linear(in_features=4096, out_features=8, bias=False))
                (lora_B): ModuleDict((default): Linear(in_features=8, out_features=4096, bias=False))
              )
              (k_proj): Linear(in_features=4096, out_features=4096, bias=False)
              (v_proj): Linear(
                in_features=4096, out_features=4096, bias=False
                (lora_dropout): ModuleDict((default): Dropout(p=0.1, inplace=False))
                (lora_A): ModuleDict((default): Linear(in_features=4096, out_features=8, bias=False))
                (lora_B): ModuleDict((default): Linear(in_features=8, out_features=4096, bias=False))
              ) 
	# 2. How does lora layers integrated with base model? reference in peft/src/peft/tuners/lora/model.py
 	     def forward():
		lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                x = x.to(lora_A.weight.dtype)
                result += lora_B(lora_A(dropout(x))) * scaling
 	     # i.e, applying y=Wx+BAx, where A initalized with guassian and B initialize with 0.
       # 3. Conclusion
       Lora added low rank matrix mlp to capture data features. Without touching transformer core logics,
       lora just changed a little bit how to get qkv from input x. 

   
