# tuning tasks on pretrained model 

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
Q: Does the pattern must consist of natual knowledge tokens?  
A: No!It can be anytime once the model can recongnize.  
Thus, it is actually not necessary to design the descrte static template.

### prefix-tuning
Instead of descrte static template, continuous trainable vitual tokens be added as task prefix to intruct finetune tasks.


    peft_config = PrefixTuningConfig(task_type="CAUSAL_LM", 
                                     num_virtual_tokens=30,
                                     prefix_projection=True)
    model = get_peft_model(base_model, peft_config)

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


## p-tuning


p-tuning uses trainable virtual tokens as prompt, able to construct the pattern self-adaptively.




   
