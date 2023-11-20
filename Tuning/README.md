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


## prompt-tuning
prompt tuning uses PET to construct inputs, to transmit finetune tasks to pre-train mlm tasks.

For example prompt to emotionally classify "I like disney films very much. [SEP]":
prompt as "I like disney films very much. [SEP] It was [MASK]. [SEP]", which becomes a mlm task having the model to predict the [MASK]
and allow cross entropy to calcuate the gradient. 
In real experiment for faster training, usally narrow the label space. In above case,can project words like "great""good""awesome" to
"positive", and projects words like "bad""awful""terrible" to "negative".

Prompt-tuning did great improvment for llm tuning compared with classic finetune, however, the bottleneck lies:
1) very sentitive to pattern design, hard to get the best performance.
2) different tasks even sentences may perfer different patterns, hard to unify. 
3) discrete prompts corresponding word embeddings didn't involved in training.

## p-tuning
Based on prompt-tuning relys on pattern design, alternative self-adaptive continuous finetune is p-tuning.

   
