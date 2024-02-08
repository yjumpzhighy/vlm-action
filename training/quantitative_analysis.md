

## 1. Basic model size
A model usually consist of (embedding, encoder/decoder, lm_head).
### embedding 
embedding layer is a look-up table, with parameters:
N = n_vocab * d_model

n_vocab is the size of vocab dict, d_model is the embedding size. 
In some cases, embedding layer got frozen and un-trainable.

### encoder/decoder
with each multi-head attention block, it can be represented as:
N(Q,K,V) = 3*(d_model*d_head*n_head) + d_head*n_head*d_model 

d_model is the block input embedding size, d_head is dimension of 
QKV, d_head is number of heads.

after multi_head attention block, followed by 2-linear layers mlp:
N(mlp) = d_model * d_ffn + d_ffn * d_model

d_model is mlp input embedding size, d_ffn is the projected ffn 
embedding size.

### lm_head
simple linear layer to project feature space into probability space:
N(lm_head) = d_model * n_vocab

d_model is the lm_head input embedding size, n_vocab is vocab dict size.

Thus, the total model parameters size becomes:
N(model) = n_vocab * d_model + n_blocks*(4(d_model*d_head*n_head) + 2(d_model*d_ffn)) + d_model * n_vocab

Use llama2-7B as example, where n_vocab=32000, d_model=4096, d_model=d_head*n_head,
d_ffn=11008, n_blocks=32. 
N(llama2-7b) = 6738149376, almost 7b parameters.
