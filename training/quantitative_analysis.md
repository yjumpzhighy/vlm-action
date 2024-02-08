

## 1. basic model size analysis
A model usually consist of (embedding, transformer blocks, head).
### Embedding 
embedding layer is a look-up table, with parameters:
N = n_vocab * d_model

n_vocab is the size of vocab size, d_model is the embedding size. 
In some cases, embedding layer got frozen and un-trainable.
