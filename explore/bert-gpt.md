
## 1.encoder / decoder difference
   ![image](https://github.com/user-attachments/assets/2440d03f-77c8-45f5-9e54-c96c9b63f086)
   
   The decoder applies cross-attention, while encoder uses self-attention:
   - use ground truth as output embedding in training, and use previsouly generated results as output embedding in inference
   - lower-triangular-masked self-attention on output embedding, as it only look ahead
   - encoder encoded input embedding as q,k and decoder output embedding as v, in cross-attention

## 2.gpt
   ![image](https://github.com/user-attachments/assets/1ecec3c0-5107-42e2-94e6-1132189ae587)  ![image](https://github.com/user-attachments/assets/5c028910-f886-48f6-be42-9d83c37d0e86)
    
    Gpt based on transformer decoder, but removed cross-attn.
    - main task is generate next word based on previous generated words
    - lower-triangular-masked self-attention on embedding, as it only look ahead
    
## 3.bert
   ![image](https://github.com/user-attachments/assets/49d3b5c4-c8ed-47ac-b3df-bad57744926e)  ![image](https://github.com/user-attachments/assets/2a062fa6-8ff9-491a-984a-adf892813d7f)
    
    Bert is based on tranformer encoder.
    - main task is generate masked tokens (MLM) and understand two setences relation (NLP)
    - bidiretional attention, thus no directional mask applied 
   
