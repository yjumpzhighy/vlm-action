
## 1.encoder / decoder difference
   ![image](https://github.com/user-attachments/assets/2440d03f-77c8-45f5-9e54-c96c9b63f086)
   
   The decoder applies cross-attention, while encoder uses self-attention:
   - use ground truth as output embedding in training, and use previsouly generated results as output embedding in reference
   - lower-triangular-masked self-attention on output embedding, as it only look ahead
   - encoder encoded input embedding as q,k and decoder output embedding as v, in cross-attention

## 2.gpt
   ![image](https://github.com/user-attachments/assets/1ecec3c0-5107-42e2-94e6-1132189ae587)

   gpt based on transformer decoder, but removed cross-attn due to encoder removed.
   - 

## 3.bert
