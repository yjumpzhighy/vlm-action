
## 1.encoder / decoder difference
   ![image](https://github.com/user-attachments/assets/2440d03f-77c8-45f5-9e54-c96c9b63f086)
   
   The decoder applies cross-attention, while encoder uses self-attention:
   - uses previsouly generated results as output embedding
   - lower-triangular-masked self-attention on output embedding, as it only look ahead
   - encoder encoded input embedding as q,k and decoder output embedding as v, in cross-attention
