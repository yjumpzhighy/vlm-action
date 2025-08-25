# Self-distillation with no labels (Emerging Properties in Self-Supervised Vision Transformers)

## conclude
1. ssl contrastive learning may lead to training collaps, i.e, different samples projected to a trival local
   feature representation, which is not distinguishable.
2. it is due to image nature, unlike languages which tokens are natually discrete providing strong constraits,
   images are continuous, without labels it would be hard to distinguish in feature space.

## pipeline
<img src="https://github.com/user-attachments/assets/2f7cb449-52e8-45d2-bd50-1e013df4361e" width="400" height="600">   
1. for image x, generate (x1,x2) after data augmentation
2. feed separately to teacher model and student model (same structure, different weights). the teacher model would
   extra perform centering (output substracted by mean values in batch). 
3. after softmax, get p1 and p2. 
4. get cross entropy loss with -p2*log(p1), to evalute the similarity
5. back-propogation on student model.
6. teach model updated by student model weights EMA.


## train
1. one image would geneate multi-view sub images, including two global (big) patches and several local (small) patches
2. all global+local patches can feed to student model, while teach model only accept global patches
3. in each pair (global, gloal) or (local, global), force the two models similar prediction distribution.


   
