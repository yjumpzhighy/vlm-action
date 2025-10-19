
# Self-Distill

## iBOT
self-supervised distill teacher-student network architecture, where the teacher network's parameters            
are a momentum-updated average of the student's parameters, acts as the "online tokenizer".               
1.teacher network (with momentum-updated parameters) that sees the full, unmasked image and outputs             
  the target tokens (global and grounding) distribution for the student to predict.                  
2.The student (masked input) learns by matching the patch tokens distribution of the teacher.             
3.student model update from loss direct gradient                 
4.teacher model update from momentum average of student's update                    

```python

u, v = Augment(x)
^u = ApplyMask(u)
^v = ApplyMask(v)

with no_grad():
  Patch_u, CLS_u = teacher(u)  // Teacher output patch tokens and CLS token for view u
  Patch_v, CLS_v = teacher(v)  // Teacher output patch tokens and CLS token for view v

Patch_^u, CLS_^u = student(^u)  // Student predict patch tokens and CLS token for masked masked view ^u
Patch_^v, CLS_^v = student(^v)  // Student predict patch tokens and CLS token for masked masked view ^v

Loss_MIM = CrossEntropy(Patch_^u[M], Patch_v[M]) + CrossEntropy(Patch_^v[M], Patch_u[M])  // Masked Image Modeling, local grounding loss                
L_CLS = CrossEntropy(CLS_^u, CLS_v) + CrossEntropy(CLS_^v, CLS_u)    // class tokens loss, for global loss
```


## Gram matrixs
for feature map F[C, H*W], calcualte the correlations between channels:         
```python
G = F * F.T   # shape [C,C]
# G(i，j)： similarity between the feature vector at spatial location i and the feature
vector at spatial location j.        
```
Overall, it meaures correlation between channels among the whole image, i.e, "style" or       
"texture" independent of position.            


## Centering
Teacher could minimize the DINO loss by making output probability distribution extremely             
"peaky" on just a single dimension, regardless of the input image, and student would         
quickly learn to copy this output, which, further strengthen this "peaky", i.e, collapsed.    
              
we use c, "average vector of the teacher's output probabilities", to guarantee teacher's        
features are normalized and well-distributed.

```python
# teacher features Z(B,C):
# initial centering vector c: [1, C]

# centering teacher output
p_teacher = softmax(Z)
p_teacher_center = p_teacher - c

# EMA update centering  vector
c_batch_mean = mean(p_teacher)
c = (0.8 * c) + ((1 - 0.8) * c_batch_mean)
```


## Koleo loss
additive regularization term to DINO and iBOT losses, to prevents feature collapse.
For feature map Z（B，C),





## DINOV3


