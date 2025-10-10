
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
  patchtokens_u, CLS_u = teacher(u)  // Teacher output patch tokens and CLS for view u
  patchtokens_v, CLS_v = teacher(v)  // Teacher output patch tokens and CLS for view v

patchtokens_^u, CLS_^u = student(^u)  // Student predict patch tokens and CLS for masked masked view ^u
patchtokens_^v, CLS_^v = student(^v)  // Student predict patch tokens and CLS for masked masked view ^v

Loss_MIM = CrossEntropy(P_^u[M], P_v[M]) + CrossEntropy(P_^v[M], P_u[M])  // Masked Image Modeling, for local grounding loss
L_CLS = CrossEntropy(CLS_^u, CLS_v) + CrossEntropy(CLS_^v, CLS_u)    // class tokens loss, for global loss

```



## DINOV3

