# Gaussian Focal Loss

## focal loss:
loss(pt) = -α * (1-pt)^r * log(pt)      #positive
         = -(1-α) * pt^r * log(1-pt)    #negative

pt: probability of the correct class (after applying the sigmoid for binary classification or softmax for multi-class).  
α: balancing factor to adjust the importance between different classes, usuall 0.25
r: adjusting how much focus given hard examples. A higher value reduces the loss contribution from easy examples. usually 2.

对于正样本，对简单易区分样本, pt较大, (1-pt)^r很小, log(pt)也小, 使得损失较小. 
          对复杂难区分样本, pt较小，使得(1-pt)^r较大，log(pt)也大, 使得损失也较大
对于负样本，对简单易区分样本, pt较小，pt^t很小, log(1-pt)也小, 使得损失较小. 
          对复杂难区分样本, pt较大，pt^t变大, log(1-pt)也变大, 使得损失较大



## gaussian focal loss
integrates a Gaussian function to soften the loss behavior near the class boundary. 

loss(pt) = -α * (1-pt)^r * exp(-(pt-0.5)^2 / 2 / σ^2) * log(pt) 

exp(): emphasizes the importance of pt near 0.5 and reduces the effect of extreme values (very close to 0 or 1).  
σ: controlling the width of the Gaussian function. 
