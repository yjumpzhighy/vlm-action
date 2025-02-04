# Gaussian Focal Loss

## focal loss:
loss(pt) = -α * (1-pt)^r * log(pt) 

pt: probability of the correct class (after applying the sigmoid for binary classification or softmax for multi-class).  
α: balancing factor to adjust the importance between different classes.  
r: adjusting how much focus given hard examples. A higher value reduces the loss contribution from easy examples.  


## gaussian focal loss
integrates a Gaussian function to soften the loss behavior near the class boundary. 

loss(pt) = -α * (1-pt)^r * exp(-(pt-0.5)^2 / 2 / σ^2) * log(pt) 

exp(): emphasizes the importance of pt near 0.5 and reduces the effect of extreme values (very close to 0 or 1).  
σ: controlling the width of the Gaussian function. 
