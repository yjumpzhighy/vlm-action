# Let's first clarify what are the optimizer states. We will compare most common optimizers SGD and Adam.

# General model:
    class Net(nn.Module):
        def __init__(self):
          super().__init__()
          self.fc1 = nn.Linear(5,3,bias=False)
          self.fc2 = nn.Linear(3,2,bias=False)
        def forward(self,x):
          x = self.fc1(x)
          x = self.fc2(x)
          return x
    net = Net().cuda()
    loss = torch.nn.modules.loss.CrossEntropyLoss()
    
## SGD optimizer
    optimizer = torch.optim.SGD(net.parameters(), lr = 1e-3)
    for i in range(5):
      optimizer.zero_grad()
      input = torch.rand(16,5).cuda()
      target = torch.ones(16).cuda()
      logits = net(input)
      logits_loss = loss(logits, target.long())
      logits_loss.backward()
      optimizer.step()
      print(optimizer.state)

    outputs >> 
    defaultdict(<class 'dict'>, {Parameter containing:
        tensor([[ 0.2335, -0.4083,  0.1899, -0.1726,  0.2455],
        [ 0.0609,  0.0835,  0.2872, -0.2975, -0.3320],
        [-0.0181, -0.2002, -0.3916,  0.0904,  0.2664]], device='cuda:0',
        requires_grad=True): {'momentum_buffer': None}, Parameter containing:
        tensor([[-0.0284, -0.2377,  0.0766],
        [ 0.4880,  0.3770, -0.5130]], device='cuda:0', requires_grad=True): {'momentum_buffer': None}})

## Adam optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr = 1e-3)

    for i in range(1):
        optimizer.zero_grad()
        input = torch.rand(16,5).cuda()
        target = torch.ones(16).cuda()
        logits = net(input)
        logits_loss = loss(logits, target.long())
        logits_loss.backward()
        optimizer.step()
    print(optimizer.state)

    outputs >>
    defaultdict(<class 'dict'>, {Parameter containing:
        tensor([[ 0.1231, -0.2155,  0.3782,  0.0744,  0.0497],
        [ 0.1089,  0.2391,  0.4422, -0.2361,  0.0235],
        [ 0.4218, -0.0264, -0.4429,  0.0436,  0.1418]], device='cuda:0',
        requires_grad=True): {'step': 1, 'exp_avg': tensor([[-0.0054, -0.0046, -0.0041, -0.0057, -0.0046],
        [ 0.0036,  0.0030,  0.0027,  0.0037,  0.0030],
        [-0.0092, -0.0079, -0.0069, -0.0097, -0.0078]], device='cuda:0'), 'exp_avg_sq': tensor([[2.9528e-06, 2.1347e-06, 1.6599e-06, 3.2591e-06, 2.1137e-06],
        [1.2711e-06, 9.1896e-07, 7.1447e-07, 1.4029e-06, 9.0980e-07],
        [8.5557e-06, 6.1855e-06, 4.8096e-06, 9.4437e-06, 6.1243e-06]],
        device='cuda:0')}, Parameter containing:
        tensor([[-0.3027,  0.4375, -0.1573],
        [-0.1073,  0.3126,  0.1741]], device='cuda:0', requires_grad=True): {'step': 1, 'exp_avg': tensor([[ 0.0095,  0.0118,  0.0064],
        [-0.0095, -0.0118, -0.0064]], device='cuda:0'), 'exp_avg_sq': tensor([[9.0677e-06, 1.3909e-05, 4.1267e-06],
        [9.0677e-06, 1.3909e-05, 4.1267e-06]], device='cuda:0')}})

## Compare
As shown above, Adam keeps weight, exp_avg, exp_avg_sq as states. In another word, 3x memory usage.
