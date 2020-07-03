---
layout:     post
title:      Foundations of Implementing Deep Learning Networks with Pytorch 
date:       2020-07-02
author:     Yongchao Wu
header-img: img/post-main.jpg
catalog: true

---

### Deep learning network

Deep learning network seems to be a very esoteric concept. However, if we neglect its fancy name and focus what indeed it is, deep learning is just a combination of multiple layers of neural networks. Pytorch provides us a very easy way to implement deep learning netwoks as simple as building blocks. This blog will introduce the foundations of Pytorch deep learning techniques. 

##### Implement Deep Learning with Class <span style="color:red">Module</span>

We can build our own model by inheriting Class Module in pytorch, an example:

```
import torch
from torch import nn

class MLP(nn.Module):
    
    def __init__(self, **kwargs):

        super(MLP, self).__init__(**kwargs) # overload init
        self.hidden = nn.Linear(784, 256) # Hidden layers
        self.act = nn.ReLU()
        self.output = nn.Linear(256, 10)  # output layers

    def forward(self, x):  #Overload forward
        a = self.act(self.hidden(x))
        return self.output(a)
```

##### Implement Deep Learning with Class <span style="color:red">Sequential</span>

We can use Class Sequential to include multiple neural networks. We don't have to write the forward function, but we have to carefully design the number of inputs, outputs for different layers. 

```
net = nn.Sequential(
        nn.Linear(num_inputs, num_hiddens),
        nn.ReLU(),
        nn.Linear(num_hiddens, num_outputs), 
        )
```

##### Implement Deep Learning with Class <span style="color:red">ModuleList</span>

ModuleList receives a list of submodules as input, and then can also perform append and extend operations like List. When using ModuleList, we have to overite the forward function. 

```
class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(10, 10) for i in range(10)])

    def forward(self, x):
        # ModuleList can act as an iterable, or be indexed using ints
        for i, l in enumerate(self.linears):
            x = self.linears[i // 2](x) + l(x)
        return x
```

##### Example: implement of a complex deep learning network

We use Sequential and Module to build a deep learning network that contains FancyMLP, Linear Layer and NestedMLP.

```
class FancyMLP(nn.Module):
    def __init__(self, **kwargs):
        super(FancyMLP, self).__init__(**kwargs)

        self.rand_weight = torch.rand((20, 20), requires_grad=False) 
        self.linear = nn.Linear(20, 20)

    def forward(self, x):
        x = self.linear(x)
        x = nn.functional.relu(torch.mm(x, self.rand_weight.data) + 1)
        x = self.linear(x)
        while x.norm().item() > 1:
            x /= 2
        if x.norm().item() < 0.8:
            x *= 10
        return x.sum()

class NestMLP(nn.Module):
    def __init__(self, **kwargs):
        super(NestMLP, self).__init__(**kwargs)
        self.net = nn.Sequential(nn.Linear(40, 30), nn.ReLU()) 

    def forward(self, x):
        return self.net(x)
        
net = nn.Sequential(NestMLP(), nn.Linear(30, 20), FancyMLP())
```

### Conlusion

- The model can be constructed by inheriting the Module class.

- The Sequential, ModuleList, and ModuleDict classes all inherit from the Module class.

- Unlike Sequential, ModuleList and ModuleDict do not define a complete network, they just store different modules together, and you need to define the forward function yourself.

- Although classes such as Sequential can make model construction easier, directly inheriting the Module class can greatly expand the flexibility of model construction.

### Reference

Chapter 4 Dive into Deep Learning. 

https://d2l.ai/chapter_deep-learning-computation/index.html