---
layout:     post
title:      Pytorch Basic Neural Networks
date:       2020-06-30
author:     Yongchao Wu
header-img: img/post-main.jpg
catalog: true

---

Basic Neural Networks and pytorch
=================

Linear regression(LR), softmax and multilayer perceptron(MLP) are among the basic neural network models. They are easy to implement and used to perform regression, classification and multi-classification tasks. This blog will briefly introduce the basic concept of these simple models. Pytorch provides a very convenient  way to implement these models, thus a simple tutorial of hwo to use pytorch will be also illustrated. 

Linear Regression
---------------------------------------------------

Linear Regression is a single-layer neural network. It contains the model, training data, loss function and optimization algorithms. 

#### Model, loss function, optimization

Suppose we have $n$ number of data examples with feature number $d$, so the prediction $\hat{y}$ is:

$$\hat{y} = w_1  x_1 + ... + w_d  x_d + b.$$

For a collection of features $\mathbf{X}$, the predictions $\hat{\mathbf{y}} \in \mathbb{R}^n$, can be expressed as:

$${\hat{\mathbf{y}}} = \mathbf{X} \mathbf{w} + b,$$

The loss function is defined as the difference between the predicted $\hat{y}$ and the real $y$, loss function can be noted as:

$$l^{(i)}(\mathbf{w}, b) = \frac{1}{2} \left(\hat{y}^{(i)} - y^{(i)}\right)^2.$$

For the optimization, mini batch SGD can be noted as:

$$(\mathbf{w},b) \leftarrow (\mathbf{w},b) - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \partial_{(\mathbf{w},b)} l^{(i)}(\mathbf{w},b).$$

#### Implementation with pytorch

##### Generate Data

```
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.tensor(np.random.normal(0, 1, (num_examples, num_inputs)), dtype=torch.float)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)
```

##### Use torch.utils.data to yield data with mini batch

```
import torch.utils.data as Data

batch_size = 10
dataset = Data.TensorDataset(features, labels)
data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)
```

##### Use torch.nn to define LR model

```
class LinearNet(nn.Module):
    def __init__(self, n_feature):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(n_feature, 1) // single output
    # forward 
    def forward(self, x):
        y = self.linear(x)
        return y

net = LinearNet(num_inputs)
```

##### Initialize weights and bias

```
from torch.nn import init

init.normal_(net.linear.weight, mean=0, std=0.01)
init.constant_(net.linear.bias, val=0)  
```

##### Define Loss and optimiser

```
import torch.optim as optim

optimizer = optim.SGD(net.parameters(), lr=0.03)
```

##### Train the model 

```
num_epochs = 3
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        output = net(X)
        l = loss(output, y.view(-1, 1))
        optimizer.zero_grad() # we have to clean the gradient
        l.backward()
        optimizer.step()
    print('epoch %d, loss: %f' % (epoch, l.item()))
```

Saddle points are those locations where all gradients of a function vannish but are neighter global nor local minimums.

Softmax
---------------------------------------------------

Linear Regression can deal with regression tasks, for example, housing price predicting. Softmax is similar to Linear regression with the fact that they are both fully-connected layer. The difference is that Softmax is used to deal with multi-classification tasks. 

#### Model, loss function, optimization

Assume that we are given a minibatch $\mathbf{X}$ of examples with feature dimensionality $d$ and batch size $n$.

Moreover, assume that we have $q$ categories (outputs). Then the minibatch features $\mathbf{X}$ are in $\mathbb{R}^{n \times d}$,

weights $\mathbf{W} \in \mathbb{R}^{d \times q}$, and the bias satisfies $\mathbf{b} \in \mathbb{R}^q$.

$$\begin{aligned}\mathbf{O} &= \mathbf{X} \mathbf{W} + \mathbf{b}, \\\hat{\mathbf{Y}} & = \mathrm{softmax}(\mathbf{O}).\end{aligned}$$  

Cross-Entropy Loss is used as the loss function. 

$$l(\mathbf{y}, \hat{\mathbf{y}}) = - \sum_j y_j \log \hat{y}_j.$$

#### Implementation with pytorch

##### Model

```
class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()
    def forward(self, x): # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1) //very useful to deal with picture data
        
net = nn.Sequential(
    # FlattenLayer(),
    # nn.Linear(num_inputs, num_outputs)
    OrderedDict([
        ('flatten', FlattenLayer()), // regard the preprocessing as a layer
        ('linear', nn.Linear(num_inputs, num_outputs))
    ])
)

```

##### Define Loss and optimiser

```
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
```

##### Train the model

```python
num_epochs, lr = 5, 0.1

def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
              params=None, lr=None, optimizer=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum()

            
            optimizer.zero_grad()

            l.backward()
            optimizer.step() 

            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))

train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, batch_size, [W, b], lr)
```

Multilayer Perceptron
---------------------------------------------------

LR and Softmax are single layer neural network models. In deep learning, multiple layers model are often used. Multilay Perceptron is one simple multilayer neural network. 

#### Model, activation function

MLP is based on single layer neural network and introduced a hidden layer. In order to avoid the affine transformation which makes hidden layer liner, activation function is introduced to do non-liner transformation. Some widly used activation functions are $Relu$, $Sigmoid$, $tanh$. The MLP model can be noted as:

$$\begin{aligned}\mathbf{H}_1 & = \sigma(\mathbf{W}_1 \mathbf{X} + \mathbf{b}_1), \\ \mathbf{O} & = \mathbf{H}_1(\mathbf{W}_3 \mathbf{H}_2 + \mathbf{b}_3)\end{aligned}$$

#### Implementation with pytorch

##### Model

```
net = nn.Sequential(
        FlattenLayer(),
        nn.Linear(num_inputs, num_hiddens),
        nn.ReLU(),
        nn.Linear(num_hiddens, num_outputs), 
        )
```

##### Define loss, optimiser, training

```
loss = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(net.parameters(), lr=0.5)

num_epochs = 5
train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)
```

### Reference

Dive into Deep Learning: https://d2l.ai/index.html 

