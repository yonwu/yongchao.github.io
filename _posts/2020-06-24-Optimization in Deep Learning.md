---
layout:     post
title:      Optimization in Deep Learning
date:       2020-06-24
author:     Yongchao Wu
header-img: img/post-main.jpg
catalog: true

---

Optimization concept
=================

Optimization is used in deep learning to minimise/maximize an objecive function(often a loss function), after a optimisation, the 'best' weights that can be concluded and thus, the neural network could be established to do prediction. There are several optimization algorithms, like convexity, gradient descent, stochastic gradient descent, Adagrad, RMSProp, Adadelta, Adam, etc. In this blog, these optimization algorithms will be briefly discussed. 

Relationship between optimization and deep learning
---------------------------------------------------

First of all, the goals of optimization and deep learning is different. Optimization is primarily concerned about minimizing an objective function based on training dataset, whereas deep learning is more concerned with finding a suitable network model to to prediction. Thus, it is obvious to know that optimization cares about reducing training errors, on the other hand, deep learning cares more about the overall performance of the model(training and testing performance). This also means, optimization will possibly bring overfitting.

Challenges that optimization faces
---------------------------------------------------

The main challenges that optimization faces in deep learning is local minima, saddle points and vanishing gradients. 

#### Local Minima

In deep learning, there could be multiple local optima. With a numerical optimization solution, the objective functino might only be minimized locally rather than globally.

#### Saddle Points

Saddle points are those locations where all gradients of a function vannish but are neighter global nor local minimums.

#### Vanishing gradients

Vanishing gradients often result in insidious problem in deep learning. If the gradient of a function is closed to nil, the optimization will get stuck and low down the deep learning progress. 

Optimization algorithms dealing with challenges
---------------------------------------------------

Optimization faces a lot of chanllenges, fortunately there are some alsogirthms that can help, some of these algorithms will be introduced as below:

#### Gradient Descent

Suppose we have a real-value function $f$, according to Taylor expansion:



$$f(x + \epsilon) = f(x) + \epsilon f'(x) + \mathcal{O}(\epsilon^2).$$



if we note  $\epsilon$ as $\eta f'(x)$ , $\eta$ can be regarded as learning rate, we can get:



$$f(x - \eta f'(x)) = f(x) - \eta f'^2(x) + \mathcal{O}(\eta^2 f'^2(x)).$$



Thus, it is obvious:



$$f(x - \eta f'(x)) \lessapprox f(x).$$



Which means, if we use $x \leftarrow x - \eta f'(x)$ to iterate x, the value of function $f(x)$ might decline. In deeplearning case, the loss might decline. 

#### Stochastic Gradient Descent

Assuming the loss function for a deep learning task of n examples of training data, the objective function is:



$$f(\mathbf{x}) = \frac{1}{n} \sum_{i = 1}^n f_i(\mathbf{x}).$$



So , the gradient is:



$$\nabla f(\mathbf{x}) = \frac{1}{n} \sum_{i = 1}^n \nabla f_i(\mathbf{x}).$$



We can see that, the computing time complexity is $O(n)$ which becomes very large with large dataset. Stochastic Gradient Descent is used to slove this problem. Basically, at each iteration of SGD, we uniformly sample an index fro data instances at random, and then compute the gradient to upadte x:



$$\mathbf{x} \leftarrow \mathbf{x} - \eta \nabla f_i(\mathbf{x}).$$



Thus, the computing complexity is $O(1)$. Thus with more training data, SGD will improve the learning speed. And because random sampling is introduce, a carefully choice of learning rate is important. 



```
torch.optim.SGD
```



#### Minibatch SGD

Minibatch furtuher improves the computation efficiency. Minibatch stochastic gradient descent offers the best of both worlds: computational and statistical efficiency.



#### Momentum

Momentum takes advantages of the math concept exponentially weighted moving average aiming at reaching the  optimal solution fast with a relative large learning rate without missing it.  

```
torch.optim.SGD, {'lr': 0.004, 'momentum': 0.9}
```



#### AdaGrad

The AdaGrad algorithm adjusts the learning rate in each dimension according to the size of the gradient value of the independent variable in each dimension, thereby avoiding the problem that the unified learning rate is difficult to adapt to all dimensions.

```
torch.optim.Adagrad, {'lr': 0.1}
```



#### RMSProp

In AdaGrad, When the learning rate drops faster in the early iterations and the current solution is still poor, the AdaGrad algorithm may find it difficult to find a useful solution because the learning rate is too small in the later iterations. RMSProp also take advantage of exponentially weighted moving average, thus making the learning rate of each element of the independent variable no longer decreases (or remains constant) during the iteration

```
torch.optim.RMSprop, {'lr': 0.01, 'alpha': 0.9}
```



#### AdaDelta

In addition to the RMSProp algorithm, another commonly used optimization algorithm, the AdaDelta algorithm, also improves on the problem that the AdaGrad algorithm may find it difficult to find useful solutions later in the iteration. The AdaDelta algorithm does not have the hyperparameter learning rate, but maintains an additional state variable to calculate the amount of change in the independent variable.

```
torch.optim.Adadelta, {'rho': 0.9}
```



#### Adam

Based on the RMSProp algorithm, the Adam algorithm also does an exponentially weighted moving average for small batches of random gradients.
The Adam algorithm uses deviation correction.

```
torch.optim.Adam, {'lr': 0.01}
```

