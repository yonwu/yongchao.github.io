---
layout:     post
title:      A brief Introduction of LSTM Based Tagger
date:       2020-03-28
author:     Yongchao Wu
header-img: img/post-bg-cook.jpg
catalog: true

---

LSTM based tagger
=================

Long Short-Term Memory (LSTM)[@HochSchm97] is a special type of
Recurrent Neural Network (RNN). A RNN is a network that contains a cycle
within the network connections which means one recurrent connection is
included in the hidden layer. Thus, the activation value of the hidden
layer depends both on the current input and the activation value of the
hidden layer of a previous time step. In simple RNN, there is problem of
making use of larger context, which means the information the output of
the network tends to be decided by information in a sequence of near
distant. LSTM is introduced to address this problem by context
management mechanism.

learning algorithms, parameters and hyperparameters
---------------------------------------------------

For simple RNN, suppose we have a sequence of input is 
$$w_1...w_t...w_T$$, at time step $$t$$ , the hidden layer activation value
$$h_t$$ depens on the current input $$x_t$$ and the hidden layer activation
value at previous time step $$t-1$$ with $g$ as activation function:

$$h_t = g(Uh_{t-1} + Wx_t)$$ 

and the output vector with $f$ as
activation function is: 

$$y_t = f(Vh_t)$$ 

to apply the softmax function
over the output vector: 

$$y_t = softmax(Vh_t)$$ 

where, $$W, U, V$$ are
weight matrix that are shared across time. Thus, the model should learn
U, V, W of the network from the training data. The process of learning
these weight matrixes is similar to Neural Network: through a loss
function and backpropagation(gradients descent) to adjust weights in the
simple RNN network.

Based on the simple RNN, LSTM adds a context layer that deploys 'gates'
to manage context information. These gates have addtional weights and
and process the input, previous hidden layer and previous context
layers. In the gates, there are sigmoid activation functions that can
make binary output 1 or 0. Output 1 means the context information should
be kept while output 0 means it should be dropped.

The first gate that we have is a \"forget gate\". Its purpose is to
erase some context information. Specifically, it will process the
previous hidden layer and current input through extra weights and
sigmoid to get the \"gate vector\" output and then multiply it by
context vector to remove unnecessary information. $$\begin{aligned}
 &forget_t = \sigma(U_{forgeth}h{t-1} + W_{forget}x_t) \\
 &k_t = c_{t-1} forget_t
\end{aligned}$$ Next, the job is to compute the information from
previous hidden state and current inputs, and then multiply by a \"add
gate\" to add the information into current context:

$$\begin{aligned}
 &g_t = \tanh (U_gh_{t-1} + W_gx_t) \\
 &add_t = \sigma(U_{add}h_{t-1} + W_{add}x_t) \\
 &j_t = g_{t} add_t \\
 &c_t = j_t + k_t
\end{aligned}$$ 

Next, another \"output gate\" is added to control the
output information for the current hidden state: 

$$\begin{aligned}
 &output_t = \sigma(U_{outputh}h{t-1} + W_{output}x_t) \\
 &h_t = output_{t} \tanh(c_t)
\end{aligned}$$

From the above explanations, we know that basic weights $U, W, V$
together with context layer with extral weights
$U_{forget}, W_{forget}, U_{add}, W_{add}, U_{output}, W_{output}$ are
also needed to be learned by training. And the learning process is
similar to simple RNN, through a loss function and backpropagation
(gradient descent) to adjust weights.

After understanding the learning algorithms, the parameters and
hyperparameters are then easier to explain. The parameters of the LSTM
tagger are: 

$$\begin{aligned}
U, W, V, U_{forget}, W_{forget}, U_{add}, W_{add}, U_{output}, W_{output}
\end{aligned}$$ 

the hyperparameters are those that can be tuned to get a
better accuracy, like EMBEDDING dimension, HIDDEN dimension loss
function, regularization and so on.

LSTM tagger prediction algorithms
---------------------------------

The training process is complex, meanwhile the prediction of LSTM is
similar to regular multiclass classification job. Given a learned
network, we can follow the process below: 

$$h_t = g(Uh_{t-1} + Wx_t)$$

$$y_t = f(Vh_t)$$

$$y_t = softmax(Vh_t)$$

A softmax will help to pick the best result from the output and make a
prediction.

LSTM tagger discussion
----------------------

As showed in the lab, the LSTM result is underwhelming due to training
data size limitations and the fact that they are not tuned. We can see
that, LSTM introduced a number of parameters, which will bring high
traing cost. To decrease the training cost, one idea is to reduce the
number of gates from LSTM, thus Gated Recurrent
Units(GRUs)[@cho-etal-2014-learning] were introduced.Another idea is to
reply on attention mechanism to establish global dependencies between
input and output[@vaswani_attention_2017] .