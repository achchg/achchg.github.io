---
layout: post
title: Adam (adaptive moment estimation) optimizer
date: 2022-10-04
description: Day 12
tags: review
categories: optimization gradient
---
Reviewing [week 3 assignment](https://web.stanford.edu/class/cs224n/assignments/a3_handout.pdf) of NLP with deep learning brought Adam optimization algorithm back to my attention. Here I'd summarize what I did in my homework 3 for my future reference:

#### Adam optimizer
From standard SGD, we would use a mini-batch (e.g. single sample) of data in the update rule below for updating the $$J(\theta)$$: 

$$\theta := \theta - \alpha \nabla_\theta J(\theta)$$

where $$\alpha$$ is the learning rate and $$\nabla_\theta J(\theta)$$ represent the partial derivatives of the cost function wrt $$\theta$$. 

Adam optimization, in addition, takes 2 additional steps beyond SGD:

##### Update biased first order moment estimate

$$
\begin{align*}
m & := \beta_1 m + (1 − \beta_1)\nabla_\theta J(\theta)\\
\theta & := \theta - \alpha m
\end{align*}
$$

As $$m$$ is set as a weighted average of the rolling gradient average of the **previous** iterations and the gradient of the **current** iteration, we can expect the **momentum** step making the gradient descent update smoother than that of the SGD (which only consider the **current** iteration). The current gradient ($$\nabla_{\theta} J(\theta)$$) will be weighted larger than the individual gradients after k ($$\frac{\beta_1}{1-\beta_1}$$) iterations, as the individual previous gradients has a weight of $$\frac{\beta_1}{k}$$ (where k = num of past iterations).

##### Update biased second order raw moment estimate

$$
\begin{align*}
v & := \beta_2 v + (1 − \beta_2)(\nabla_\theta J(\theta) \odot \nabla_\theta J(\theta))\\
\theta & := \theta - \alpha \frac{m}{\sqrt{v}}
\end{align*}
$$

As $$m$$ is divided by the $$\sqrt{v}$$ (the gradient of Adam), the gradients that have smaller gradient magnitude ($$v$$) will get larger updates. As $$v$$ is derived from squared of $$\nabla_{\theta} J(\theta)$$, $$v$$ is usually associated with smaller gradient. This might help further smoothing out the gradient descent from the **momentum** step, by giving larger weights to the smaller gradients when the SGD update does not guarantee continuous descending in the gradients.


##### Note: Method of moment
[Method of moment](https://en.wikipedia.org/wiki/Method_of_moments_(statistics)) in statistics implies the following:

The $$k^{th}$$ moment of a random variable X with its pdf, $$f(x)$$ can be expressed as:

$$E(X^k) = \int_X x^k f(x) dx$$

Therefore, the first moment of X is $$E(X)$$, which is the mean of the distribution; and the second moment of X is $$E(X^2)$$, which is the sum of mean squared and the variance ($$\text{Var}(X) = E(X^2) - E(X)^2$$).
<!-- 
Example notebook with above example can be found [here](https://github.com/achchg/achchg.github.io/blob/master/jupyternb/2022-09-29-Stochastic_gradient_descent.ipynb). -->

Original Adam paper is [here](https://arxiv.org/pdf/1412.6980.pdf); Helpful [documentation](https://gregorygundersen.com/blog/2020/04/11/moments/) of moment statistics.
