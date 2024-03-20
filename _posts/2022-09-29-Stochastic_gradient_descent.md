---
layout: post
title: Stochastic gradient descent
date: 2022-09-29
description: Day 7 - 8
tags: review
categories: probability gradient
---
I'm reviewing the gradient descent and stochastic gradient descent again with NLP in deep learning, which are common machine learning parameter optimization methods well covered by packages these days and we normally use them hand-waving. Here, I'll use logistic regression as my example:

#### (Stochastic) Gradient descent
Our often goal to develop a ML model is to make better prediction (i.e. minimize the cost of a model, $$J(\theta)$$), where $$\theta$$ is a single parameter of our model here. In gradient descent, we'd:
1. randomly initiate a value for $$\theta$$ at start
2. use the whole sample space (or a batch of data) to calculate the cost of a model under the $$\theta$$ estimation at iteration $$j$$
3. make the $$\theta$$ estimation at the next iteration with the following algorithm (where $$\alpha$$ is the learning rate): $$\theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j}J(\theta)$$

4. repeat #3 until $$\theta_j$$ is converged ($$J(\theta)$$ is minimized)

In stochastic gradient descent, instead, we'd repeat #1 - #3 at every single sample to approximate the gradient with single data point. This allows us to update the gradient and cost incrementally fairly quickly.

#### Cost of logistic regression
If considering a logistic regression model with the simpliest setup to classify a binary outcome $$y$$ that follows a bernouli trial:

$$
\begin{align*}
y_i & = \begin{cases}
      1 & \text{, p}\\
      0 & \text{, 1-p}
    \end{cases} \\
y_i & \sim \text{Bernouli}(p)\\
\\
f(y) & = p^y(1-p)^{1-y}\\
\\
L(y) & = \Pi_{i=1}^n p^y_i(1-p)^{1-y_i}
\end{align*}
$$


$$
\begin{align*}
\text{logit}(p) & = \alpha + \beta x_i \\
p_i & = \frac{\exp(\alpha + \beta x_i)}{1 + \exp(\alpha + \beta x_i)}
\end{align*}
$$

Therefore, we can derive the log-likelihood function:

$$
\begin{align*}
\ell(y) & = \Sigma_{i=1}^n y_i\log(p_i) + (1-y_i)\log(1-p_i)\\
& = \Sigma_{i=1}^n \log(1-p_i) + y_i\log(\frac{p_i}{1-p_i})\\
& = \Sigma_{i=1}^n -\log(1 + \exp(\alpha + \beta x_i)) + y_i(\alpha + \beta x_i)\\
\end{align*}
$$

If we use the negative average log-likelihood (cross entropy) as the cost function (J) would be:

$$
J = -\frac{1}{n}\Sigma_{i=1}^n y_i(\alpha + \beta x_i) + \log(1 + \exp(\alpha + \beta x_i))
$$

And the gradients of $$\alpha$$ and $$\beta$$ are:

$$
\begin{align*}
\frac{\partial}{\partial \alpha} & = -\frac{1}{n}\Sigma_{i=1}^n y_i +\frac{\exp(\alpha + \beta x_i)}{1 + \exp(\alpha + \beta x_i)} = \frac{1}{n}\Sigma_{i=1}^n(p_i-y_i) = \frac{1}{n}\Sigma_{i=1}^n(\hat{y_i}-y_i)\\
\frac{\partial}{\partial \beta} & = -\frac{1}{n}\Sigma_{i=1}^n y_ix_i +\frac{\exp(\alpha + \beta x_i) x_i}{1 + \exp(\alpha + \beta x_i)}= \frac{1}{n}\Sigma_{i=1}^n x_i(p_i-y_i) = \frac{1}{n}\Sigma_{i=1}^n x_i(\hat{y_i}-y_i)
\end{align*}
$$


With gradient descent, we'd optimize $$\alpha$$ and $$\beta$$ with the following algorithm and learning rates:

$$
\begin{align*}
\alpha_j & := \alpha_j - \eta_1 \frac{1}{n}\Sigma_{i=1}^n(\hat{y_i}-y_i)\\
\beta_j & := \beta_j - \eta_2 \frac{1}{n}\Sigma_{i=1}^nx_i(\hat{y_i}-y_i)
\end{align*}
$$

Pseudo code that I wrote from CS224 assignment. Here x is the parameter to be optimized and step is the learning rate:
{% highlight python linenos %} for iter in range(start_iter + 1, iterations + 1):
        loss = None
        loss, grad = f(x)
        x -= step * grad
{% endhighlight %}

Example notebook with above example can be found [here](https://github.com/achchg/achchg.github.io/blob/master/assets/jupyternb/2022-09-29-Stochastic_gradient_descent.ipynb).

Great reference material can be found [here](https://web.stanford.edu/~jurafsky/slp3/5.pdf)
