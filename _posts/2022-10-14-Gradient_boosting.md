---
layout: post
title: Gradient boosting
date: 2022-10-14
description: Day 22
tags: review
categories: ml, boosting, gradient
---
While looking at [ResNets](https://achchg.github.io/blog/2022/ResNet/), One thought came to my mind - "So isn't this boosting?". An immediate next step of researching this topic led me to this stackoverflow [discussion](https://stats.stackexchange.com/questions/214273/are-residual-networks-related-to-gradient-boosting), and later a review session of gradient boosting algorithms.

#### Gradient boosting
Actually the Wikipedia [page](https://en.wikipedia.org/wiki/Gradient_boosting) of gradient boosting summarizes the algorithm fairly well. High-level description of what it is with my own words: 

**"An additive model of weaker trees with forward feature selection trained by gradient descent method that aims to learn to avoid making errors made in previous stages (trees)."**

How gradient boosting works:
1. Define a boosting tree ($$F_M$$) that we aim for $$M$$ stages (M weak learners/trees):
   $$\hat{F}_M(x_i) = \Sigma_{m=1}^M \alpha_m h_m(x_i) + \text{const.}$$
where $$\alpha_m$$ is the learning rate.

2. Define loss at the $$m^{th}$$ stage/base learner: 
   - loss function: $$L(y, F_m(x))$$
   - example base learner: $$h_m(x_i) = y_i - \hat{y}_{i,m} = y_i - F_m(x_i)$$
   - goal is to minimize the loss
  

3. At stage 0, as there was no stage before it. Therefore, in the very first tree, we are fitting the tree with $$y_i$$ directly:
   $$F_0(x_i) = \arg \min_\alpha \Sigma_{i=1}^n L(y_i, \alpha)$$

4. At stage $$m$$ (where $$ m \neq 0 $$), we are fitting the $$m^{th}$$ tree with the residual:
   $$F_m(x_i) = F_{m-1}(x_i) + \arg \min_{h_m} \Sigma_{i=1}^n L(y_i, F_{m-1}(x_i) + \alpha h_m(x_i))$$

5. Repeat #4 and keep updating the model until convergence ($$F_m(x_i) = F_{m-1}(x_i) + \alpha_mh_m(x_i)$$).

Also, nicely explained source for boosting algorithm by [CS 329P : Practical Machine Learning (2021 Fall)](https://c.d2l.ai/stanford-cs329p/_static/pdfs/cs329p_slides_7_3.pdf):

{% highlight python linenos %} class GradientBoosting:
    def __init__(self, base_learner, n_learners, learning_rate):
        self.learners = [clone(base_learner) for _ in range(n_learners)]
        self.lr = learning_rate
    def fit(self, X, y):
        residual = y.copy()
        for learner in self.learners:
            learner.fit(X, residual)
            residual -= self.lr * learner.predict(X)
    def predict(self,X):
        preds = [learner.predict(X) for learner in self.learners]
        return np.array(preds).sum(axis=0) * self.lr

{% endhighlight %}

Here, we can leverage different base_learner (e.g. regression or classification models with differen objective functions)

##### Gradient boosting regression

**Loss function:** MSE!

$$
\begin{align*}
L(y, F_m(x_i)) & = \frac{1}{n}\Sigma_{i=1}^n(F_m(x_i)-y_i)^2\\
\arg \min_{F_m} \Sigma_{i=1}^n L(y_i, F_m) & = -\frac{\partial L(y, F_m(x_i))}{\partial F_m} \propto \Sigma_{i=1}^n(y_i-F_m(x_i)) \rightarrow \boxed{r_m = \text{pseudo-residual}}\\
\arg \min_{\gamma} L(y, F_{m-1}(x_i)+\gamma) &\approx \arg \min_{\gamma} [L(y, F_{m-1}(x_i)) +\frac{\partial L(y, F_{m-1}(x_i))}{\partial F}\gamma +\frac{1}{2}\frac{\partial^2 L(y, F_{m-1}(x_i))}{\partial F^2}\gamma^2]\\
& = 0 + \frac{\partial L(y, F_{m-1}(x_i))}{\partial F} + \frac{\partial^2 L(y, F_{m-1}(x_i))}{\partial F^2}\gamma \stackrel{\text{set}}{=} 0\\
\gamma & = - \frac{\frac{\partial L(y, F_{m-1}(x_i))}{\partial F}}{\frac{\partial^2 L(y, F_{m-1}(x_i))}{\partial F^2}} = \boxed{\frac{\Sigma_{i=1}^n y_i - F_m(x_i)}{n}}
\end{align*}
$$

**Source [code](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#sklearn.ensemble.GradientBoostingRegressor) from sklearn**


Great video materials:
- [StatQuest](https://www.youtube.com/watch?v=2xudPOBz-vs)

##### Gradient boosting classification

**Loss function:** Can use the same as logistic regression, details illustrated [here](https://achchg.github.io/blog/2022/Stochastic_gradient_descent/).

$$
\begin{align*}
L(y, F_m(x_i)) & = \Pi_{i=1}^n F_m(x_i)^y_i(1-F_m(x_i))^{1-y_i}\\
\ell(y, F_m(x_i)) & = \Sigma_{i=1}^n [y_i\log(F_m(x_i)) + (1-y_i)\log(1-F_m(x_i))] \\
& = \Sigma_{i=1}^n y_i\log(\frac{F_m(x_i)}{1-F_m(x_i)}) + \log(1-F_m(x_i))\\
& = \Sigma_{i=1}^n y_i\log(\text{Odds}) + \log(1-\frac{\exp(\log(\text{Odds}))}{1+\exp(\log(\text{Odds}))})\\
& = \Sigma_{i=1}^n y_i\log(\text{Odds}) - \log(1+\exp(\log(\text{Odds})))\\
\arg \min_{\log(\text{Odds})} \ell(y, F_m(x_i)) &= \frac{\partial \ell(y, F_m(x_i))}{\partial \log(\text{Odds})} 
= \Sigma_{i=1}^n y_i - \frac{\exp(\log(\text{Odds}))}{1+\exp(\log(\text{Odds}))} \\ &= \Sigma_{i=1}^n y_i - F_m(x_i) \rightarrow \boxed{r_m = \text{pseudo-residual}}\\
\arg \min_{\gamma} \ell(y, F_{m-1}(x_i)+\gamma) &\approx \arg \min_{\gamma} [\ell(y, F_{m-1}(x_i)) +\frac{\partial \ell(y, F_{m-1}(x_i))}{\partial F}\gamma +\frac{1}{2}\frac{\partial^2 \ell(y, F_{m-1}(x_i))}{\partial F^2}\gamma^2]\\
& = 0 + \frac{\partial \ell(y, F_{m-1}(x_i))}{\partial F} + \frac{\partial^2 \ell(y, F_{m-1}(x_i))}{\partial F^2}\gamma \stackrel{\text{set}}{=} 0\\
\gamma & = - \frac{\frac{\partial \ell(y, F_{m-1}(x_i))}{\partial F}}{\frac{\partial^2 \ell(y, F_{m-1}(x_i))}{\partial F^2}} = \boxed{\frac{\Sigma_{i=1}^n y_i - F_m(x_i)}{\Sigma_{i=1}^n F_m(x_i)*(1-F_m(x_i))}}
\end{align*} 
$$


**Source [code](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier) from sklearn**

Great video materials:
- [StatQuest](https://www.youtube.com/watch?v=StWY5QWMXCw)
  
<!-- 
Example notebook with above example can be found [here](https://github.com/achchg/achchg.github.io/blob/master/jupyternb/2022-09-29-Stochastic_gradient_descent.ipynb). -->
