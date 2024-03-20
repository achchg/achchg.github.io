---
layout: post
title: Inverse transform sampling
date: 2022-09-28
description: Day 6
tags: review
categories: probability
---
Continuing from Day 5, I read another interesting [post](https://stackoverflow.com/questions/5837572/generate-a-random-point-within-a-circle-uniformly) along the same line on stackoverflow. However, this time, slightly reforming the question as "How to generate a random point within a circle (uniformly)?" Eventhough the answer from **aioobe** on stackoverflow explained it very well, I still decided to summarize and document for my future self and also to review the concept of **Inverse transform sampling**.

In order to dirctly sample from a circle without using a square and Monte Carlo simulation, we can try to gather the cumulative density function (CDF, or F(r)) and apply Inverse transform sampling to solve it.

#### How to come up with CDF?
As explained in the above stackoverflow, to uniformly sample from a circle we cannot solely sample $$r$$ uniformly from $$(0, R)$$ and $$\theta$$ from $$(0, 2\pi)$$. This is because as r is away from the center $$(0, 0)$$, the area being summarized given an incremental increase in $$r$$ (e.g. $$d$$) also get larger, meaning that we'd need more samples to be randomly chosen, as the r increases from d to 2d, to achieve uniformly distributed sampling. 

If we consider the increase in circumference when the random radius increases from $$d$$ to $$2d$$, the increase in the length of the circumference from $$2 \pi d$$ to $$4\pi d$$ suggests that the instaneous increase in the probability is linearly to increase in random variable $$r$$ as d (a.k.a. $$f(r) \propto r $$)

With the property of probability:

$$
\int_0^R f(r) dr = \int_0^R kr dr = 1
$$

We know that:

$$
k\frac{R^2}{2} = 1, k = \frac{2}{R^2}
$$

Therefore:

$$
f(r) = \frac{2}{R^2}r, F(r) = \frac{2}{R^2}\frac{r^2}{2} = \frac{r^2}{R^2}
$$


#### Leveraging Inverse transform sampling!
You might like to check the [wikipedia page](https://en.wikipedia.org/wiki/Inverse_transform_sampling), but the concept is relatively simple: It might be hard to generate a random sample directly. However, if we know its CDF ($$F(x)$$), we can usually easily sample randomly from a uniform distribution (u) and apply $$F^-1(u)$$ to be a random sample of X! In math:

$$F_x(x) = Pr(X \leq x) = u$$

where 

$$u \sim U(0, 1)$$

then 

$$x = F_X^{-1}(u)$$


#### Linking both pieces together
Here if we sample $$u$$ uniformly from $$U(0, 1)$$

$$F(r) = \frac{r^2}{R^2} = u$$

$$r^2 = R^2 u$$

As a result:
- We will sample r with the following algorithm:

$$r = F^{-1}(u) = R\sqrt{u}$$

- And we will uniformly sample $$\theta$$ as:

$$\theta = 2\pi u_1$$

where 

$$u_1 \sim U(0, 1)$$

- Transformation from polar to Cartesian coordinate:
  
$$ x = r\cos(\theta)$$

$$ y = r\sin(\theta)$$

{% highlight python linenos %} 
R = 1
N = 1000

def random_circle_sample(R, N):
    """
    Random unifromly sample N samples from a circle
    N: Number of simulation/samples
    R: radius of the circle
    """
    
    u = np.random.uniform(0, 1, N)
    theta = 2 * math.pi * np.random.uniform(0, 1, N)
    r = R * np.sqrt(u)
    
    y = r*np.sin(theta)
    x = r*np.cos(theta)
    
    plt.figure(figsize=(8, 8), dpi=80)
    plt.scatter(x, y)
    plt.show()
{% endhighlight %}

Example notebook with above example can be found [here](https://github.com/achchg/achchg.github.io/blob/master/assets/jupyternb/2022-09-28-Inverse_transform_sampling.ipynb).

