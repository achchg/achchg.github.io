---
layout: post
title: Monte Carlo simulation
date: 2022-09-27
description: Day 5
tags: review
categories: probability mc_simulation
---
A [question](https://users.aber.ac.uk/jcf12/teaching/montecarlo/) that I read today: "How to estimate $$\pi$$?"

You'd find the thorough overview in the above link (and maybe a lot more blogs available online). I also gave my brain a little exercise today:

#### We can leverage MC simulation!
This is a very straight-forward method that everyone might come up with. As shown below, we know that the relative ratio of the areas of a circle with radius ($$r$$) and a square of width ($$2r$$) can be expressed like:

$$
\text{circle}:\text{square} = \pi R^2 : (2R)^2 = \pi : 4
$$

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/square.jpg" title="example image" %}
    </div>
</div>

Therefore, on the Cartesian coordinate, if we can sample N samples of (x, y) pairs uniformly from the square and to keep track of the common samples that also fell into the circle (threshold: $$x^2 + y^2 < R^2$$), we can estimate $$\pi$$ as:

$$\hat{\pi} = 4 \times \frac{\text{# of common samples also appears in the circle}}{\text{# of samples in the square}}$$. Here is part of my sample codes:

{% highlight python linenos %} 
R = 1
N = 1000

def estimate_pi(R, N):
    """
    Estimate pi from simulation from circle
    N: Number of simulation/samples
    R: radius of the circle
    """
    
    x = np.random.uniform(-R, R, N)
    y = np.random.uniform(-R, R, N)
    in_circle = x**2 + y**2

    circle_count = 0
    for i in range(N):
        if in_circle[i] < R**2:
            circle_count += 1

    pi_hat = 4 * circle_count / N
    
    return pi_hat
{% endhighlight %}

Example notebook with above example can be found [here](https://github.com/achchg/achchg.github.io/blob/master/assets/jupyternb/2022-09-27-MCsimulation.ipynb).

