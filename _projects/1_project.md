---
layout: distill
title: Centralizing Real World Data with Machines
description: This post was summarized from my final project @Stanford CS229.
img: assets/img/exp123456.png
importance: 1
category: learning
bibliography: rwd_gan.bib
related_publications: true
date: 2021-12-05
---

The paper aimed to propose a framework that leverages machine learning methods to utilize information from multiple data sources, with the ultimate goal being able to generate a de-biased data layer that allows health data scientists/researchers to perform analyses on. 

As an demonstration of the concept, I assumed a hypothetical goal:

**To estimate the share of a particular item (A) against a list of competing items, possibly given a set of features.**

This looks like a typical problem we'd solve with statistical inference. I tried to tackle the prompt with the following three perspectives and their impact on the need of data centralization:
1. **Data biasness:** when multiple unbiased/biased real-world datasets are available, is a direct pooling of all datasets just always better?
2. **Model biasness:** when applying algorithms in different data scenarios, what is the scale of model prediction error on top of the data biasness?
3. **Synthetic biasness:** whether leveraging Generative Adversarial Networks (GAN) could generate synthetic datasets that allows us making direct unbiased model inference generated learning from biased data sources compared to baseline models

### Simulated Data

We simulated different real world data scenarios leveraging the Fashion MNIST dataset <d-cite key="xiao2017fashionmnist"></d-cite> because of its high-dimensional feature space and that the target variable being multi-class. 
- Overall, the dataset contains 60,000 samples with 28x28 dimension that describe the pixel graph of an individual item sample; in which we assumed these data as the true population distribution of the market (if we know what the true is). 
- As illustrated in **Table 1**, We assumed "T-shirt" as the target product item (A) of interest. There were eight other competitor products on the market and one additional group as "All others". We'd discuss the paper in terms of Product A vs. competitor products. With the true target market share of interest being 10%.
<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/table1_gan.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Table 1. Data & selection bias assumptions within individual table.
</div>  

For each study scenario, We sampled three datasets (**DS-1**, **2**, **3**) randomly with different size assumptions setting specific seed from the true population. Random Gaussian noise within individual datasets was assumed as the embedded data batch effect beyond the random sample noises. Selection biases within individual datasets were assumed followed the specifying distribution in **Table 1**.

#### Unbiased datasets with random Gaussian noise
To illustrate the underlying research question and the unbiased sample dataset in equation:

$$Y_{j}^{(i)} \sim b_{data_j}^{(i)} + \beta X^{(i)} + \epsilon^{(i)}_j$$

where

$$Y^{(i)}_j \sim \textbf{Multinomial}(n, p_1, ..., p_{10})$$ **:** n is sample size, and $p_1$ to $p_{10}$ are proportions of the 10 product labels

$$X^{(i)}_j  \sim $$ Fashion MNIST (28x28) feature space

$$\epsilon^{(i)}_j \sim $$ Fashion MNIST (28x28) sample noise

$$b_{data_j} \sim \mathcal{N}(0, \Sigma_{data_j})$$**:** assumed batch-effect ($\Sigma_{data_1}$, $\Sigma_{data_2}$, $\Sigma_{data_3}$) $=$ (0.1, 0.2, 0.1)




#### Biased datasets with random Gaussian noise
Additional category selection biasness was assumed and specified in \textbf{Table 1}, which was illustrated as variable $Z_j$ here:

$$Y_{j}^{(i)} \sim b_{data_j}^{(i)} + \beta X^{(i)} + Z_j^{(i)} + \epsilon^{(i)}_j$$

where

$$
Z_j^{(i)} | K = k \sim \textbf{Bernouli}(p_{k,j})$$**:**  k is the hidden category in real world that attributed to selection bias within a given data set j (if $Z_j$ = 1 then sample was observed); $$\mbox{Item}_1 ... \mbox{Item}_{10}$$  were summarized by category K without additional noise.

The biased datasets were aimed to simulate the real world scenarios that datasets observed are often non-random subset of the true population of interest. Specifically, **DS-1** was assumed as a biased set toward non-target classes; **DS-2** as an unbiased set and **DS-3** as a biased set toward target class.

### Methods

**Algorithms that predict the outcome of interest.** This component aimed to study the scenario when product labels are unknown and a model is used to predict the product label. Different state-of-the-art approaches were studied here given embeded data biasness and batch noise. Max absolute scaling <d-cite key="scikit-learn"></d-cite> was implemented before fitting the models.Specifically, a logistic regression without adjusting data group and a logistic regression adjusted with additive data group effect as categorical variable.

**The algorithm that generates the centralized data distribution and predict the outcome of interest.** We utilized and modified a GAN module ([SGAN](https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/sgan/sgan.py)) from an open-sourced Github package (Pytorch-GAN) <d-cite key="pytorchgan"></d-cite> in Pytorch <d-cite key="NEURIPS2019_9015"></d-cite>.

SGAN is a semi-supervised GAN model that expanded from the orignal GAN model <d-cite key="goodfellow"></d-cite> which are aiming to predict both the class label (e.g. K classes, where k = 10 in our example) and whether the data is real or generated.<d-cite key="odena2016semisupervised"></d-cite> To summarize the default SGAN model, the discriminator loss was separated into an adversarial (a sigmoid activation function to classify fake vs. real) and an auxiliary (a soft-max activation function to classify the labels) portion. The generator went through two layers of batch-normalization and up-sampling processed before 2-D convoluted networks were applied. A Tanh activation function was applied at the end to ensure the generated values fell between (-1, 1). 



### Experiments
Three assumptions on DS-1 (data distribution biased toward non-target), DS-2 (unbiased dataset) and DS-3 (data distribution biased toward target) mixture with 10%, 50% and 90% population set added as DS-2 were experimented. The three data mixtures were purposely selected to ensure that the overall mixture sizes were similar to one another and to avoid additional noise attributed from different size of the samples. The test sets were what we held out as unbiased sets from the individual datasets before selection biasness applied. 

**Experiment \#1 (Data Mixture 1): 50% DS-1, 50% DS-2, 50% DS-3**

**Experiment \#2 (Data Mixture 2): 90% DS-1, 10% DS-2, 50% DS-3**

**Experiment \#3 (Data Mixture 3): 50% DS-1, 10% DS-2, 90% DS-3**


### Analysis
<div class="row">
    <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid loading="eager" path="assets/img/table2_gan.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Table 2. Comparison of data mixture in biasness and model accuracy
</div> 
**Key-takeaway \#1: A row-stacking data mixture would always benefit direct inference when the target label was known.**

- Quite intuitive as more data in general helps in informing less biased decision, as shown in **Table 2** when comparing values in the first set of comparisons.
- As observed in scenarios of experiment #2 and #3 (more likely to be the case of RWD where the directions of biasness are likely unknown and at different scale in sets of data), the benefit of aggregating the datasets was further highlighted.
    

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/exp123456.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Figure 1. Example dataset bias shown with different sample size and different assumed population variation
</div> 
**Key-takeaway \#2: As the less biased dataset dominating the data mixture, we made less biased data inference.**

- As observed in **Figure 2**, when the overall data size is fixed, the blue and orange trends tend to be closer to zero as the size of data mixture 2 became larger (green background expanded).

**Key-takeaway \#3: Model performance among different data mixtures are all within a range of 85 - 87% accuracy. However, we observed that the model performance increased as the size of the unbiased data mixture - 2 increased. This might indicate that the increase in unbiased dataset in model training avoided the model from overfitting.**

**Key-takeaway \#4: We encountered common GAN model training issue that the discriminator for label classification became too strong and the generator gave-up on improving its data generation to fool discriminator.**

- Therefore, the model accuracies reported in **Table 2** could only been considered as results of the supervised learning models classifying the labels from the training datasets at this point (rather than that we're able to synthesize "less biased" data from the generator for better inference.)

### Conclusions

Findings of this study further confirmed our intuition that the quality and volume aggregation of data sources might be the most crucial parts in industry (e.g. health care) that are highly relied on real-world data sources. Across all simulated data biasness scenarios, we found that centralizing all datasets would almost always led to less data attributed biasness compared to individual largest datasets. Though the author failed to prove that involving a GAN framework as a more powerful in reducing overall data + model biasness, we explored and learned a lot of GAN concepts during this process. One immediate next step could have been to do more experiments in tuning the SGAN model to improve the generator performance. Beyond this, exploring GAN framework trained on federated data system (e.g. datasets sit on different vendors' server) remains an interesting area that author would love to explore next. Last but not least, the original proposed framework was hoping to scope a dynamic reinforcement learning framework that incorporates the achievements here. As RWD datasets are commonly refreshed periodically, we expect including a multi-arm bandit like reinforcement learning component could help learning the data drifts more in time and allocate the data weights based on target outcome of interest, where the rewards being levels of biasness reduction from a bench-marking population statistics. 

