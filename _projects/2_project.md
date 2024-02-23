---
layout: distill
title: Meta-Learning for Question Answering on SQuAD 2.0
description: This post was summarized from my final project @Stanford CS224N.
img: assets/img/mamlBert_illustration.png
importance: 2
category: learning
bibliography: maml.bib
authors:
  - name: Chi-Hsuan Chang
date: 2022-03-14
---

In a Question Answering (QA) system, the system learn to answer a question by properly understanding an associated paragraph. However, developing a QA system that performs robustly well across all domains can be extra challenging as we do not always have abundant amount of data across domains. Therefore, one area of focus in this field has been learning to train a model to learn new task with limited data available (e.g. Few-Shot Learning, FSL).

Meta-learning in supervised learning, in particular, has been known to perform well in FSL, with the concept being teaching the models learn to set up initial parameters well that enable the model to learn a new task after seeing a few samples of the associated data.<d-cite key="finn2017, abs-1904-05046"></d-cite> In this study, we were given a large amount of in-domain (IND) samples with only limited samples of out-of-domain (OOD) set. We were provided with a fine-tuned (FT) DistilBERT model <d-cite key="sanh2020distilbert"></d-cite> that knew to perform well on the IND set. To improve the robustness of the FT baseline model performance on OOD set, we trained:

- **MAML models from scratch**
- **MAML models after baseline model was pre-trained and fine-tuned**

### Background

**[SQuAD 2.0 dataset.](https://rajpurkar.github.io/SQuAD-explorer/explore/v2.0/dev/)**
Three in‐domain (SQuAD, NewsQA, Natural Questions) and three out‐of‐domain (DuoRC, RACE, RelationExtraction) datasets. The in‐domain (IND) and out‐of‐domain (OOD) datasets contain 50K and 127 question‐passage‐answer samples each.

**Model‐Agnostic Meta‐Learning (MAML).**
MAML was originally proposed by Finn et al 2017 <d-cite key="finn2017"></d-cite> to train the models their own initial parameters so that the parameters allow the algorithm to perform well on a new task (”learn‐to‐learn”) after one or a few gradient steps of updates with few‐shot data availability.

### Methods

**Fine-tuned Baseline.** A fine‐tuned (FT) pre‐trained transformer model ‐ DistilBERT.<d-cite key="sanh2020distilbert"></d-cite> The baseline QA model was trained on the overall IND training set, and was validated on the IND validation set.

**MAML DistilBERT.** We adapted MAML<d-cite key="finn2017"></d-cite> as a framework to train our robust QA system that performs well across different domains.
- We defined the baseline DistilBERT <d-cite key="sanh2020distilbert"></d-cite>  as our base learner ($f_{\theta}$)
    
- We implemented a task method rather than to pre-define a K-shot task pool ($p(\mathcal{T})$). As K sample support ($\mathcal{D}_i$) and query ($\mathcal{D}_i$' ) sets can come from IND and OOD training datasets in different experiments
  
- We used the same loss function ($\mathcal{L}$, $\textbf{loss} = - \log p_{start}(i) - \log p_{end}(j)$) as the baseline
  
<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/mamlBert_illustration.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Figure 1. Model architecture of MAML DistilBERT. Training support and query sets can come from In‐domain or OOD datasets and are a factor we experimented on.
</div>

**FT Baseline + MAML DistilBERT.** In addition to training MAML model from scratch, we leveraged the FT DistillBERT (Baseline) model and trained the MAML models from the FT checkpoint.


### Experiments
If not otherwise specified, batch size for all experiments were 16. To avoid GPU out-of-memory issue, data was loaded in either batch size of 1 or 4 to accumulate the loss. Model is updated at batch size of 16.

**Experiment \#1: MAML DistilBERT without FT Baseline**
- **K-shot:** MAML-20-d vs. MAML-2000-d
- **learning rate:** MAML-20-a vs. MAML-20-b vs. MAML-20-d
- **domain variability in training support:** MAML-20-b vs. MAML-20-c
<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/table1.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Table 1. Experiment 1: Model configuration.
</div>  
 
**Experiment \#2: Training MAML after FT Baseline**
- **K-shot:** M1/2/4 vs. M3, M7 vs. M8, M9 vs. M10
- **IND or OOD for MAML training:** M1 vs. M6 vs. M7 vs. M10, M2 vs. M5 vs. M7 vs. M10
- **training time:** M1 vs. M2 vs. M4, M5 vs. M5

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/table2.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Table 2. Experiment 1: Model configuration.
</div> 

### Analysis
**Key-takeaway \#1: MAML DistilBERT without FT Baseline couldn't achieve the same level of model performance as the FT Baseline.**

- This can be because of the large IND data available during baseline model pre-training/fine-tuning. 
- Larger learning rate helped in faster adaptation with the MAML model given the same sample size as it  allowed more aggressive exploration in the gradient at the beginning.
- Larger domain variability in support/query reached similar F1 performance but lower EM performance. This was intuitive as the MAML was learning to learn and exposed to a lot of topics as few-shot learning though benefit understanding synergies across domains, the model also became more "general" and "robust".
    

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/model1.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Figure 2. Experiment #1 model descending sorted by EM (OOD eval)
</div> 

**Key-takeaway \#2: Training MAML after FT Baseline outperformed FT Baseline occasionally. More experimentation configurations in learning rate and domain variability could be explored.**

- **M2**, a 10-task 20-shot MAML training on OOD samples post pre-training outperformed the FT Baseline in OOD validation set by **1.22%** in F1 and **3.04%** in EM. Its performance in IND validation set dropped by **4.57%** in F1 and **6.49%** in EM. This showed the scarification of model performance on the IND datasets in gaining additional robustness on an OOD dataset.
- **M8**, a 10-task 200-shot MAML training on IND samples post pre-training outperformed the FT Baseline in OOD validation set by **0.44%** in F1 and **0%** in EM, and in IND validation set by **0.78%** in F1 and **1.10%** in EM. This showed that continuously training with the same domain datasets with MAML contributed less improvements than training with few OOD samples.


<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/model2.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Figure 3. Experiment #2 model descending sorted by EM (OOD eval)
</div> 

### Conclusions

MAML was a good‐to‐explore to achieve cross‐domain model robustness. MAML might not be the best framework in context of a large amount IND set and small amount OOD set. Training MAML post baseline model pre‐training and fine‐tuning performed occasionally better than the FT baseline model likely due to additional OOD tasks used to learn by the MAML model. 

Full copy of the paper could be found [here](https://web.stanford.edu/class/cs224n/reports/default_116613241.pdf).

Github of this project could be found [here](https://github.com/achchg/cs224_robustqa).