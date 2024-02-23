---
layout: post
title: Attention
date: 2022-10-24
description: Day 32
tags: review
categories: ml, dl
---
There are a lot of nice materials explaining Attention model fairly well. My favorite have been [the "Attention Interfaces" of this blog post](https://distill.pub/2016/augmented-rnns/#attentional-interfaces) and the [CS224N lecture note](https://web.stanford.edu/class/cs224n/readings/cs224n-2019-notes06-NMT_seq2seq_attention.pdf).

#### Attention
Great [video](https://www.youtube.com/watch?v=SysgYptB198) with the intuition of Attention model explained by Andrew Ng.

Typical sequence-to-sequence models like RNN were used in machine translation, where a input sentence of language A is translated to an output sentence of target language B in an encoder-decoder architecture. One problem with it was that the models go word by word within a sentence during encoding and depending on the final hidden layer before decoding to memorize everything fed into the system for translation; the decoder than taking the hidden layer and pass on another sequence to predict the most likely word that should pop up next given the current translated word.

On the contrary, the decoder network (language B) of the Attention model was trained by **the entire input sequence** (language A) at every decoding step (y_t) with the goal of learning the **attention weight** ($$\alpha_{t,1}$$ to $$\alpha_{t,T}$$) of individual translated word in B on all the input words (x_1, ..., x_T) in A. Below cited the Figure 1 from the [original Attention paper](https://arxiv.org/pdf/1409.0473.pdf).


<div class="row">
    <div class="col-md-3 offset-md-3">
        {% include figure.liquid path="assets/img/attention.png" title="example image" %}
    </div>
</div>


Great video materials:
- [Attention model](https://www.youtube.com/watch?v=FMXUkEbjf9k) by Andrew Ng.
  