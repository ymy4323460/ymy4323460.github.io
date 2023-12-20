---
layout: post
title: Discussion about CausalVAE (from content to implementation)
date: 2023-12-20 09:16:00
description: find questions and answers about CausalVAE.
tags: formatting code
categories: sample-posts
featured: true
---
  
  CausalVAE was officially published at CVPR in 2021 and received much attention. After two years, I have gathered various questions about CausalVAE. I will summarize some common questions and answers for everyone here. These questions mainly cover aspects related to the original paper and publicly available code.

  In addition, I will also provide some problems, discussions and challenges of CausalVAE during the time I did this work.

## Questions about the CausalVAE
Q1. What is causalVAE? How does it differ from previous methods?
A1. CausalVAE is a fully supervised causal disentanglement representation learning method. By providing the algorithm with images and feature labels, we can stably discover causal representations and support interventions on these representations to generate counterfactual images. Before CausalVAE, there were many explorations into causal representation learning, most of which focused on detecting independent causal representations or representations influenced by a unified root cause. In comparison, CausalVAE is the first method to directly explore how to learn representations of factors that have causal relationships with each other in observed data.

Q2. What is $u$?
A2. u in paper means the additional supervision signals, in our method, we concat labels of each factor like 'light position', and 'shadow length' as vector u.

Q3. Why did the paper use different causal assumptions in eq.1 and eq.2? 
A3. Because eq.1 and eq.2 serve different purposes. eq.1 aims to obtain representation z through exogenous variables, and formulating this process with non-linearity is challenging. eq.2, on the other hand, is designed for interventions, where non-linearity can model more complex generative processes. In practical modelling, using non-linearities in generation often yields better results compared to linear approach.

