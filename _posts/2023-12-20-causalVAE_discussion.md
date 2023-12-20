---
layout: post
title: Discussion about CausalVAE: from content to implementation
date: 2023-12-20 09:16:00
description: find questions and answers about CausalVAE.
tags: discussion
featured: true
---
  
  CausalVAE was officially published at CVPR in 2021 and received much attention. After two years, I have collected various questions about CausalVAE. I will summarize some common questions and answers for everyone here. These questions mainly cover aspects related to the original paper and code/implementation.

  In addition, I will also provide some problems, discussions and challenges of CausalVAE during the time I did this work.

## Questions about the CausalVAE

**Q1. What is causalVAE? How does it differ from previous methods?**

A1. CausalVAE is a fully supervised causal disentanglement representation learning method. By providing the algorithm with images and feature labels, we can stably discover causal representations and support interventions on these representations to generate counterfactual images. Before CausalVAE, there were many explorations into causal representation learning, most of which focused on detecting independent causal representations or representations influenced by a unified root cause. In comparison, CausalVAE is the first method to directly explore how to learn representations of factors that have causal relationships with each other in observed data.

**Q2. What is $u$?**

A2. u in paper means the additional supervision signals, in our method, we concat labels of each factor like 'light position', and 'shadow length' as vector u.

**Q3. Why did the paper use different causal assumptions in eq.1 and eq.2?**

A3. Because eq.1 and eq.2 serve different purposes. eq.1 aims to obtain representation z through exogenous variables, and formulating this process with non-linearity is challenging. eq.2, on the other hand, is designed for interventions, where non-linearity can model more complex generative processes. In practical modelling, using non-linearities in generation often yields better results compared to linear approach.

**Q4. Is the relationship between causal variables in eq.2 linear Gaussian? and does it not comply with the identifiability theory of causal discovery?**

A4. Due to the Gaussian assumption of epsilon used in the causal layer and the linearity of the causal layer, causal relationships cannot be identified at this layer. The effective identification of causal relationships is actually achieved in the mask layer. In the CausalVAE class code, the mask layer is implemented through two steps. The first step involves filtering with the causal graph, and the second step is the implementation of the mix function, which represents a reversible nonlinear function. The code for this can be found in the 'nns' module, and it can be examined in detail.

**Q5. How many steps of propagation need to be considered for the intervention step in the mask layer?**

A5. For the intervention step in the mask layer, one needs to consider the length of the longest causal chain in the entire causal graph. In the paper, the longest causal chain consists of only two steps: smile -> mouth open -> eyes open. This implies that after intervening in "smile," it is necessary to perform two causal propagations to ensure the intervention reaches "eyes open."

**Q6. What does the identifiability theory tell us?**

A6. To understand identifiability, it is essentially important to grasp whether the learned representation can encompass all the information from the ground truth. In our paper, as matrix B is diagonal and invertible, it ensures that the learned representation not only has a linear relationship with the ground truth of that representation but also, due to invertibility, does not lose any information from the ground truth.

## Questions about the Code

We release our CausalVAE code in repo: https://github.com/huawei-noah/trustworthyAI/tree/master/research/CausalVAE. 

I have rectified some typos of CausalVAE code in my repo https://github.com/ymy4323460/trustworthyAI/tree/master/research/CausalVAE.

**Q1. The code cannot generate images with good quality.**

A1. Regarding the issue of suboptimal performance on synthetic data, it is due to incomplete code documentation. The resolution to this issue can be found in the following link: https://github.com/huawei-noah/trustworthyAI/issues/59. You can refer to this solution to improve the performance. 

**Q2. What does scale mean in code?**

A2. For two synthetic datasets (e.g. Pendulum and Flow), scaling is performed to normalize the labels to the range between -1 and 1. The first value in the scale is the mean, and the second value is half of the range size. For example, if there is a one-dimensional value in the pendulum data that ranges from -120 to 120, the scale would be set as [mean=0, scale=120]. After scaling, the value 120 would be normalized to 1.

**Q3. how are the MIC and TIC metrics implemented?**

A3. We computed the correlation between u and z_given_dag (i.e. z_given_dag is the learned representation) by TIC and MIC score. The specific calculation process involved taking the average of the 4-dimensional vectors corresponding to one concept in z_given_dag to obtain a 1-dimensional vector. Then, the correlation between the 1-dimensional vectors of the four concepts and u can be computed.

**Q4. Is the causal graph parameter A implemented using a binary matrix?**

In implementation, the causal graph uses continuous values and is not mapped to binary values.

**Q5. The code on CelebA.**

A5. After I resigned from Huawei. I don't have access to the source code. Recently I asked some related staff in Huawei - Furui Liu, Zhitang Chen and Dong Li. They could not find the related code anymore due to the changes in devices. That made me very disappointed.....

The architecture of the neural network on CelebA might have some differences from mask.py I hope that the following suggestions might be helpful.

1. You can try to use a more complex encoder/decoder like ResNet or modify the dimension and the number of layers in ConvNet.

2. Change the self.scale in CausalVAE class as self.scale = np.array([[0, 1],[0, 1],[0,1],[0,1]])

3. Use smaller lambdav/alpha in CausalVAE class like 1e-5, which can help to lower the variance of representation. Smaller beta like 0.1-0.3 can also help.

4. Decrease the weighting of 'kl' term and mask_l in nelbo = rec + 'kl' + mask_l

5. Init self.A in DagLayer class as an upper triangular matrix, or pre-train causal graph matrix by augmented Lagrange method. Like No-tears and DAG-GNN. A tip: You can try the true graph as the initialization to make sure some architectures of the decoder can work well. But the true graph initialization is not allowed when you want to train a whole model.

## Discussion of CausalVAE

#### The challenges in CausalVAE.

In experimental observations, we noticed that sometimes even if the causal graph is not learned well enough, good intervention results can still be achieved. This observation does not align with our initial expectations for the paradigm of embedding causal learning into other tasks. This phenomenon implies that causal relationships are captured by other parts of the model, such as the decoder/generator. This is disastrous for embedding causal methods into other tasks and models because even if we can observe some results with causal properties, it does not necessarily mean that causal relationships are truly being identified. In the CausalVAE paper, our identifiability theory can only guarantee that there is a one-to-one correspondence between the learned representation z and the true representation. Additionally, the optimization theory for u in the mask layer theoretically helps us learn the true causal graph. However, in multi-objective learning (such as CausalVAE, which includes causal learning and image generation), the power of this identifiability in experiments may be reduced. Therefore, in our experiments, we have to set a pre-train step causal graph learning to some extent, which disrupts the end-to-end architecture.

We believe that there are still several unfinished challenges in CausalVAE:

1. How to ensure that each module only performs its own function, for example, the decoder only generates data, and the causal module only learns causal effects. This challenge will contribute to the end-to-end learning of embedding causal learning into other tasks.

2. How to improve the identifiability theory to ensure that the model's parameters are also unique. This way, by uniquely determining the parameters of the causal model, we can ensure that the remaining parameters are also unique.

3. How to balance the optimization of the objective functions in multi-objective learning that includes causal learning.







