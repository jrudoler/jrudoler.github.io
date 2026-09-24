---
layout: archive
classes: wide
title: Research
toc: true
author_profile: true
permalink: /research/
---
> "Little strokes fell great oaks." \
> \- Benjamin Franklin 
{: .epigraph}

My research has wandered from machine learning applications in neuroscience toward the theory and methodology of deep learning. Across both areas, I'm interested in how learning systems pick up useful patterns, and when those patterns carry over to new people, data, or tasks.

At Penn, I worked with electroencephalography (EEG)—electrical recordings of brain activity—and used machine learning to ask whether brain signals could help predict behavior, like whether someone would remember an item. We're still a ways off from mind-reading, but it's pretty cool stuff. These days I'm interested in why deep learning works so well, where it runs into trouble, and whether a better theoretical understanding can help us build models that are more reliable and interpretable.

<!-- Looking forward, I'm deeply interested in machine learning theory, methodology, and applications. There are really two main uses for data: inference (how/why something happened in the past) and prediction (what will happen in the future). We have good tools for both of these, but they rarely work together. 
Deep learning models, for example, achieve high prediction accuracy but are often criticized for being "black box" models without interpretable parameters. Bayesian approaches explicitly model a data generating process and are therefore highly interpretable, but they require making lots of structural assumptions about probability distributions in real world data that might not be justified - this makes them biased and potentially less robust. Across the board, lots of high-perfoming models have a tendency to overfit training data and consequently fail to make robust predictions out in the wild.  -->

<!-- A long-term interest of mine is developing machine learning methods that are effective tools for both *inference* and *prediction*. These methods need to be both *interpretable* and *robust* - no easy task! I hope to devote my time and attention in graduate school and beyond to studying these challenges in data science, along with applications to neuroscience and other fields.  -->

## Theory of deep learning

Right now I'm especially interested in how training procedures shape the solutions neural networks learn, and what those solutions can tell us about generalization.

### Inductive bias / implicit regularization

Paper: [Estimating Implicit Regularization in Deep Learning](https://arxiv.org/abs/2605.05436)

One possible explanation for why neural networks generalize well is that they have some kind of inductive bias that encourages them to learn generalizing solutions (perhaps e.g. a simplicity bias that prevents overfitting). Lots of theory has been devoted to studying how our training methods (e.g. stochastic gradient descent) implicitly regularize models' effective loss landscape such a way.

The animation shows what happens to a simple two-parameter model as we turn up $$\ell_2$$ regularization. The red surface is the original loss, the blue contours show the regularizer, and the gray surface is the combined objective. As the regularization gets stronger, the minimum moves toward the point preferred by the regularizer.

<figure class="research-figure">
  <img src="/files/regularization-path.gif" alt="Animation showing how regularization shifts the minimum of a two-parameter loss landscape">
  <figcaption>As regularization increases, the minimum of the combined objective shifts.</figcaption>
</figure>

The new solution isn't the same as the original one: regularization nudges it to a different point in parameter space. I'm interested in using shifts like this to reverse-engineer the inductive biases that training introduces.

## EEG analysis and machine learning applications

### Working towards foundation models for neural data
For my master's thesis, I trained deep neural networks to predict behavior (especially memory) from neural data collected across different people. This is a kind of transfer learning called domain adaptation: the hope is that a model can pick up patterns shared across brains and use them to make predictions for someone it hasn't seen before.

The work received a departmental award, though we didn't end up with a paper. Since then, similar approaches have shown promise for tasks with clearer neural signals than memory, like motor activity, sleep stages, and stress. I'm not working on this now, but I still think it's a really promising direction.

### Decoding brain states and improving memory
Paper: [Decoding EEG for optimizing naturalistic memory](https://www.sciencedirect.com/science/article/abs/pii/S0165027024001651), *Journal of Neuroscience Methods*
- In this project we asked whether using machine learning to optimize the timing of item presentations during learning could improve memory performance. Presented as a poster at the *Cognitive Neuroscience Society (CNS)* annual meeting, *Context and Episodic Memory Symposium*, and *MathPsych* in spring/summer 2022.  



### Oscillatory biomarkers of memory
Paper: [Hippocampal theta and episodic memory](https://www.jneurosci.org/content/43/4/613), *Journal of Neuroscience*

- I investigate how a method of distinguishing pink noise in brain recordings from true brain rhythms helps us understand what patterns of brain activity actually relate to successful memory encoding and retrieval. Presented at the *Context and Episodic Memory Symposium* in August 2021 and *Computational and Systems Neuroscience (COSYNE)* in March 2022.
<!-- -->

<figure class="research-figure">
  <img src="/files/exp_animation.gif" alt="Animation illustrating changes in model parameters">
</figure>

### EEG pre-processing methods
Undergraduate research project: [Optimal EEG Referencing Schemes for Brain State Classification](/files/Referencing_Report.pdf). The project compares reference schemes for EEG electrodes, explains how they act as spatial filters, and evaluates their utility for classifying brain state and memory success.

## Sports analytics

In my spare time, I like to dabble in sports analytics. In 2022, my team of Penn grad students was named a finalist for the NFL Big Data Bowl!
You can check out our [Kaggle notebook](https://www.kaggle.com/jrudoler56/optimal-run-path-for-kick-returners) as well as the NFL's [press release](https://operations.nfl.com/updates/football-ops/nfl-announces-finalists-for-fourth-annual-nfl-big-data-bowl/) announcing the finalists and my team's [video presentation](https://www.nfl.com/videos/2022-big-data-bowl-ryan-gross-joseph-rudoler-tai-nguyen-ryan-brill) of our project.

Our project used high-resolution player-tracking data to predict kick-return outcomes, then built a framework for finding *optimal return paths* and evaluating players' decisions.
<figure class="research-figure">
  <img src="/files/bdb.gif" alt="Animation from the Big Data Bowl kick-return analysis">
</figure>
