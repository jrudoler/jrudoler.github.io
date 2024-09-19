---
layout: archive
title: Research
toc: true
author_profile: true
permalink: /research/
---
> "Little strokes fell great oaks." \
> \- Benjamin Franklin 

To date, most of my research at Penn has focused on analyzing electroencephalographic (EEG) recordings - measurements of the changing electrical potential in people's brains caused by neurons firing.
A topic of particular interest for my previous lab is how we can use machine learning to actively predict a person's behavior (like whether or not they will remember a studied item) based on on their brain activity. We're still a ways off from mind-reading, but it's cool stuff. 

Looking forward, I'm deeply interested in machine learning theory, methodology, and applications. There are really two main uses for data: inference (how/why something happened in the past) and prediction (what will happen in the future). We have good tools for both of these, but they rarely work together. 
Deep learning models, for example, achieve high prediction accuracy but are often criticized for being "black box" models without interpretable parameters. Bayesian approaches explicitly model a data generating process and are therefore highly interpretable, but they require making lots of structural assumptions about probability distributions in real world data that might not be justified - this makes them biased and potentially less robust. Across the board, lots of high-perfoming models have a tendency to overfit training data and consequently fail to make robust predictions out in the wild. 

A long-term interest of mine is developing machine learning methods that are effective tools for both *inference* and *prediction*. These methods need to be both *interpretable* and *robust* - no easy task! I hope to devote my time and attention in graduate school and beyond to studying these challenges in data science, along with applications to neuroscience and other fields. 

## EEG analysis and machine learning applications

### Working towards foundation models for neural data
My master's thesis was about training deep neural nets to predict behavior (in particular, memory) from neural data aggregated across different people. 
This is an instance of a type of transfer learning called domain adaptation -- 
essentially you want the model to learn some shared properties of neural activity across brains so that it can predict neural activity in a brain that it's never seen before. This is similar in spirit to how large language models are trained on large corpora of text so they can learn properties of natural language that generalize to new language-related tasks. 

While my work was recognized by an award from my department, we didn't really have any publishable results. Since then, similar work has come out showing that this kind of approach is successful for many tasks with stronger neural correlates than memory (e.g. motor tasks, sleep stages, stress/emotion, etc.). I'm not actively working on this but think it's a super cool and promising research direction.  

### Decoding brain states and improving memory
Paper: [Decoding EEG for optimizing naturalistic memory](https://www.sciencedirect.com/science/article/abs/pii/S0165027024001651), *Journal of Neuroscience Methods*
- In this project we asked whether using machine learning to optimize the timing of item presentations during learning could improve memory performance. Presented as a poster at the *Cognitive Neuroscience Society (CNS)* annual meeting, *Context and Episodic Memory Symposium*, and *MathPsych* in spring/summer 2022.  



### Oscillatory biomarkers of memory
Paper: [Hippocampal theta and episodic memory](https://www.jneurosci.org/content/43/4/613), *Journal of Neuroscience*

- I investigate how a method of distinguishing pink noise in brain recordings from true brain rhythms helps us understand what patterns of brain activity actually relate to successful memory encoding and retrieval. Presented at the *Context and Episodic Memory Symposium* in August 2021 and *Computational and Systems Neuroscience (COSYNE)* in March 2022.
<!-- -->

<img src="/files/exp_animation.gif" alt="Changing Parameters" width="600" align="left"/><br clear="left">

### EEG pre-processing methods
Undergraduate Research Project: [Optimal EEG Referencing Schemes for Brain State Classification](./files/Referencing_Report.pdf)
- Analyzing changing electrical potential requires choosing a reference point for the measurement. When we have some set of electrodes recording brain activity in distinct spatial locations in the brain, should they all be referenced the same way? To a common electrode? To their nearest neighboring electrode? To a weighted sum of other electrodes? I discuss a number of approaches, explain how they act as variable "spatial filters", and compare their utility for classifying brain state and memory success.

## Sports Analytics

In my free time I like to dabble in sports analytics a bit. I (along with a few other Penn grad students) was named a finalist for the 2022 NFL Big Data Bowl! 
You can check out our [Kaggle notebook](https://www.kaggle.com/jrudoler56/optimal-run-path-for-kick-returners) as well as the NFL's [press release](https://operations.nfl.com/updates/football-ops/nfl-announces-finalists-for-fourth-annual-nfl-big-data-bowl/) announcing the finalists and my team's [video presentation](https://www.nfl.com/videos/2022-big-data-bowl-ryan-gross-joseph-rudoler-tai-nguyen-ryan-brill) of our project.

Our submission showed how high resolution player-tracking data allows us to train a model that predicts the outcome of a kick return, and we develop a framework for using this to compute *optimal return paths* and evaluate player decision-making.
<img src="/files/bdb.gif" alt="Big Data Bowl" width="600" align="left"/><br clear="left">
