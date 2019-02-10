---
layout: post
title: "PU Learning"
subtitle: "Positive/unknown class machine learning approaches"
description: "A challenge that keeps presenting itself at work is one of not having a labelled negative class in the context of needing to train a binary classifier. Typically, the issue is paired with horribly imbalanced data sets and pressed for time, I have often taken the simplistic route of sub-sampling the unknown set and treating them as unknowns. Obviously this isn’t ideal as the unknown set is contaminated and as a result the classifiers dont train that well."
image: /post/20190111/pu_learning/20180728_111831_rocherpan_full.jpg
thumbnail: /post/20190111/pu_learning/20180728_111831_rocherpan_full.jpg
Author: "Philip Massie"
date: 2019-01-11
draft: false
tags: [ "PU learning", "machine learning", "data science", "class imbalance"]
url: "/post/20190111/pu_learning/"
output: 
  html_document: 
    keep_md: yes
    toc: no
---
# Introduction
A challenge that keeps presenting itself at work is one of not having a labelled negative class in the context of needing to train a binary classifier. Typically, the issue is paired with horribly imbalanced data sets and pressed for time, I have often taken the simplistic route of sub-sampling the unknown set and treating them as unknowns. Obviously this isn't ideal as the unknown set is contaminated and as a result the classifiers dont train that well. Nevertheless, out in the wild, with real-life deadlines, the approach was time efficient, and the results were often surprisingly useful.

Recently, I was lucky to have a few days to read around the topic a little. I found some interesting approaches and thought it would be worth taking a few notes, and they turned into this post.

_Disclaimer: This post is not intended to be an exhaustive review of the PU learning approaches. Essentially, it's just me making some notes and storing them somewhere I can find them again, and where they may be helpful to someone else._

There are a few different PU approaches around. All the approaches involve isolating a set of so-called 'Reliable Negatives' (RNs) from the unknown data set. As I read, the most widely cited, initial approaches are attributed to Liu et al. 2002 and 2003 wherein a set of RNs are iteratively grown from within the unknown class.

Another approach was described by Fusilier et al. 2015. In their paper the authors describe an approach which iteratively reduces the set of RNs from within the unknown class, effectively tightening the net around those cases which are the most dissimilar to the positive class. This approach appealed to me as it implicitly deals with class imbalance.

The third approach I came across (Mordelet & Vert 2013) which also implicitly accounts for class imbalance involves bagging, or randomly sampling from the unknown class and treating the sample as negatives. Where this deviates from my naive approach mentioned above is that the process is repeated many times and a series of models are trained. The models characterise the positive class against unknown data sets with varying degrees of contamination. The resulting model scores are ensembled and the result should better isolate the reliable negatives from the unknown class. 

Below, I will go into a little detail about each of the three approaches.

# Methods

## 'Original' approach (Liu et al. 2002 and 2003)
Given a training set containing only positives (P) and unknown (U) classes follow the following steps:

1. Treating all U as negatives (N) train a classifier P vs. U
2. Using the classifier, score the unknown class and isolate the set of 'reliable' negatives (RN).
3. Train a new classifier on P vs. RN, use it to score the remaining U, isolate additional RN and enlarge RN.
4. Repeat step 3, iteratively enlarging the set of RN until the stopping condition is met. 

The stopping condition is met when no new negative cases are classified.

Where `Q` is defined as the set of unknowns classified as negatives and `i` is the iterator, the stopping condition is defined as:

>```|Qi| > 0```

## Modified approach (Fusilier et al. 2014)
Given a training set containing only positives (P) and unknown (U) classes follow the following steps:

1. Treating all U as negatives (N) train a classifier P vs. U
2. Using the classifier, score the unknown class and isolate the set of 'reliable' negatives (RN).
3. Train a new classifier on P vs. RN. Score RN and exclude predicted positives from RN
4. Repeat step 3, iteratively refining the RN set, until the stopping condition is met. 

Where `Q` is defined as the set of unknowns classified as negatives and `i` is the iterator, the stopping condition is defined as:

>```|Qi| <= |Q(i-1)| & |P| < |Qi|```


The stopping condition ensures that Q reduces in size (avoiding sudden large reductions in RN size) while the RN set never gets smaller than the P set. More explicitly:

>while the size of the set of unknowns classified as negatives in _this iteration_ is smaller than or equal to the size of the set of unknowns classified as negatives in _the previous iteration_ and the size of the set of positive classes  is smaller than the set of refined RNs resulting from _this iteration_

## Bagging approach (Inductive) (Mordelet & Vert 2013)
Given a training set containing only positives (P) and unknowns (U), where K = size of bootstrap samples and T = number of samples, follow the following steps:
1. Draw a bootstrap sample Ut of size K from U
2. Train a classifier P vs Ut
3. Repeat steps 1 and 2 T times
4. Score the test data with an ensemble approach using the bagged models.

The stopping criterion here is determined by the value of T and the authors suggest that there is typically not much additional value to be gained by setting T > 100. Judging from their plots however, where `|P|` and `K` are both large, there is little change above T = 5. I suspect it's worth trying to keep track of this during training if possible or setting up an early stopping type criterion in your function because depending on your time constraints as training 100 models may not be viable.

> ## Things to follow up on:
> 1. Most of the articles use SVM, but they also tend to be NLP problems. Does the classifier family matter much?
> 2. How do the original papers identify the cut-off for determining 'reliability'
> 3. Modified approach: WRT the stopping criterion, why could Q get larger with the iterations?
> 4. Bagging approach: Consider how best to penalise false negatives.
>       - Cutoff selection?

# Conclusion
These three methods provide sensible approaches to the problem of PU learning but only the modified and bagging approaches provide inherent ways to deal with imbalanced data. My plan is to try and implement these 2 approaches and compare their results. While I cant share the data publicly, I will try and share the code and general results on the blog and in GitHub. We work primarily in Python/PySpark or Scala/Spark. Some nice links:

- https://github.com/ispras/pu4spark PU learning libraries written in Scala/Spark
- https://astrakhantsev.com/pu-learning/ nice post written by author of pu4spark
- https://roywright.me/2017/11/16/positive-unlabeled-learning/ Nice overview of PU learning approaches

# References
Fusilier DH, Montes-y-Gómez M, Rosso P, Guzmán Cabrera R (2015) Detecting positive and negative deceptive opinions using PU-learning. Inf Process Manag 51:433–443. doi: 10.1016/j.ipm.2014.11.001

Liu B, Dai Y, Li X, et al (2003) Building text classifiers using positive and unlabeled examples. In: Third IEEE International Conference on Data Mining. pp 179–186

Liu B, Lee WS, Yu PS, Li X (2002) Partially Supervised Classification of Text Documents. In: Proc. 19th Intl. Conf. on Machine Learning. pp 387–394

Mordelet F, Vert J-P (2014) A bagging SVM to learn from positive and unlabeled examples. Pattern Recognit Lett 37:201–209. doi: 10.1016/j.patrec.2013.06.010

