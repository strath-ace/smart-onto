# SpaceLDA: Topic distributions aggregation from a heterogeneous corpus for space systems

## Table of contents
* [Summary for neophytes](#summary)
* [Latent Dirichlet Allocation (LDA) for Space Mission Design ](#LDA)
* [Getting Started](#start)
* [Dataset](#dataset)
* [Citation](#cite)
* [License](#lic)
* [Contact](#con)

## Summary for neophytes
The development of a satellite is a complex process generating large amounts of documentation. Tracking design requirements within these documents is a challenge for engineers. Each requirement is linked to a spacecraft subsystem. In this paper we propose an innovative approach to manage requirements based on Latent Dirichlet Allocation (LDA).

LDA identifies and extracts topics contained in a document. We trained a model on domain-specific documents to learn topics such as “thermal” and “power systems”. Our spaceLDA model specialises in the discovery of topics related to space systems. We use our model to match new design requirements with spacecraft subsystems.

## Latent Dirichlet Allocation (LDA) for Space Mission Design 
The code stored in this repository was used to generate the results for 'SpaceLDA: Topic distributions aggregation from a heterogeneous corpus for space systems' published in the Engineering Application of Artificial Intelligence journal [(link)](https://www.sciencedirect.com/science/article/abs/pii/S0952197621001202).

**The code presented here allows to train and evaluate unsupervised and semi-supervised LDA models on a space mission design corpus. The models are combined either with the Jensen-Shannon Divergence method or with a weighted sum. The models are evaluated through a categorisation task.**
 
## Getting Started
This code was run with Python 3.7. 

Start by running *set_up.py*.
 
*LDA.py* is used to train unsupervised LDA models, while *LDA_semisupervised* is used to train semi-supervised models. \
*JSDivergence.py* aggregates model based on the JS Divergence method, while *hyperopt_optimisation.py* yields the optimised weights to balance per-document topics distributions.\
*categorisation.py* runs the categorisation application.

## Dataset
The dataset [Dataset of space systems corpora](https://doi.org/10.15129/8e1c3353-ccbe-4835-b4f9-bffd6b5e058b) is available from the University of Strathclyde KnowledgeBase.


## Citation
If you use this code, do cite our research:

@article{BERQUAND2021104273, \
title = {SpaceLDA: Topic distributions aggregation from a heterogeneous corpus for space systems}, \
journal = {Engineering Applications of Artificial Intelligence}, \
volume = {102}, \
pages = {104273}, \ 
year = {2021}, \
issn = {0952-1976}, \ 
doi = {https://doi.org/10.1016/j.engappai.2021.104273}, \
url = {https://www.sciencedirect.com/science/article/pii/S0952197621001202},\
author = {Audrey Berquand and Yashar Moshfeghi and Annalisa Riccardi},\
keywords = {Topic Modelling, LDA, Spacecraft design, Requirements, Aggregation},\
abstract = {The design of highly complex systems such as spacecraft entails large amounts of documentation. Tracking relevant information, including hundreds of requirements, throughout several design stages is a challenge. In this study, we propose a novel strategy based on Topic Modelling to facilitate the management of spacecraft design requirements. We introduce spaceLDA, a novel domain-specific semi-supervised Latent Dirichlet Allocation (LDA) model enriched with lexical priors and an optimised Weighted Sum (WS). We collect and curate the first large collection of unstructured data related to space systems, combining several sources: Wikipedia pages, books, and feasibility reports provided by the European Space Agency (ESA). We train the spaceLDA model on three subsets of our heterogeneous training corpus. To combine the resulting per-document topic distributions, we enrich our model with an aggregation method based on an optimised WS. We evaluate our model through a case study, a categorisation of spacecraft design requirements. We finally compare our model’s performance with an unsupervised LDA model and with a literature aggregation method. The results demonstrate that the spaceLDA model successfully identifies the topics of requirements and that our proposed approach surpasses the use of a classic LDA model and the state of the art aggregation method.}\
}

## License
This code is licensed under Version 2.0 of the Mozilla Public License.

## Contact
Open an 'issue' or contact [Audrey Berquand](mailto:audrey.berquand@strath.ac.uk).

