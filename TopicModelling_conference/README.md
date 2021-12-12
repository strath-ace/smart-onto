# Topic Modeling for Space Mission Requirement Categorisation

## Table of contents
* [Introduction - The Design Engineering Assistant project](#Introduction)
* [Latent Dirichlet Allocation (LDA) for Space Mission Design ](#LDA)
* [Getting Started](#start)
* [Dataset](#dataset)
* [Citation](#cite)
* [License](#lic)
* [Contact](#con)

## Introduction - The Design Engineering Assistant project
The Design Engineering Assistant (DEA) project is run at the Intelligent Computational Intelligence Lab,
University of Strathclyde, Glasgow UK, by Audrey Berquand, PhD student and under the supervision of Annalisa Riccardi. 
The project also involves ESA, RHEA Systems, AIRBUS and satsearch in the frame of an ESA Networking/Partnership Initiative (NPI).

The goal is to develop an Expert System (ES) to support decision making at the early stages of space mission design (e.g., during feasibility studies). 
Implementing quick and efficient Information Retrieval (IR) has become essential to reduce the time spent by engineers searching for information through previous missions reports, online databases, etc.
Topic Modeling (TM) is used to identify, learn and extract topics from a corpus of documents, and can therefore support several IR tasks such as categorisation or Q/A.

**This study presented the first application of Latent Dirichlet Allocation (LDA), a method of Topic Modeling, on a space mission design
corpus.**

More about the project: http://icelab.uk/projects/research-projects/dea/ \
Publications: https://www.researchgate.net/project/Design-Engineering-Assistant-DEA-for-Space-Mission-Design

## Latent Dirichlet Allocation (LDA) for Space Mission Design 
The code stored in this repository was used to generate results presented at the International Astronautical Congress 2019, in Washington DC (USA), in the session on 'Knowledge Management for space activities in the digital era'. The paper,
'The automatic categorisation of space mission requirements for the Design Engineering Assistant' (IAC-19,D5,2,7,x51013), is available at [researchgate](https://www.researchgate.net/publication/337256904_The_automatic_categorisation_of_space_mission_requirements_for_the_Design_Engineering_Assistant).


**The code presented here allows to train and evaluate unsupervised, supervised and updated LDA models, a common TM method, from a wikipedia-based space mission design corpus. The models can be evaluated via the categorisation of space mission requirements, extracted from freely-available European Space Agency mission documents.**
 
The repository includes
* The functions to train an unsupervised, semi-supervised and updated LDA models,
* The functions to perform 5 cross-fold validation to optimise a model topic number,
* A first 'Space Mission Design' wikipedia corpus, the requirements corpus and the Natural Language Processing pipeline used to process them,
* Examples of unsupervised and semi-supervised topics dictionaries, labeled and validated by Human annotators,
* The 'Space' lexical priors used to train the semi-supervised LDA model, validated by Human annotators.

## Getting Started
This code was run with Python 3.7. 

Start by running *set_up.py*.

*LDA.py* is used to train unsupervised LDA models, while *LDA_semisupervised* is used to train semi-supervised models. 

## Dataset
The dataset [Dataset of space systems corpora](https://doi.org/10.15129/8e1c3353-ccbe-4835-b4f9-bffd6b5e058b) is available from the University of Strathclyde KnowledgeBase.


## Citation
If you use this code, we kindly request that you cite our research:

If you use this code, we kindly request that you cite our research, 
you may use the following BibTex entry or equivalent:

@inproceedings{berquandTM, \
      title = {{The automatic categorisation of space mission requirements for the Design Engineering Assistant}},\
      author = {Audrey Berquand and Iain McDonald and Annalisa Riccardi and Yashar Moshfeghi},\
      booktitle = {{Proceedings of the 70th International Astronautical Congress (IAC)}},\
      year = 2019,\
      month = October,\
      publisher = {IAF},\
      address = {Washington D.C., USA},\
      language={English}}


## License
This code is licensed under Version 2.0 of the Mozilla Public License.

## Contact
Open an 'issue' or contact [Audrey Berquand](mailto:audrey.berquand@strath.ac.uk).

