# SpaceTransformers: Language Modeling for Space Systems

## Table of contents
* [Abstract](#Abstract)
* [Repository Content](#RepositoryContent)
* [Dataset](#Dataset)
* [Citation](#Citation)
* [License](#License)
* [Contact](#Contact)

## Abstract
The transformers architecture and transfer learning have radically modified the Natural Language Processing (NLP) landscape, enabling new applications in fields where open source labelled datasets are scarce. Space systems engineering is a field with limited access to large labelled corpora and a need for enhanced knowledge reuse of accumulated design data. Transformers models such as the Bidirectional Encoder Representations from Transformers (BERT) and the Robustly Optimised BERT Pretraining Approach (RoBERTa) are however trained on general corpora. To answer the need for domain-specific contextualised word embedding in the space field, we propose SpaceTransformers, a novel family of three models, SpaceBERT, SpaceRoBERTa and SpaceSciBERT, respectively further pre-trained from BERT, RoBERTa and SciBERT on our domain-specific corpus. We collect and label a new dataset of space systems concepts based on space standards. We fine-tune and compare our domain-specific models to their general counterparts on a domain-specific Concept Recognition (CR) task. Our study rightly demonstrates that the models further pre-trained on a space corpus outperform their respective baseline models in the Concept Recognition task, with SpaceRoBERTa achieving significant higher ranking overall.

## Repository Content
The code stored in this repository was used to further pre-train the models and generate the Concept Recognition results for ['SpaceTransformers: Language Modeling for Space Systems'](https://ieeexplore.ieee.org/document/9548078) published in the IEEE Access journal link.
 
This code was developed in Python 3.7. The configuration and pre-trained weights of the BERT-Base, RoBERTa-Base and SciBERT models are
accessed through the [HuggingFace library](https://huggingface.co/) Transformers library. The further pre-trained models can be similarly loaded with this library.

The [notebook](https://github.com/strath-ace/smart-nlp/blob/master/SpaceTransformers/CR/Fine_tune_SpaceTransformer.ipynb) for fine-tuning the pre-trained models on the Concept Recognition tasks is available for download, or can be directly excecuted via [Google Colab](https://colab.research.google.com/drive/1EGh9bdxq6RqIzbvKuptAWvmIBG2EQJzJ?usp=sharing).

The *FPT* folder contains the code to further pre-train the domain-specific models. The models were further pre-trained with one NVIDIA V100 GPU hosted on the [ARCHIE-WeSt High Performance Computer](https://www.archie-west.ac.uk) based at the
University of Strathclyde. The *FT* folder contains the code to fine-tune the further pre-train models on a Concept Recognition Task.

**The models are to be accessed through the HuggingFace library at [huggingface.co/icelab](https://huggingface.co/icelab).**

## Dataset
The [further pre-training](https://doi.org/10.15129/8e1c3353-ccbe-4835-b4f9-bffd6b5e058b) corpora is available through the Strathclyde Knowledge Base. The [fine-tuning](https://github.com/strath-ace/smart-nlp/blob/master/SpaceTransformers/CR/CR_ECSS_dataset.json) dataset is available for download and consists of 874 unique manual annotated ECSS requirements.

## Citation
If you use our models and/or code, do cite our research:
@ARTICLE{9548078,
  author={Berquand, Audrey and Darm, Paul and Riccardi, Annalisa},
  journal={IEEE Access}, 
  title={SpaceTransformers: Language Modeling for Space Systems}, 
  year={2021},
  volume={9},
  number={},
  pages={133111-133122},
  doi={10.1109/ACCESS.2021.3115659}}

## License
This code is licensed under Version 2.0 of the Mozilla Public License.

## Contact
Open an 'issue' or contact [Audrey Berquand](mailto:audrey.berquand@protonmail.com).

