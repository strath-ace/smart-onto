# smart-nlp
Strathclyde Mechanical and Aerospace Research Toolboxes for Natural Language Processing

## Available Repositories
* [Space Lexicon Generator](#SpaceLexiconGenerator)
* [Topic Modelling](#TopicModelling)
* [Engineering Models Migration to Knowledge Graph](#EngineeringModelsMigrationtoKnowledgeGraph)

## Space Lexicon Generator
The code stored in this repository was used to generate the results of the paper "Space mission design ontology: extraction of
domain-specific entities and concepts similarity analysis", presented at the 2020 AIAA SciTech Forum in Orlando, USA in the Invited Session on Cognitive Assistants. The paper is available on [researchgate](https://www.researchgate.net/publication/338400758_Space_mission_design_ontology_extraction_of_domain-specific_entities_and_concepts_similarity_analysis).

**The code allows to automatically extract a domain-specific lexicon from a domain-specific corpus, in this case, the corpus is related to space mission design. The lexicon items are embedded with word2vec and similar concepts are identified with cosine similarity.**

## Topic Modelling

### Conference version
The code stored in this repository was used to generate results presented at the International Astronautical Congress 2019, in Washington DC (USA), in the session on 'Knowledge Management for space activities in the digital era'. The original paper 'The automatic categorisation of space mission requirements for the Design Engineering Assistant' (IAC-19,D5,2,7,x51013) is available on [researchgate](https://www.researchgate.net/publication/337256904_The_automatic_categorisation_of_space_mission_requirements_for_the_Design_Engineering_Assistant).

**The code allows to train and evaluate unsupervised, supervised and updated LDA models, a common Topic Modeling method, from a wikipedia-based space mission design corpus. The models can be evaluated via the categorisation of space mission requirements, extracted from freely-available European Space Agency mission documents.**
 
### Journal version
The code stored in this repository was used to generate the results for 'SpaceLDA: Topic distributions aggregation from a heterogeneous corpus for space systems' published in the Engineering Application of Artificial Intelligence journal [(link)](https://www.sciencedirect.com/science/article/abs/pii/S0952197621001202).

**The code allows to train and evaluate unsupervised and semi-supervised LDA models on a space mission design corpus (wikipedia webpages, books and ESA feasibility study reports). The models are combined either with a Jensen-Shannon Divergence method or with a weighted sum. The models are evaluated through a categorisation task.**

## Engineering Models Migration to Knowledge Graph
The code stored in this repository was used to generate results presented at the 9th International Systems & Concurrent 
Engineering for Space Applications Conference (SECESA 2020), a digital event held in October 2020. 
The paper 'From engineering models to knowledge graph: delivering new insights into models' is available on [researchgate](https://www.researchgate.net/publication/344451299_From_Engineering_Models_to_Knowledge_Graph_Delivering_New_Insights_Into_Models).

**The code allows to automatically migrate Engineering Models based on the ECSS-E-TM-10-25A TM (in our case exported from the RHEA CDP4-CE platform), to a **[Grakn](https://grakn.ai/)** Knowledge Graph (KG). Code is also provided to infer a new type of relationship isIncludedInMassBudget within the graph, and automatically generate a dry mass budget for each design option. Finally, a pipeline to train a doc2vec model with the Gensim Python library, embed requirements sets found in the populated KG and assess their similarity with cosine similarity is provided.**

Comment: The authors warmly thank Sabrina Mirtcheva and Serge Valera from ESA for kindly providing the ECSS requirements data set used to train the doc2vec model. This data set is based on the EARM_ECSS_export(DOORS-v0.7_May2019).xlsx document that can be found [here](https://ecss.nl/standards/downloads/earm/).