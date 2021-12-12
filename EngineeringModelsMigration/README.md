# Migration of Engineering Models to Knowlege Graph 
The code stored in this repository was used to generate results presented at the 9th International Systems & Concurrent 
Engineering for Space Applications Conference (SECESA 2020), a digital event held in October 2020. 
The paper 'From engineering models to knowledge graph: delivering new insights into models', part of the session 5 on 
Concurrent Engineering - Challenges and Opportunities, is available on [researchgate](https://www.researchgate.net/publication/344451299_From_Engineering_Models_to_Knowledge_Graph_Delivering_New_Insights_Into_Models).

This study makes the following contributions: \
a.	Provides a pipeline in Python to automatically migrate any ECSS-E-TM-10-25A-based EM to a Grakn KG.\
b.	Provides rules to infer a mass budget for each design option of an iteration.\
c.	Trains a doc2vec model on a data set of ECSS requirements to assess past and current missions’ similarities. (training data set to be added soon provided the original creator of the data set gives us her permission.)

## Table of contents
* [Getting Started](#start)
* [Dataset] (#Dataset)
* [Citation](#cite)
* [License](#lic)
* [Contact](#con)

## Getting Started
This code runs with Python 3.7. 
* Install Grakn 1.8.0 (https://grakn.ai/),
* Install Grakn Workbase 1.3.0 (https://grakn.ai/)
* Start Grakn server (*grakn server start*)
* Load Schema *EMSchema.gql* into Grakn keyspace (*grakn console -k keyspaceName -f filepath/EMSchema.gql*)
* To infer a mass budget relationship, load the rules *EMRules.gql* into same Grakn keyspace (*grakn console -k keyspaceName -f filepath/EMRules.gql*)
* Place your Engineering Model .json files into *dataset* folder, our Engineering Models were generated with RHEA's CDP4-Community Edition
* Run *set_up.py*.
* Run *migrate_em_json.py* to populate the Grakn keyspace with your engineering model

To generate mass budget results as in the first case study of the paper, run *queryMassBudget.py*.
To assess requirements sets' similarity, run *assessMissionSimilarity.py*.

When done:
* Stop Grakn server (*grakn server stop*)

## Dataset
The dataset [Dataset of space systems corpora](https://doi.org/10.15129/8e1c3353-ccbe-4835-b4f9-bffd6b5e058b) is available from the University of Strathclyde KnowledgeBase.

 
## Citation
If you use this code, we kindly request that you cite our research, 
you may use the following BibTex entry or equivalent:

@inproceedings{berquandSECESA2020, \
      title = {{From engineering models to knowledge graph: delivering new insights into models}},\
      author = {Audrey Berquand and Annalisa Riccardi},\
      booktitle = {{Proceedings of the 9th International Systems & Concurrent 
        Engineering for Space Applications Conference (SECESA 2020)},\
      year = 2020,\
      month = October,\
      publisher = {ESA},\
      address = {Digital event},\
      doi={},\
      language={English}}

## License
This code is licensed under Version 2.0 of the Mozilla Public License.

## Contact
Open an 'issue' or contact [Audrey Berquand](mailto:audrey.berquand@strath.ac.uk).