# This Source Code Form is subject to the terms of the Mozilla Public ---------------------
# License, v. 2.0. If a copy of the MPL was not distributed with this ---------------------
# file, You can obtain one at http://mozilla.org/MPL/2.0/. */ -----------------------------
# ---------------- Copyright (C) 2021 University of Strathclyde and Author ----------------
# -------------------------------- Author: Audrey Berquand --------------------------------
# ------------------------- e-mail: audrey.berquand@strath.ac.uk --------------------------

'''
LDA_semisupervised.py :  Train a semi-supervised LDA model with the Gensim Python library.
Radim Rehurek and Petr Sojka. “Software Framework for Topic Modelling with Large Corpora”.
English. In: Proceedings of the LREC 2010 Workshop on New Challenges for NLP Frameworks.
http://is.muni.cz/publication/884893/en. Valletta, Malta: ELRA, May 2010, pp. 45–50.]

The difference between unsupervised (implemented in LDA.py) and semisupervised LDA consists in the eta matrix,
a matrix of seed words per topics, provided as input to the model training.

(The User needs to provide the number of topics, path to corpus and model names in main())
'''

import math
import re, os, time, json
import numpy as np
import pyLDAvis
import pyLDAvis.gensim # don't remove
import itertools

from gensim import corpora, models
from gensim.corpora import MmCorpus, Dictionary
from sklearn.model_selection import train_test_split,cross_val_score
from os import listdir
from os.path import isfile

start = time.time()
fileDir = os.path.dirname(os.path.abspath(__file__))  #
parentDir = os.path.dirname(fileDir)  # Directory of the Module directory

'''---------------------------
---------- FUNCTIONS ---------
---------------------------'''
def count_seeds(model_name):
    '''
    Additional function to count the number of seed words (= lexical priors) found in a pre-trained LDA model
    :param model_name: model whose topics word distributions will be screened for seed words
    :return: print percentage of seed words found
    '''

    topics_seed = ['thermal', 'thermal_control', 'thermal_control_system', 'heat_pipe', 'heat', 'temperature', 'radiator',
         'insulation','cooling', 'thermal', 'louver', 'heating', 'degree', 'thermodynamics', 'multi_layer_insulation', 'coating',
         'overheating', 'mirror', 'heater', 'reflector', 'reflective','propulsion', 'propulsion_system', 'tether', 'spacecraft_propulsion',
         'propellant_mass', 'delta_v', 'thruster','engine', 'propellant', 'ion', 'plasma', 'sail', 'electric', 'electric_propulsion', 'nuclear', 'thrust',
         'fuel', 'isp', 'total_impulse', 'impulse', 'exhaust','power', 'battery', 'cell', 'solar_cell', 'photovoltaic', 'solar_power', 'voltage', 'watt', 'current',
         'charge', 'discharge', 'power_supply', 'battery_powered', 'primary', 'secondary', 'lithium', 'circuit',
         'energy', 'cycle', 'depth_of_discharge', 'panel', 'efficiency', 'capacity','satellite_communication', 'communication', 'band', 'bandwidth', 'packet', 'x_band', 'transmitter',
         'receiver', 'ka_band', 'c_band', 'frequency', 'antenna', 'relay', 's_band', 'l_band', 'telemetry',
         'tracking', 'telecommand', 'reception', 'command', 'network', 'loss', 'signal', 'range', 'wavelength',
         'modulation','attitude', 'attitude_control', 'guidance', 'navigation', 'reaction_wheel', 'momentum', 'angular', 'body',
         'freedom', 'gyroscope', 'wheel', 'motion', 'torque', 'star_tracker', 'spin_stabilised',
         'stabilization', 'sensor', 'gravity_gradient', 'magnetotorquers', 'torquer', 'inertial', 'coordinate', 'frame',
         'axis','data_handling', 'data_rate', 'clock', 'memory', 'storage', 'dram', 'sram', 'gbit', 'data', 'bitrate', 'cpu',
         'ram', 'tag', 'encoder', 'decoder', 'downlink', 'uplink', 'computer', 'bit', 'measurement', 'execution',
         'instruction', 'operation', 'processor','environment', 'radiation', 'gamma_ray', 'gamma_radiation', 'particle', 'shield', 'dose', 'ray',
         'shielding', 'electron', 'geomagnetic', 'van_allen', 'star', 'solar_wind', 'wind', 'belt',
         'single_event_upset', 'protection', 'cosmic', 'single_event', 'space_debris', 'debris', 'charging', 'background', 'magnetosphere']

    # load pre-trained model
    lda_model = parentDir + '/LDAmodels/books_unsupervised/'+ str(model_name)
    lda = models.ldamodel.LdaModel.load(lda_model)
    print('topics number:', lda.num_topics)

    # Recreating the topics word distributions
    ldaTopics = lda.show_topics(formatted=False, num_topics=lda.num_topics, num_words=50)
    print(*ldaTopics, sep='\n')

    summaryPerTopic=[]
    i=1
    for eachTopic in ldaTopics:

        #get words from distributions
        dic=[x for (x,y) in eachTopic[1]]

        count=0
        heritatedSeed=[]
        for item in dic:
            if item in topics_seed:
                count=count+1
                heritatedSeed.append(item)

        summaryPerTopic.append([i,(count/len(dic)*100), heritatedSeed])
        i=i+1

    # assess percentage of seed words found in word distributions
    print('\n',*summaryPerTopic, sep='\n')

    return

def main_semiLDA(inputPath, model_name, topic_number):
    '''
    Train, save and evaluate a semi-supervised LDA model with space systems lexical priors.

    :param inputPath: path to preprocessed corpus
    :param model_name: name of semi-supervised model to be trained
    :param topic_number: topic number of LDA model
    '''

    # Pre-process Corpus
    filepath = parentDir + inputPath

    # Load pre-processed corpus - list of tokenized document
    documents = [f for f in listdir(filepath) if isfile(filepath + f)]
    doc_preprocessed = []

    print(len(documents), 'documents found ')
    for d in documents:
        with open(filepath + d, 'r') as infile:
            input = json.load(infile)
        doc_preprocessed.append(list(itertools.chain(*input)))

    allTokens = list(itertools.chain(*doc_preprocessed))
    print('Number of tokens:', len(allTokens))
    print('Full Dictionary Size', len(set(allTokens)))

    # Separate between training and testing corpus, 80% training, 20% testing
    corpus_train, corpus_test = train_test_split(doc_preprocessed, test_size=0.2)

    # ------------------------------------------------------------------------------------------------------------
    #                                   MODEL DICTIONARY AND TERM-DOC  MATRIX
    # ------------------------------------------------------------------------------------------------------------
    # Enter Number of Topics to be found by model, initial value = 22
    num_topics = topic_number

    # Create dictionary
    dictionary = corpora.Dictionary(corpus_train)
    #dictionary.filter_extremes(no_below=1)

    dicItems = []
    for k, v in dictionary.token2id.items():
        dicItems.append(k)
    print('LDA Model Inputs:\n Training Corpus Dictionary Size:', len(dictionary))

    # Create Term Document Frequency input
    corpus = [dictionary.doc2bow(tokens) for tokens in corpus_train]

    # ----------------------------------------------------------------------------------------------------------------------
    #                                       SEMI-SUPERVISED LDA
    # ----------------------------------------------------------------------------------------------------------------------
    print("\nSemi-Supervised LDA Results:")

    # --------------------------------------- DEFINE ETA -------------------------------------------
    ''' In the Gensim Python library used to train the model, eta, a matrix representing for each topic, 
    the probability of each word to belong to it, can be provided to the model to impose the asymmetric
    priors over the word distribution. The seed words probabilities of each topic is set to 0,95 
    while the remaining words probabilities are set to 0. '''

    space_eta = np.ones((num_topics, len(dictionary))) * 0

    # !!! USER INPUT !!!
    boost_value = 0.95

    topics_seed = [
                    ['thermal', 'thermal_control', 'thermal_control_system', 'heat_pipe', 'heat', 'temperature', 'radiator', 'insulation',
                    'cooling', 'thermal', 'louver', 'heating', 'degree', 'thermodynamics', 'multi_layer_insulation', 'coating',
                    'overheating', 'mirror', 'heater', 'reflector', 'reflective'],

                   ['propulsion', 'propulsion_system', 'tether', 'spacecraft_propulsion', 'propellant_mass', 'delta_v', 'thruster',
                    'engine', 'propellant', 'ion', 'plasma', 'sail', 'electric', 'electric_propulsion', 'nuclear', 'thrust',
                    'fuel', 'isp', 'total_impulse', 'impulse', 'exhaust'],

                   ['power', 'battery', 'cell', 'solar_cell', 'photovoltaic', 'solar_power', 'voltage', 'watt', 'current',
                    'charge', 'discharge', 'power_supply', 'battery_powered', 'primary', 'secondary', 'lithium', 'circuit',
                   'energy', 'cycle', 'depth_of_discharge', 'panel', 'efficiency', 'capacity'],

                   ['satellite_communication', 'communication', 'band', 'bandwidth', 'packet', 'x_band', 'transmitter',
                    'receiver', 'ka_band', 'c_band', 'frequency','antenna','relay', 's_band', 'l_band', 'telemetry',
                    'tracking', 'telecommand', 'reception', 'command', 'network', 'loss', 'signal', 'range', 'wavelength', 'modulation'],

                   ['attitude', 'attitude_control','guidance', 'navigation', 'reaction_wheel', 'momentum', 'angular', 'body',
                   'freedom', 'gyroscope', 'wheel', 'motion', 'torque', 'star_tracker', 'spin_stabilised',
                   'stabilization', 'sensor', 'gravity_gradient', 'magnetotorquers', 'torquer', 'inertial', 'coordinate', 'frame', 'axis'],

                   ['data_handling', 'data_rate', 'clock', 'memory', 'storage', 'dram', 'sram', 'gbit', 'data', 'bitrate', 'cpu',
                    'ram', 'tag', 'encoder', 'decoder', 'downlink', 'uplink', 'computer', 'bit', 'measurement', 'execution','instruction', 'operation', 'processor'],

                   ['environment', 'radiation', 'gamma_ray', 'gamma_radiation', 'particle', 'shield', 'dose', 'ray',
                   'shielding', 'electron', 'geomagnetic', 'van_allen', 'star', 'solar_wind', 'wind', 'belt', 'single_event_upset',
                    'protection', 'cosmic','single_event', 'space_debris', 'debris', 'charging', 'background', 'magnetosphere']]

    notfound=[]
    seedlist=[]

    # check how many topic seeds are not in corpus dictionary
    test = list(itertools.chain(*topics_seed))
    print('Number of seed words', len(test))
    count = 0
    for item in test:
        if item not in dicItems:
            count = count + 1
    print('Not found in dictionary:', count, '\n')

    # For each line (or topic number id) of the topic_seed matrix
    count=-1
    for line in topics_seed:
        count = count + 1
        for word in line:
            # Verify that the word is in the model dictionary
            if word in dicItems:
                # boost the probability of the word to belong to the topic
                space_eta[count][dicItems.index(word)] = boost_value
            else:
                notfound.append(word)

    print('ETA update done, boosted words:', count)
    i=0
    while i <= num_topics-1:
        boostedList = []
        topicDist = space_eta[i]
        boosted = np.argwhere(topicDist == boost_value)
        boosted = list(itertools.chain.from_iterable(boosted))

        if boosted:
            for x in boosted:
                boostedList.append(dicItems[x])
            print('--> topic ', i, ', boosted with', len(boosted), ' words:', boostedList)

        i=i+1

    print(len(notfound), 'Seed words not found:', notfound)

    # ------------------------------------- TRAIN SEMI-SUPERVISED LDA MODEL ------------------------------------------------
    # Train model with eta matrix
    modelG = models.ldamodel.LdaModel(corpus, id2word=dictionary, num_topics=num_topics, passes=500, eta=space_eta)

    # ------------------------------------------ DISPLAY TOPICS + PYLDAVIS ------------------------------------------------
    print("Semi-Supervised LDA Topics:")
    for i in modelG.show_topics(formatted=False, num_topics=modelG.num_topics, num_words=30):
        print(i)

    # Save model
    #vis = pyLDAvis.gensim.prepare(modelG, corpus, dictionary, sort_topics=False)
    #pyLDAvis.save_html(vis, parentDir + 'LDAmodels/new_semisupervised/LDA_Visualization_' + str(num_topics) + '.html')
    MmCorpus.serialize(parentDir + '/LDAmodels/books_semisupervised/corpus_' + str(model_name) + '.mm',
                       corpus)
    dictionary.save(parentDir + '/LDAmodels/books_semisupervised/dic_semisupervised_' + str(model_name) + '.dict')
    modelG.save(parentDir + '/LDAmodels/books_semisupervised/semisupervised_' + str(model_name))
    print('LDA model generated and saved')

    # ------------------------------------------ EVALUATE WITH TESTING CORPUS ----------------------------------------------
    corpusTest = [dictionary.doc2bow(tokens) for tokens in corpus_test]
    perplexity = math.exp(modelG.log_perplexity(corpusTest))
    print('\n Final Evaluation, perplexity:', perplexity)
    print('\n Computation Time:', round((time.time() - start) / 60, 2), 'minutes')

    return

def main():
    # Input training corpus path (JSON format)
    inputPath = '/processedCorpora/Books/'

    # Select topic number, from our journal paper we recommend:
    # topic number of 22 for wikipedia corpus
    # topic number of 30 for ESA CDF reports corpus
    # topic number of 24 for books corpus
    topic_number = 24

    # Select the names of the models
    m = ['books_24_2', 'books_24_3', 'books_24_4', 'books_24_5', 'books_24_6', 'books_24_7', 'books_24_8', 'books_24_9',
         'books_24_10']

    # Run training of semi-supervised models
    for item in m:
        model_name = item
        main_semiLDA(inputPath, model_name, topic_number)

    return

'''---------------------------
------------ MAIN ------------
---------------------------'''
if __name__ == "__main__":
    main()




