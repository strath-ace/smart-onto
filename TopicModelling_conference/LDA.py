# This Source Code Form is subject to the terms of the Mozilla Public ---------------------
# License, v. 2.0. If a copy of the MPL was not distributed with this ---------------------
# file, You can obtain one at http://mozilla.org/MPL/2.0/. */ -----------------------------
# ---------------- Copyright (C) 2019 University of Strathclyde and Author ----------------
# -------------------------------- Author: Audrey Berquand --------------------------------
# ------------------------- e-mail: audrey.berquand@strath.ac.uk --------------------------

'''
LDA.py : Topic Modeling with unsupervised LDA, based on Gensim Python Library:
Radim Rˇ ehu°rˇek and Petr Sojka. “Software Framework for Topic Modelling with Large Corpora”.
English. In: Proceedings of the LREC 2010 Workshop on New Challenges for NLP Frameworks.
http://is.muni.cz/publication/884893/en. Valletta, Malta: ELRA, May 2010, pp. 45–50.

Inputs:
- corpus of wikipedia pages for model training and evaluation
- 5 cross-fold optimisation of topics numbers
- LDA evaluation with perplexity and coherence

Outputs:
- if the optimisation process is run (opti = True), the outputs are perplexity and coherence figures saved in
outputs\outUnsupervised which should help the User select the right topic number
- if the User enters him/herself a topic number (opti = False), the outputs is a trained LDA model, saved under
LDAmodels\new_unsupervised including a pyldavis.
'''

import math
import re, os, time
import matplotlib.pyplot as plt
import pyLDAvis
import pyLDAvis.gensim # don't remove
from gensim import corpora, models
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.model_selection import KFold
from gensim.models import CoherenceModel
from TopicModeling.NLPpipeline import *

fileDir = os.path.dirname(os.path.abspath(__file__))  #
parentDir = os.path.dirname(fileDir)  # Directory of the Module directory

# ------------------------------------------------------------------------------------------------------------
#                                             METHODS
# ------------------------------------------------------------------------------------------------------------
def ldaModelGeneration(train_corpus, test_corpus, topic_number, value_save_model):
    '''
    LDA model training, visualisation and evaluation

    Inputs:
    - train_corpus: 80% of the wikipedia corpus to be used for training of LDA model
    - test_corpus: 20% of the wikipedia corpus to be used for final evaluation of trained model(perplexity+coherence)
    - topic_number: number of latent topics to be found by model
    - value_save_model: if True, will save model, model dictionary and pyldavis visualisation

    Output:
    - perplexity value: evaluation of perplexity of trained model over unseen documents (test_corpus) --> common LDA
    evaluation metrics
    - coherence value: coherence score of trained model over unseen documents (test_corpus) --> less reliable
    '''

    fileDir = os.path.dirname(os.path.abspath(__file__))  #
    parentDir = os.path.dirname(fileDir)  # Directory of the Module directory

    # ------------------------------------------------------------------------------------------------------------
    # LDA MODEL - GENERATION/VISUALISATION WITH TRAINING CORPUS
    # ------------------------------------------------------------------------------------------------------------
    # Choose LDA model name for this iteration
    model_name = 'model_' + str(topic_number)

    # Create model dictionary
    dictionary = corpora.Dictionary(train_corpus)
    dictionary.filter_extremes(no_below=0.2)
    print('\n LDA Model Inputs:\n Dictionary Size:', dictionary)

    # Create Document-Term matrix
    corpus = [dictionary.doc2bow(tokens) for tokens in train_corpus]

    # Generate LDA model
    ldamodel = models.ldamodel.LdaModel(corpus, id2word=dictionary, num_topics=topic_number, passes=300)

    if value_save_model == True:

        # Visualise topics: words and their weights
        print("LDA Topics:")
        for i in ldamodel.show_topics(formatted=False, num_topics=ldamodel.num_topics, num_words=20):
            print(i)

        # Save model
        dictionary.save(parentDir + '/TopicModeling/LDAmodels/new_unsupervised/dic_' + str(model_name) + '.dict')
        ldamodel.save(parentDir+'/TopicModeling/LDAModels/new_unsupervised/'+str(model_name))
        print('LDA model generated and saved')

        # Save pyldavis (usually takes a few minutes to generate)
        vis = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary, sort_topics = False)
        pyLDAvis.save_html(vis, parentDir + '/TopicModeling/LDAmodels/new_unsupervised/LDA_Visualization_' + str(topic_number)+'.html')

    # ------------------------------------------------------------------------------------------------------------
    #                                LDA MODEL - EVALUATION
    # ------------------------------------------------------------------------------------------------------------

    # Use same dictionary as the model was trained with to transform unseen data into Document-Term matrix
    corpusTest = [dictionary.doc2bow(tokens) for tokens in test_corpus]

    # Model Perplexity - must be minimised
    perplexity = ldamodel.log_perplexity(corpusTest)
    perplexityExp = math.exp(perplexity)

    # Topic Coherence
    cm = CoherenceModel(model=ldamodel, corpus=corpusTest, coherence='u_mass')
    coherence = cm.get_coherence()  # get coherence value

    return perplexityExp, coherence

# ------------------------------------------------------------------------------------------------------------
#                                               MAIN
# ------------------------------------------------------------------------------------------------------------
def main_LDA():
    '''
    main_LDA: either optimise topic number for given corpus or select a topic number and train/evaluate/save an
    unsupervised LDA model

    The User will need to indicate whether he/she wants to run the 5 cv optimisation, switch parameter opti to True or
    False. If False, also indicates the number of topics the model should find within corpus, by replacing the value of
    parameter best_topic_number
    '''

    start = time.time()
    fileDir = os.path.dirname(os.path.abspath(__file__))  #
    parentDir = os.path.dirname(fileDir)  # Directory of the Module directory

    # --------------------------------------- LOAD, PREPROCESS CORPUS -------------------------------------------
    # Input Corpus Directory - parsed wikipedia pages in json format, based on Corpora/wikiURLList
    # The code to identify and scrap all hyperlinks within a Wikipedia Page with the Python Selenium is available on demand
    filepath = parentDir + '/TopicModeling/Corpora/wikiCorpus/'

    # Pre-process Corpus
    doc_preprocessed = corpusProcessing(filepath)

    # ----------------------------------- SEPARATE TRAINING AND TEST SET -----------------------------------------
    # Divide Corpus between training and test set (80/20%) with train_test_split method
    # from sklearn, splits arrays into random train and test subsets
    corpus_train, corpus_test = train_test_split(doc_preprocessed, test_size=0.2)

    # ----------------------------------- LDA OPTIMISATION ON TRAINING SET ----------------------------------------
    # opti = True, will launch the optimisation process (based on 5 fold cross validation) to find the optimal number
    # of topics for the corpus. Might take some time to run.
    # opti = False, the user provides a topic number and by-pass the optimisation process.

    # !!! USER INPUT !!!
    opti = False

    if opti == True:
        # Optimise an lda model over the training corpus with 5 fold cross validation
        # We are looking for the topics number which will minimise perplexity
        # The number of latent topics to be found by the LDA model is a key parameter of the Topic Modeling

        # Topics Number range to be tested
        topic_number_range = list(range(4, 72, 2))

        # For each topics number, we run a 5-fold cross validation
        # This means that for each topics number, we get 5 perplexity and coherence score. These numbers are averaged to get the
        # evaluation of one topics number. The perplexity variance is also saved.
        LDAevaluation = []

        for topic_number in topic_number_range:
            progress = (topic_number_range.index(topic_number) + 1) / len(topic_number_range) * 100
            print('Topic Number:', topic_number, ' Progress [%]: ', round(progress, 1))
            # generate n-fold distributions
            cv = KFold(n_splits=5, shuffle=True)

            # run n lda model for n fold, each time get perplexity + coherence scores.
            fold_perplexity = []
            fold_perplexityExp = []
            fold_coherence = []

            for train_index, test_index in cv.split(corpus_train):
                # generate training and test corpus
                train_corpus = [corpus_train[index] for index in train_index]
                test_corpus = [corpus_train[index] for index in test_index]
                # run lda model for each fold
                perplexity, coherence = ldaModelGeneration(train_corpus, test_corpus, topic_number, False)
                # save output
                fold_perplexity.append(perplexity)
                fold_coherence.append(coherence)

            # get results of the fold for n topic number
            mean_perplexity = np.mean(fold_perplexity)
            var_perplexity = np.var(fold_perplexity)
            mean_coherence = np.mean(fold_coherence)

            LDAevaluation.append([topic_number, mean_perplexity, var_perplexity, mean_coherence])

        # Save results
        f = open(parentDir + '/TopicModeling/outputs/outUnsupervised/LDAevaluation_new.txt', mode="w", encoding="utf-8")
        for i in LDAevaluation:
            f.write(str(i))
            f.write('\n')
        f.close()

        # print results
        topicsnum = [x for (x, y, z, v) in LDAevaluation]
        mean_perplexity = [y for (x, y, z, v) in LDAevaluation]
        coherence = [v for (x, y, z, v) in LDAevaluation]

        # perplexity results
        fig1 = plt.figure()
        plt.plot(topicsnum, mean_perplexity)
        plt.ylabel('Average Perplexity')
        #plt.grid()
        plt.xlabel('Topics Number')
        fig1.savefig(parentDir+'/TopicModeling/outputs/outUnsupervised/mean_perplexity.png')
        plt.close


         # topic coherence results
        fig3 = plt.figure()
        plt.plot(topicsnum, coherence)
        plt.ylabel('Coherence')
        #plt.grid()
        plt.xlabel('Topics Number')
        fig3.savefig(parentDir+'/TopicModeling/outputs/outUnsupervised/coherence.png')
        plt.close

        # ---------------------------------------------- SELECTION OF BEST MODEL ---------------------------------------
        # The optimal number of topic should get the exponentielle of perplexity closest to 0. To avoid over-fitting, it
        # is however recommended to balance this choice with a pyldavis visualisation which will allow to visualise the
        # topics distribution.
        # Usually the User will select the best topic number suggested by this optimisation process, and re-run the model
        # generation to include the Final Testing (opti = False).

        # get min perplexity
        min_p = min(mean_perplexity)

        # find index of min perplexity, and get topic number
        best_topic_number = LDAevaluation[mean_perplexity.index(min_p)][0]
        print('Suggested best topic number range: ', best_topic_number)

    else:
        # The User already provides a number of latent topics to be found by the LDA model

        # !!! USER INPUT !!!
        # Initially set up to 22
        best_topic_number = 22
        print('\n Number of Topics proposed by User: ', best_topic_number)
        # -------------------------------- FINAL TESTING OF BEST MODEL WITH TEST CORPUS --------------------------------
        # run lda model generation with complete training set, save model, test with testing set and display evaluation
        print('\n FINAL MODEL: ')
        perplexity, coherence = ldaModelGeneration(corpus_train, corpus_test, best_topic_number, True)
        print('Final Evaluation, perplexity:', perplexity, ', Topic Coherence:', coherence)

    print('Computation Time:', round((time.time() - start) / 60, 2), 'minutes')

    return()

main_LDA()





