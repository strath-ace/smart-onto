# This Source Code Form is subject to the terms of the Mozilla Public ---------------------
# License, v. 2.0. If a copy of the MPL was not distributed with this ---------------------
# file, You can obtain one at http://mozilla.org/MPL/2.0/. */ -----------------------------
# ---------------- Copyright (C) 2019 University of Strathclyde and Author ----------------
# -------------------------------- Author: Audrey Berquand --------------------------------
# ------------------------- e-mail: audrey.berquand@strath.ac.uk --------------------------
# ------------------------- Based on the work of Iain McDonald ----------------------------
# --------------------- e-mail: iain.mcdonald.2015@uni.strath.ac.uk -----------------------

import time
from sklearn.metrics import accuracy_score
from gensim import models
from gensim.corpora import Dictionary
from operator import itemgetter
from TopicModeling.NLPpipeline import *

fileDir = os.path.dirname(os.path.abspath(__file__))  #
parentDir = os.path.dirname(fileDir)  # Directory of the Module directory

# -----------------------------------------------------------
#                           METHODS
# -----------------------------------------------------------

def categorisation(semi, model_name, category, update):
    '''
    Apply pre-trained LDA model to a set of space mission design requirements

    Inputs:
    - semi: if True, the model to be used for the categorisation is semi-supervised
    - model_name: name of saved model to load, usually under the format: 'model_topicNumber',
    all saved models can be found under LDAmodels
    - category: category of requirements to use for the categorisation test, one of the following options, found in
    Corpora/requirementsCorpus: 'AOCS', 'com', 'environment', 'GS', 'Launch', 'MA', 'OBDH', 'payload', 'Power', 'prop',
    'thermal'
    - update: If yes the unsupervised LDA model retained will be updated with the Update corpus found in
    Corpora/updateCorpus, for the chosen category.

    Outputs: the Accuracy Score and Mean Reciprocal Ranking of the categorisation

    CAREFUL 1 : The LDA model generation being a stochastic process, in the case of an updated model, the User will need
    to manually label the topic dictionaries, and save them as .txt, under TopicModeling/inputs4Categorisation
    See labels txt file for LDA models in this folder as examples.

    CAREFUL 2: Same applies for the semi-supervised and unsupervised models. Manual labels files are provided for the
    unsupervised and semi-supervised models used in the paper. But new trained models requires new labels, each time.

    CAREFUL 3: some modifications in the requirement pre-processing may have changed the semi-supervised model result
    w.r.t the original paper presented at the IAC 2019.'''

    start = time.time()

    # Unsupervised LDA model case
    if not semi:
        # Load LDA model and corresponding dictionary ------------------------------------------------------------------
        ldaModel = parentDir + '/TopicModeling/LDAmodels/unsupervised/' + str(model_name)
        lda = models.ldamodel.LdaModel.load(ldaModel)
        print('Model Topics Number:', lda.num_topics)

        dic = parentDir + '/TopicModeling/LDAmodels/unsupervised/dic_' + str(model_name) + '.dict'
        modelDic = Dictionary.load(dic)

        # Recreating the topics dictionaries ---------------------------------------------------------------------------
        ldaTopics = lda.show_topics(formatted=False, num_topics=lda.num_topics, num_words=15)
        print('Loaded LDA Topics Dictionaries, top 15 words:', *ldaTopics, sep='\n')

        # Get manual labels --------------------------------------------------------------------------------------------
        labels = []
        with open(parentDir + '/TopicModeling/inputs4Categorisation/manualLabels_' + model_name + '.txt', 'r',
                  encoding="utf-8") as labelsFile:
            labelLine = labelsFile.read().split('\n')
            for line in labelLine:
                if line:
                    labels.append(line.split(', '))

        labels = [[int(label[0]), label[1]] for label in labels]
        labels = list(itertools.chain.from_iterable(labels))
        print('\n Loaded Model Labels:', labels)

        if update:
            # Updated Unsupervised LDA model case
            # Update LDA model with wikipedia pages focused on one topic -----------------------------------------------
            print('\n Generating a specific LDA model for category', category, ':')
            # Only currently available for GS (Ground Segment), Launch, MA (Mission Analysis), OBDH, payload categories
            filepath = parentDir + '/TopicModeling/Corpora/updateCorpus/' + category + '_update/'

            # Pre-processing of .json docs into tokens
            reqdoc = corpusProcessing(filepath)

            # Use lda model dictionary to transform into document-term matrix understood by the model
            addcorpus = [modelDic.doc2bow(text) for text in reqdoc]

            # Update model
            lda.update(addcorpus, passes=600, offset=1500)

            # Print new dictionary of topics
            ldaTopics = lda.show_topics(formatted=False, num_topics=lda.num_topics, num_words=15)
            print('\n LDA Topics after update', *ldaTopics, sep='\n')

            # Get manual labels ----------------------------------------------------------------------------------------
            labels = []
            with open(parentDir + '/TopicModeling/inputs4Categorisation/manualLabels_' + model_name + '_'+ category +'.txt', 'r',
                      encoding="utf-8") as labelsFile:
                labelLine = labelsFile.read().split('\n')
                for line in labelLine:
                    if line:
                        labels.append(line.split(', '))

            labels = [[int(label[0]), label[1]] for label in labels]
            labels = list(itertools.chain.from_iterable(labels))
            print('\n Labels:', labels)

    else:
        # Semi-unsupervised LDA model case

        # Load LDA model and corresponding dictionary
        ldaModel = parentDir+'/TopicModeling/LDAmodels/semisupervised/guided'+str(model_name)
        lda = models.ldamodel.LdaModel.load(ldaModel)
        print('topics number:', lda.num_topics)

        dic = parentDir +'/TopicModeling/LDAmodels/semisupervised/dic_guided'+str(model_name)+'.dict'
        modelDic = Dictionary.load(dic)

        # Recreating the topics dictionaries
        ldaTopics = lda.show_topics(formatted=False, num_topics=lda.num_topics, num_words=20)
        print('LDA Topics ', *ldaTopics, sep='\n')

        # Get manual labels --------------------------------------------------------------------------------------------
        labels = []
        with open(parentDir + '/TopicModeling/inputs4Categorisation/manualLabels_'+model_name+'_semisupervised.txt', 'r',
                  encoding="utf-8") as labelsFile:
            labelLine = labelsFile.read().split('\n')
            for line in labelLine:
                if line:
                    labels.append(line.split(', '))

        labels = [[int(label[0]), label[1]] for label in labels]
        labels = list(itertools.chain.from_iterable(labels))
        print('\n Labels:', labels)

    # Get test requirements List ---------------------------------------------------------------------------------------
    requirementsList = []
    with open(parentDir + '/TopicModeling/Corpora/requirementsCorpus/req_'+ category+'.txt', 'r', encoding="utf-8") as filteredList:
        requirements = filteredList.read().split('\n')
        for req in requirements:
            if req:
                requirementsList.append(req.split(" | "))

    # Categorisation ---------------------------------------------------------------------------------------------------
    gt = []
    allResults = []
    all_req=[]

    for item in requirementsList:
        req = item[0]
        gt.append(item[1])

        # pre-process requirement
        req = NLPPipe(req)
        all_req.append(req)

        # Use the same dictionary as pre-trained model to  convert a list of words into bag of word format
        unseen_doc = modelDic.doc2bow(req)

        # get topic probability distribution for the unseen document
        vector = lda[unseen_doc]
        sorted_vector = sorted(vector, key=itemgetter(1), reverse=True)

        # Treshold - keep top 2 topics associated, with probabilities
        results = list(map(list, sorted_vector[0:2]))

        # associate top results with manually assigned labels
        for item in results:
            item[0] = labels[labels.index(item[0]+1) + 1]

        allResults.append(results)

    #print('\n All requirements:\n', *all_req, sep='\n')
    print('\n All Results for category', category, ' :')
    print(len(requirementsList), ' requirements were analysed.')
    print(*allResults, sep='\n')

    # Categorisation Evaluation -------------------------------------------------------------------------------------------
    # we have per requirement i, the ground truth gt[i] and the LDA model topic distribution results[i]

    # Accuracy calculation
    firstChoice = [item[0][0] for item in allResults]
    firstChoiceAccuracy = accuracy_score(gt, firstChoice)
    print('First Choice Accuracy : ', firstChoiceAccuracy)

    # Mean Reciprocal Ranking
    bigScore = 0
    for item in allResults:
        i = allResults.index(item)
        score = 0
        if item[0][0] == gt[i]:
            score = 1
        elif len(item)>1:
            if item[1][0] == gt[i]:
                score = 0.5
        bigScore = bigScore +score
    meanReciprocalrank = bigScore / len(requirementsList)
    print('Mean Reciprocal Rank : ', meanReciprocalrank, '\n ---------')

    print('Computation Time:', round((time.time() - start) / 60, 2), 'minutes')

    return

# -----------------------------------------------------------
#                           MAIN
# -----------------------------------------------------------

model_name='model_22'
# As a reminder: categorisation(semi, model_name, category, update)
categorisation(True, model_name, 'thermal', False)
