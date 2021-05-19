# This Source Code Form is subject to the terms of the Mozilla Public ---------------------
# License, v. 2.0. If a copy of the MPL was not distributed with this ---------------------
# file, You can obtain one at http://mozilla.org/MPL/2.0/. */ -----------------------------
# ---------------- Copyright (C) 2021 University of Strathclyde and Author ----------------
# -------------------------------- Author: Audrey Berquand --------------------------------
# ------------------------- e-mail: audrey.berquand@strath.ac.uk --------------------------

import time, os, itertools, re
import sys

from sklearn.metrics import accuracy_score
from gensim import models
from gensim.corpora import Dictionary
from operator import itemgetter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from TopicModelling_journal.TM_methods import *
import matplotlib.pyplot as plt
import numpy as np
from pylab import plot, show, savefig, xlim, figure, ylim, legend, boxplot, setp, axes

sys.path.append("..")
fileDir = os.path.dirname(os.path.abspath(__file__))  #
parentDir = os.path.dirname(fileDir)  # Directory of the Module directory

'''---------------------------
---------- FUNCTIONS ---------
---------------------------'''

def NLPPipe(req, acronyms, exp, ecssMultiwords, ecssMultiwordsWithAcronyms , stopset):
    '''
    NLP pipeline tailored to space systems with ECSS acronyms and glossary of terms
    Used to process the requirements
    :param req: requirement to process
    :param acronyms: list of acronyms based on ECSS list
    :param exp: acronyms expansion
    :param ecssMultiwords: list of multi-words found in ecss glossary of terms
    :param ecssMultiwordsWithAcronyms: same but multi-words with acronyms
    :param stopset: list of stop words
    :return: processed requirement
    '''

    listReplacementMW = []
    listReplacementAcc = []

    # NLP Pipeline
    tokens = word_tokenize(req)

    # Trim - remove empty space start/end word
    tokens = [word.strip() for word in tokens]

    # Remove tokens that are only numbers
    tokens = [x for x in tokens if re.findall('[a-zA-Z]', x)]

    # Remove tokens with special characters: /,%
    tokens = [x for x in tokens if not re.findall('/%', x)]

    # Remove tokens mixing characters and numbers
    tokens = [x for x in tokens if not re.findall('[0-9]', x)]

    # Remove urls
    tokens = [re.sub(r'www.*[\r\n]*', '', token) for token in tokens]

    # Remove non English/number characters:
    tokens = [re.sub('[^A-Za-z0-9_\-/]+', '', token) for token in tokens]

    # Remove Empty tokens
    tokens = [x for x in tokens if x]

    # Replace - by _ to homogenize multi words
    tokens = [token.replace('-', '_') for token in tokens]

    # Replace multi words which contain acronyms
    tokens, listReplacementMW = replaceMultiwords(tokens, listReplacementMW, ecssMultiwordsWithAcronyms)

    # Replace acronyms
    tokens, listReplacementAcc = acronymExpansion(tokens, listReplacementAcc, acronyms, exp)

    # Normalise Text
    tokens = [w.lower() for w in tokens]

    # Replace multi words
    tokens, listReplacementMW = replaceMultiwords(tokens, listReplacementMW, ecssMultiwords)

    # Remove stopwords + punctuation
    tokens = [w for w in tokens if w not in [stopset, 'shall']]

    # Lemmatization - currently based on wordnet
    wnl = WordNetLemmatizer()
    tokens = [wnl.lemmatize(word) for word in tokens]

    # Remove stopwords + punctuation
    tokens = [w for w in tokens if w not in stopset]

    # Remove very short tokens
    processedReq= [x for x in tokens if len(x) > 2]

    return processedReq

def getModel(model_name, aggregated, semi, fmodel):
    '''
    Load model to be used in categorisation
    :param model_name:  model name
    :param aggregated: if it has been generated with JS divergence = True, else False
    :param semi: if it has been trained in semi-supervised fashion = True, else False
    :param fmodel: path of model folder
    :return: lda model, its dictionary to encode unseen data, its labels manually defined
    '''

    if aggregated:
        ldaModel = parentDir + '/LDAModels/JSAggregatedModels/' + fmodel + '/' + str(model_name)
        lda = models.ldamodel.LdaModel.load(ldaModel)
        lda.expElogbeta = lda.eta
        dic = parentDir + '/LDAModels/JSAggregatedModels/' + fmodel + '/dic_' + str(model_name) + '.dict'
        modelDic = Dictionary.load(dic)

        labels = []
        with open(
                parentDir + '/LDAModels/JSAggregatedModels/' + fmodel + '/manualLabels_model_' + model_name + '.txt','r',
                encoding="utf-8") as labelsFile:
            labelLine = labelsFile.read().split('\n')
            for line in labelLine:
                if line:
                    labels.append(line.split(', '))

        labels = [[int(label[0]), label[1]] for label in labels]
        labels = list(itertools.chain.from_iterable(labels))

    else:
        # Not aggregated models
        # Unsupervised LDA model case
        if not semi:
            # Load LDA model and corresponding dictionary ------------------------------------------------------------------
            ldaModel = parentDir + '/LDAModels/WSCombinedModels/'+fmodel+'/' + str(model_name)
            lda = models.ldamodel.LdaModel.load(ldaModel)
            #print('Model Topics Number:', lda.num_topics)

            dic = parentDir + '/LDAModels/WSCombinedModels/'+fmodel+'/dic_' + str(model_name) + '.dict'
            modelDic = Dictionary.load(dic)

            # Recreating the topics dictionaries ---------------------------------------------------------------------------
            ldaTopics = lda.show_topics(formatted=False, num_topics=lda.num_topics, num_words=15)
            # print('Loaded LDA Topics Dictionaries, top 15 words:', *ldaTopics, sep='\n')

            # Get manual labels --------------------------------------------------------------------------------------------
            labels = []
            with open(
                    parentDir + '/modelLabels/combinedCategorisation/manualLabels_' + model_name + '.txt',
                    'r',
                    encoding="utf-8") as labelsFile:
                labelLine = labelsFile.read().split('\n')
                for line in labelLine:
                    if line:
                        labels.append(line.split(', '))

            labels = [[int(label[0]), label[1]] for label in labels]
            labels = list(itertools.chain.from_iterable(labels))
            # print('\n Loaded Model Labels:', labels)

        else:
            # Semi-unsupervised LDA model case

            # Load LDA model and corresponding dictionary
            ldaModel = parentDir + '/LDAModels/WSCombinedModels/'+fmodel+'/semisupervised_' + str(model_name)
            lda = models.ldamodel.LdaModel.load(ldaModel)
            #print('topics number:', lda.num_topics)

            dic = parentDir + '/LDAModels/WSCombinedModels/'+fmodel+'/dic_semisupervised_' + str(model_name) + '.dict'
            modelDic = Dictionary.load(dic)

            # Recreating the topics dictionaries
            ldaTopics = lda.show_topics(formatted=False, num_topics=lda.num_topics, num_words=20)
            # print('LDA Topics ', *ldaTopics, sep='\n')

            # Get manual labels --------------------------------------------------------------------------------------------
            labels = []
            with open(
                    parentDir + '/modelLabels/combinedCategorisation/manualLabels_model_' + model_name + '_semisupervised.txt',
                    'r',
                    encoding="utf-8") as labelsFile:
                labelLine = labelsFile.read().split('\n')
                for line in labelLine:
                    if line:
                        labels.append(line.split(', '))

            labels = [[int(label[0]), label[1]] for label in labels]
            labels = list(itertools.chain.from_iterable(labels))

    return lda, dic, modelDic, labels

def loadRequirements(category):
    '''
    Load requirements from the "category".txt
    :param category: label name
    '''
    requirementsList = []
    with open(parentDir + '/Corpora/requirementsCorpus/req_' + category + '.txt', 'r',
              encoding="utf-8") as filteredList:
        requirements = filteredList.read().split('\n')
        for req in requirements:
            if req:
                requirementsList.append(req.split(" | "))
    return requirementsList

def getCategory(item, modelDic, lda, labels, gt, acronyms, exp, ecssMultiwords, ecssMultiwordsWithAcronyms, stopset):
    '''
    :param item: requirement and its ground truth
    :param modelDic: model dictionary
    :param lda: model
    :param labels: labels of the model topics
    :param gt: Ground truth of requirements categorised so far
    :param acronyms: list of ECSS acronyms used to preprocess the requirement
    :param exp: expansion of above acronyms
    :param ecssMultiwords: list of ECSS mulit-words used to preprocess the requirement
    :param ecssMultiwordsWithAcronyms: list of ECSS multi-words with acronyms used to preprocess the requirement
    :param stopset: list of stop words used to preprocess the requirement
    :return: category to which the model has assigned the requirement
    '''
    req = item[0]
    gt.append(item[1])

    # pre-process requirement
    req = NLPPipe(req, acronyms, exp, ecssMultiwords, ecssMultiwordsWithAcronyms, stopset)
    # print('after process:', req)

    # Use the same dictionary as pre-trained model to  convert a list of words into bag of word format
    unseen_doc = modelDic.doc2bow(req)

    # get topic probability distribution for the unseen document
    vector = lda[unseen_doc]
    sorted_vector = sorted(vector, key=itemgetter(1), reverse=True)

    # Treshold - keep top 2 topics associated, with probabilities
    #results = list(map(list, sorted_vector[0:2]))
    results = list(map(list, sorted_vector))

    # associate top results with manually assigned labels
    for item in results:
        item[0] = labels[labels.index(item[0] + 1) + 1]

    # Sum probabilities associated to same label
    resultsSum = []
    values = set(map(lambda x: x[0], results))
    for x in values:
        newlist = [[y[1] for y in results if y[0] == x]]

        resultsSum.append([x, sum(newlist[0])])

    resultsSum = sorted(resultsSum, key=itemgetter(1), reverse=True)

    #if no topics associated to unseen text (very rare)
    if not resultsSum:
        resultsSum=[['unknown', 1]]

    return gt, resultsSum

# JS method
def runAggregatedCategorisation(aggregated, semi, model_name, fmodel, categories):
    '''
    Use model aggregated with JS method to categorise requirements
    :param aggregated: if True, the model used here has been aggregated with JS divergence
    :param semi: if True, the model is based on an aggregation of semi-supervised models. If false, based on unsupervised models.
    :param model_name: name of model to be loaded
    :param fmodel: folder where model to be loaded has been saved
    :param categoriesList: list of category labels
    '''
    start = time.time()

    # Load Model -------------------------------------------------------------------------------------------------------
    lda, dic, modelDic, labels = getModel(model_name, aggregated, semi, fmodel)


    for category in categories:
        # Get test requirements List ---------------------------------------------------------------------------------------
        requirementsList = loadRequirements(category)

        # Categorisation ---------------------------------------------------------------------------------------------------
        gt = []
        allResults = []
        acronyms, exp, ecssMultiwords, ecssMultiwordsWithAcronyms = loadAcronymsMW()
        stopset = loadStopset()

        for item in requirementsList:
            gt, results = getCategory(item, modelDic, lda, labels, gt, acronyms, exp, ecssMultiwords, ecssMultiwordsWithAcronyms, stopset)
            allResults.append(results)

        print('\n All Results for category', category, ' :')

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

# WS methods
def combinedCategorisation(modelsList, categoriesList):
    '''
    Categorisation with weighted sum
    :param modelsList: list of models to sum, with their weights
    :param categoriesList: labels of categories
    :return: precision and MRR scores
    '''

    precisionRes=[]
    mrrRes=[]

    for category in categoriesList:
        # Part 1: For one category and for each model, get categorisation output
        allResults=[]
        weigths=[]
        for m in modelsList:
            model_name=m[0]
            semi=m[1]
            weigths.append(m[3])#
            fmodel=m[4]
            aggregated=False

            # Load Model
            lda, dic, modelDic, labels = getModel(model_name, aggregated, semi, fmodel)

            # Get test requirements List
            requirementsList = loadRequirements(category)

            #get Categorisation results
            gt = []
            modelResults = []
            acronyms, exp, ecssMultiwords, ecssMultiwordsWithAcronyms = loadAcronymsMW()
            stopset = loadStopset()

            for item in requirementsList:
                gt, results = getCategory(item, modelDic, lda, labels, gt, acronyms, exp, ecssMultiwords,
                                          ecssMultiwordsWithAcronyms, stopset)
                modelResults.append(results)
            #print(*modelResults, sep='\n')
            allResults.append(modelResults)

        # Part 2: Combine Results with Weighted Sum
        combinedResults=[]
        for reqIndex in range(0, len(requirementsList)):

            # Get each model output for requirement
            resultsInput = [model[reqIndex] for model in allResults]

            # Combining each model output per label
            newResult=[]

            # get list of categories found in all model results involved
            listCategories = [y[0] for y in list(itertools.chain.from_iterable(resultsInput))]
            listCategories = list(set(listCategories))

            for label in listCategories:
                sum=0
                sum_weigth=0
                # for each model output for requirement of interest
                for r in resultsInput:
                    labels=[x[0] for x in r]
                    w = weigths[resultsInput.index(r)]
                    if label in labels:
                        r=list(itertools.chain.from_iterable(r))
                        p = r[r.index(label)+1]
                    else:
                        p=0
                    sum=sum+w*p
                    sum_weigth=sum_weigth+w
                newLabelP=sum/sum_weigth
                newResult.append([label,newLabelP])

            newResult=sorted(newResult, key=itemgetter(1), reverse=True)
            s=0
            for x in newResult:
                s=s+x[1]
            if s>1:
                print('Careful one combined probability above 1!!')
            combinedResults.append(newResult[0:2])

        # Part 3: Evaluation
        # --> Accuracy
        firstChoice = [item[0][0] for item in combinedResults]
        firstChoiceAccuracy = accuracy_score(gt, firstChoice)
        precisionRes.append(firstChoiceAccuracy)

        # --> Mean Reciprocal Ranking
        bigScore = 0
        for item in combinedResults:
            i = combinedResults.index(item)
            score = 0
            if item[0][0] == gt[i]:
                score = 1
            elif len(item) > 1:
                if item[1][0] == gt[i]:
                    score = 0.5
            bigScore = bigScore + score
        meanReciprocalrank = bigScore / len(requirementsList)
        mrrRes.append(meanReciprocalrank)

    return allResults, precisionRes, mrrRes

def runWSCategorisation(semi,update,categoriesList):
    '''
    Combine models categorisation results with a Weighted Sum
    Calls combinedCategorisation function
    :param semi: semi-supervised models: True, unsupervised: False
    :param update: always set to false here, was used for conference paper
    :param categoriesList: list of categories
    '''

    Prec = []
    MRR = []

    if semi:
        # Optimised weights found with hyperopt_optimisation.py
        hyper=[[0.8047588519384588, 0.08092085626584467, 0.6021272378926299]]

        for hyper_pmt1 in hyper:
            # Format = [model_name, semi, update, weight, model folder]
            modelsListSemi = [['wiki_22_10', semi, update, hyper_pmt1[0], 'wiki_semisupervised'],
                              ['wiki_22_11', semi, update, hyper_pmt1[0], 'wiki_semisupervised'],
                              ['wiki_22_12', semi, update, hyper_pmt1[0], 'wiki_semisupervised'],
                              ['wiki_22_13', semi, update, hyper_pmt1[0], 'wiki_semisupervised'],
                              ['wiki_22_14', semi, update, hyper_pmt1[0], 'wiki_semisupervised'],
                              ['wiki_22_15', semi, update, hyper_pmt1[0], 'wiki_semisupervised'],
                              ['wiki_22_16', semi, update, hyper_pmt1[0], 'wiki_semisupervised'],
                              ['wiki_22_17', semi, update, hyper_pmt1[0], 'wiki_semisupervised'],
                              ['wiki_22_18', semi, update, hyper_pmt1[0], 'wiki_semisupervised'],
                              ['wiki_22_19', semi, update, hyper_pmt1[0], 'wiki_semisupervised'],
                              ['model_30_3', semi, update, hyper_pmt1[1], 'reports_semisupervised'],
                              ['reports_30_31', semi, update, hyper_pmt1[1], 'reports_semisupervised'],
                              ['reports_30_32', semi, update, hyper_pmt1[1], 'reports_semisupervised'],
                              ['reports_30_33', semi, update, hyper_pmt1[1], 'reports_semisupervised'],
                              ['reports_30_34', semi, update, hyper_pmt1[1], 'reports_semisupervised'],
                              ['reports_30_35', semi, update, hyper_pmt1[1], 'reports_semisupervised'],
                              ['reports_30_36', semi, update, hyper_pmt1[1], 'reports_semisupervised'],
                              ['reports_30_37', semi, update, hyper_pmt1[1], 'reports_semisupervised'],
                              ['reports_30_38', semi, update, hyper_pmt1[1], 'reports_semisupervised'],
                              ['reports_30_39', semi, update, hyper_pmt1[1], 'reports_semisupervised'],
                              ['model_24_41', semi, update, hyper_pmt1[2], 'books_semisupervised'],
                              ['books_24_2', semi, update, hyper_pmt1[2], 'books_semisupervised'],
                              ['books_24_3', semi, update, hyper_pmt1[2], 'books_semisupervised'],
                              ['books_24_4', semi, update, hyper_pmt1[2], 'books_semisupervised'],
                              ['books_24_5', semi, update, hyper_pmt1[2], 'books_semisupervised'],
                              ['books_24_6', semi, update, hyper_pmt1[2], 'books_semisupervised'],
                              ['books_24_7', semi, update, hyper_pmt1[2], 'books_semisupervised'],
                              ['books_24_8', semi, update, hyper_pmt1[2], 'books_semisupervised'],
                              ['books_24_9', semi, update, hyper_pmt1[2], 'books_semisupervised'],
                              ['books_24_10', semi, update, hyper_pmt1[2], 'books_semisupervised']]
            print('------')
            print(hyper_pmt1)

            # Apply categorisation
            scores1, precisionScores1, mrrScores1 = combinedCategorisation(modelsListSemi, categoriesList)
            print(precisionScores1)
            print(mrrScores1)
            area = np.trapz(precisionScores1)
            print('Area Unsupervised: ', area)

            Prec.append(precisionScores1)
            MRR.append(mrrScores1)

    else:
        hyper = [[0.919042157570666, 0.279064077256177, 0.27358645988980634]]

        for hyper_pmt2 in hyper:
            # Format = [model_name, semi, update, weight, model folder]
            modelsListUnsup = [['wiki_22_10', semi, update, hyper_pmt2[0], 'wiki_unsupervised'],
                              ['wiki_22_11', semi, update, hyper_pmt2[0], 'wiki_unsupervised'],
                              ['wiki_22_12', semi, update, hyper_pmt2[0], 'wiki_unsupervised'],
                              ['wiki_22_13', semi, update, hyper_pmt2[0], 'wiki_unsupervised'],
                              ['wiki_22_14', semi, update, hyper_pmt2[0], 'wiki_unsupervised'],
                              ['wiki_22_15', semi, update, hyper_pmt2[0], 'wiki_unsupervised'],
                              ['wiki_22_16', semi, update, hyper_pmt2[0], 'wiki_unsupervised'],
                              ['wiki_22_17', semi, update, hyper_pmt2[0], 'wiki_unsupervised'],
                              ['wiki_22_18', semi, update, hyper_pmt2[0], 'wiki_unsupervised'],
                              ['wiki_22_19', semi, update, hyper_pmt2[0], 'wiki_unsupervised'],
                              ['model_30_21', semi, update, hyper_pmt2[1], 'reports_unsupervised'],
                              ['reports_30_22', semi, update, hyper_pmt2[1], 'reports_unsupervised'],
                              ['reports_30_23', semi, update, hyper_pmt2[1], 'reports_unsupervised'],
                              ['reports_30_24', semi, update, hyper_pmt2[1], 'reports_unsupervised'],
                              ['reports_30_25', semi, update, hyper_pmt2[1], 'reports_unsupervised'],
                              ['reports_30_26', semi, update, hyper_pmt2[1], 'reports_unsupervised'],
                              ['reports_30_27', semi, update, hyper_pmt2[1], 'reports_unsupervised'],
                              ['reports_30_28', semi, update, hyper_pmt2[1], 'reports_unsupervised'],
                              ['reports_30_29', semi, update, hyper_pmt2[1], 'reports_unsupervised'],
                              ['reports_30_30', semi, update, hyper_pmt2[1], 'reports_unsupervised'],
                              ['books_24_34', semi, update, hyper_pmt2[2], 'books_unsupervised'],
                              ['books_24_2', semi, update, hyper_pmt2[2], 'books_unsupervised'],
                              ['books_24_3', semi, update, hyper_pmt2[2], 'books_unsupervised'],
                              ['books_24_4', semi, update, hyper_pmt2[2], 'books_unsupervised'],
                              ['books_24_5', semi, update, hyper_pmt2[2], 'books_unsupervised'],
                              ['books_24_6', semi, update, hyper_pmt2[2], 'books_unsupervised'],
                              ['books_24_7', semi, update, hyper_pmt2[2], 'books_unsupervised'],
                              ['books_24_8', semi, update, hyper_pmt2[2], 'books_unsupervised'],
                              ['books_24_9', semi, update, hyper_pmt2[2], 'books_unsupervised'],
                              ['books_24_10', semi, update, hyper_pmt2[2], 'books_unsupervised']]
            print('------')
            print(hyper_pmt2)
            scores1, precisionScores1, mrrScores1 = combinedCategorisation(modelsListUnsup, categoriesList)
            print(precisionScores1)
            print(mrrScores1)
            #area = np.trapz(precisionScores1)
            #print('Area Semisupervised: ', area)

            Prec.append(precisionScores1)
            MRR.append(mrrScores1)

    # Plot results: Accuracy + MRR
    labels = ['AOCS/GNC', 'Communication', 'Environment', 'OBDH', 'Power', 'Propulsion', 'Thermal']

    for item in Prec:
        plt.plot(labels, item)

    plt.xlabel('Categories')
    plt.ylabel('Mean Accuracy')
    h1, = plot([1, 1], 'k-')
    h2, = plot([1, 1], 'k--')
    h3, = plot([1, 1], 'k-.')
    h4, = plot([1, 1], 'k-.')
    h5, = plot([1, 1], 'k--')
    h6, = plot([1, 1], 'k-.')
    h7, = plot([1, 1], 'k-.')

    legend((h1, h2, h3, h4, h5, h6, h7), ('w1', 'w2', 'w3', 'w4', 'w5', 'w6', 'w7'))
    h1.set_visible(False)
    h2.set_visible(False)
    h3.set_visible(False)
    h4.set_visible(False)
    h5.set_visible(False)
    h6.set_visible(False)
    h7.set_visible(False)
    plt.show()

    for item in MRR:
        plt.plot(labels, item)

    plt.xlabel('Categories')
    plt.ylabel('Mean Accuracy')
    h1, = plot([1, 1], 'k-')
    h2, = plot([1, 1], 'k--')
    h3, = plot([1, 1], 'k-.')
    h4, = plot([1, 1], 'k-.')
    h5, = plot([1, 1], 'k--')
    h6, = plot([1, 1], 'k-.')
    h7, = plot([1, 1], 'k-.')

    legend((h1, h2, h3, h4, h5, h6, h7), ('w1', 'w2', 'w3', 'w4', 'w5', 'w6', 'w7'))
    h1.set_visible(False)
    h2.set_visible(False)
    h3.set_visible(False)
    h4.set_visible(False)
    h5.set_visible(False)
    h6.set_visible(False)
    h7.set_visible(False)
    plt.show()

    return

# main
def main():
    '''
    Run categorisation of unseen requirements, either with the WS or the aggregated models
    '''

    categoriesList = ['AOCS', 'com', 'environment', 'OBDH', 'Power', 'prop', 'thermal']
    update = False
    semi=True

    # Run Weighted Sum for semi-supervised or unsupervised models
    runWSCategorisation(semi, update, categoriesList)

    # Run JS Divergence for semi-supervised or unsupervised models
    aggregated = True
    runAggregatedCategorisation(aggregated, semi, 'aggregUnsup', 'aggregatedModel',categoriesList)

    return


'''---------------------------
------------ MAIN ------------
---------------------------'''
if __name__ == "__main__":
    main()

