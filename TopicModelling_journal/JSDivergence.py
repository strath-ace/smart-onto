# This Source Code Form is subject to the terms of the Mozilla Public ---------------------
# License, v. 2.0. If a copy of the MPL was not distributed with this ---------------------
# file, You can obtain one at http://mozilla.org/MPL/2.0/. */ -----------------------------
# ---------------- Copyright (C) 2021 University of Strathclyde and Author ----------------
# ------------------------------- Author: Audrey Berquand ---------------------------------
# ------------------------- e-mail: audrey.berquand@strath.ac.uk --------------------------

'''
Aggregate pre-trained LDA models with the JS divergence methods.
'''

import gensim
from gensim import corpora, models
from TopicModelling_journal.categorisation import *
from TopicModelling_journal.TM_methods import *

sys.path.append("..")
fileDir = os.path.dirname(os.path.abspath(__file__))  #
parentDir = os.path.dirname(fileDir)  # Directory of the Module directory

'''---------------------------
---------- FUNCTIONS ---------
---------------------------'''

def aggregate(d):
    '''
    Used to aggregate len(d) models which are similar
    :param d: List of (list of float): Per word distributions to aggregate
    The input distributions should be built on the same dictionary
    :return: agg_d: aggregated topic distribution
    '''

    list = np.array(d)
    res = np.sum(list, 0)
    agg_d = res/len(d)

    return agg_d

def sortedDistribution(aggregatedTopics, dictionary):
    '''
    Organise words distribution of aggregated topics
    :param aggregatedTopics: topics aggregated
    :param resulting word distribution
    '''
    topics = []

    for eachTopic in aggregatedTopics:
        wordDist = []
        # Replace probability by ["word", probability], word corresponds to dictionary
        idx_c = -1
        for eachProb in eachTopic:
            idx_c = idx_c + 1
            wordDist.append([dictionary[idx_c], eachProb])
        # sort, and keep top n words
        wordDist = sorted(wordDist, key=itemgetter(1), reverse=True)
        wordDist = wordDist[0:19]
        topics.append(wordDist)

    return topics

def generate_aggregated_model(modelsList,outputModelName, semi):
    '''
    Aggregate models listed in modelsList as per the JS divergence method
    :param modelsList: list of models to aggregate
    :param outputModelName: name under which the aggregated model is saved
    :param semi: True if the aggregated model were trained in a semi-supervised fashion, False otherwise
    '''

    topTopics = 500

    # JS Divergence pre-processing:
    # Need to merge probability distribution over different dictionary sizes
    allTopics=[]
    countModel=0
    for modelType in modelsList:
        countModel=countModel+len(modelType[1])
        for model_name in modelType[1]:
            fmodel=modelType[0]

            # load model and topics
            if semi:
                ldaModel = parentDir + '/LDAmodels/'+fmodel+'/semisupervised_' + str(model_name)
            else:
                ldaModel = parentDir + '/LDAmodels/' + fmodel + '/' + str(model_name)
            lda = models.ldamodel.LdaModel.load(ldaModel)
            ldaTopics = lda.show_topics(formatted=False, num_topics=lda.num_topics, num_words=topTopics)

            for (idTopic, topicDist) in ldaTopics:
                allTopics.append(topicDist)

    print('\n Original Number of Topics: ', len(allTopics), ', from ', countModel , ' models')

    # Set "vocabulary" from aggregated topics
    vocabulary = []
    for topic in allTopics:
        for (word,p) in topic:
            vocabulary.append(word)
    vocabulary=list(set(vocabulary))
    vocabulary=sorted(vocabulary)

    # Create aggregated model dictionary
    dictionary = corpora.Dictionary([vocabulary])
    print('\n Dictionary Size:', dictionary)
    dictionary.save(parentDir + '/LDAmodels/aggregatedModel/dic_' + str(outputModelName) + '.dict')

    # Reorganise Probability Distributions according to model dictionary
    # Output: matrix of size(Number of topics from models to merge, topic vocabulary size)
    # Using the size of the vocabulary from topics dictionaries top n words as it will less computationally expensive than
    # using the whole model dictionary. Also allow to be independent from models dictionaries when merging heterogeneous data

    topicDist= np.zeros((len(allTopics), len(vocabulary)))
    idTopic=-1
    for topic in allTopics:
        idTopic=idTopic+1
        for (word, p) in topic:
            idWord=vocabulary.index(word)
            topicDist[idTopic][idWord]=p

    # Perform Jensen-Shannon Divergence to identify similar topics
    # If distributions are very similar, divergence closer to 0
    # If distributions are very different, divergence closer to 1
    divergenceMatrix=np.ones((len(topicDist), len(topicDist)))
    rankBase=-1
    scores=[]
    for base_topic in topicDist:
        rankBase=rankBase+1
        rankComparedTopic=-1
        for topic in topicDist:
            rankComparedTopic=rankComparedTopic+1
            js_distance=round(distance.jensenshannon(base_topic, topic), 4)
            divergenceMatrix[rankBase][rankComparedTopic]= js_distance
            scores.append(js_distance)

    '''
    fig2 = plt.figure()
    plt.hist(scores, np.arange(0, 1, 0.05))
    plt.xlabel('JS distance')
    plt.grid()
    plt.ylabel('distribution over models')
    plt.show()'''

    print('average: ', np.mean(divergenceMatrix))

    # Replace all diagonals elements by 1)
    np.fill_diagonal(divergenceMatrix, 1)

    # Aggregate similar topics to generate new per-topic word distribution
    # the output matrix topicsToaggregate should have length equal to the total number of topics
    # Topics which are not similar to any other topics will be kept as such
    JS_threshold = 0.3
    topicsToaggregate = []
    for row in divergenceMatrix:
        temp=np.nonzero(row <= JS_threshold)
        if temp:
            topicsToaggregate.append(list(temp[0]))
        else:
            topicsToaggregate.append([])

    # Now we know which topics to aggregate
    # Matrix in which we will store the aggregated per-word topic distribution
    aggregatedTopics=[]
    listAggregationPerformed=[]
    notAggregated=[]
    idx=-1
    for item in topicsToaggregate:
        idx = idx+1
        if not item:
            # Meaning if topic has no similar topic, maintain as it it
            aggregatedTopics.append(topicDist[idx])
            notAggregated.append(idx)
        else:
            topics2aggregate = [topicDist[idx]]
            l=[idx]
            for target in item:
                topics2aggregate.append(topicDist[target])
                l.append(target)
            l=sorted(l)

            # If this combination has not been done before
            if l not in listAggregationPerformed:
                aggregatedTopic=aggregate(topics2aggregate)
                aggregatedTopics.append(aggregatedTopic)
                listAggregationPerformed.append(l)

    # Example of Aggregated topics:
    # [0, 27, 113]
    #[('radio', 0.040934443), ('frequency', 0.019000826), ('signal', 0.018600175), ('antenna', 0.018114354), ('receiver', 0.017933553), ('wave', 0.013453547), ('communication', 0.0125021925), ('transmitter', 0.010083963), ('band', 0.008365398), ('transmission', 0.0067610745)]
    #[('radio', 0.035162706), ('antenna', 0.03394519), ('frequency', 0.01823096), ('signal', 0.015422261), ('receiver', 0.014971288), ('wave', 0.0144607555), ('transmitter', 0.008502046), ('communication', 0.007024172), ('band', 0.0056262524), ('transmission', 0.005529613)]
    #[('radio', 0.051336788), ('frequency', 0.026586065), ('signal', 0.02518409), ('receiver', 0.023991005), ('antenna', 0.017223552), ('wave', 0.016432464), ('band', 0.015262359), ('transmitter', 0.013103108), ('communication', 0.011151363), ('wireless', 0.0074550062)]
    # yields:
    #[['radio', 0.04247797901431719], ['antenna', 0.02309436599413554], ['frequency', 0.021272617081801098],     ['signal', 0.01973550859838724], ['receiver', 0.018965282166997593], ['wave', 0.0147822555154562],['transmitter', 0.010563038910428682], ['communication', 0.010225909296423197], ['band', 0.00975133649383982]['television', 0.00624640720585982]

    # Optional: The matrix aggregatedTopics include the probability distribution over all words of the dictionary
    # For readibility, this matrice is mapped into a ["word", prob] shape and stored in finalTopics ([len(aggregatedTopics, topTopics])
    #topics=sortedDistribution(aggregatedTopics, dictionary)

    print('\n -- OUTPUT OF AGGREGATION --')
    print('JS divergence threshod:', js_distance)
    print(len(notAggregated), ' topics not aggregated')
    print(len(listAggregationPerformed), ' aggregation performed')
    print('List of topics combined: ', *listAggregationPerformed, sep='\n')
    print('Final number of topics in aggregated model', len(aggregatedTopics))

    # Initialise LDA model object
    # eta represents the a-priori per-topic word distribution, the aggregatedTopics matrix represents the posteriori (post training) per-topic word distribution. It is stored in eta for the sake
    # of easlity transmitting it as it is not possible to initialise the posteriori distribution ( parameter expElogbeta) for the model object in gensim.
    lda_model_test = gensim.models.ldamodel.LdaModel(num_topics=len(aggregatedTopics), id2word=dictionary, eta=aggregatedTopics)
    lda_model_test.save(parentDir + '/LDAmodels/aggregatedModel/' + str(outputModelName))

    return

def main():
    '''
    Collect user inputs
    Run aggregation of models or load pre-trained one
    '''

    # Generate a new aggregated model: True
    # Load a pre-trained aggregate model: False
    generateNewModel=False

    if generateNewModel:

        # List of models to be aggregated
        modelsListSemi = [
            ['wiki_semisupervised', ['wiki_22_10', 'wiki_22_11', 'wiki_22_12', 'wiki_22_13', 'wiki_22_14','wiki_22_15', 'wiki_22_16','wiki_22_17', 'wiki_22_18', 'wiki_22_19']],
            ['reports_semisupervised', ['model_30_3','reports_30_31','reports_30_32','reports_30_33','reports_30_34','reports_30_35','reports_30_36','reports_30_37','reports_30_38','reports_30_39']],
            ['books_semisupervised', ['model_24_41', 'books_24_2','books_24_3', 'books_24_4', 'books_24_5','books_24_6','books_24_7','books_24_8','books_24_9','books_24_10' ]]
                ]

        modelsListUnsup = [
            ['wiki_unsupervised',
             ['wiki_22_10', 'wiki_22_11', 'wiki_22_12', 'wiki_22_13', 'wiki_22_14', 'wiki_22_15', 'wiki_22_16',
              'wiki_22_17', 'wiki_22_18', 'wiki_22_19']],
            ['reports_unsupervised',
             ['model_30_21', 'reports_30_22', 'reports_30_23', 'reports_30_24', 'reports_30_25', 'reports_30_26',
              'reports_30_27', 'reports_30_28', 'reports_30_29', 'reports_30_30']],
            ['books_unsupervised',
             ['books_24_34', 'books_24_2', 'books_24_3', 'books_24_4', 'books_24_5', 'books_24_6', 'books_24_7',
              'books_24_8', 'books_24_9', 'books_24_10']]
        ]

        # Enter output model name
        outputModelName = 'aggregUnsup'

        # if semi-supervised True, False otherwise
        semi=False

        # Run aggregation
        generate_aggregated_model(modelsListUnsup, outputModelName, semi)
        print('New Aggregated Model Successfully Generated')

        # load model (optional)
        lda = models.ldamodel.LdaModel.load(parentDir + '/LDAmodels/aggregatedModel/' + str(outputModelName))
        # the posteriori per-topic word distribution was stored as eta
        lda.expElogbeta = lda.eta
        dic = parentDir + '/LDAmodels/aggregatedModel/dic_' + str(outputModelName) + '.dict'
        dictionary = Dictionary.load(dic)

    else:
        # Load Saved Aggregated Model
        model_name = 'aggregUnsup'
        lda = models.ldamodel.LdaModel.load(parentDir + '/LDAmodels/aggregatedModel/' + str(model_name))
        # the posteriori per-topic word distribution was stored as eta
        lda.expElogbeta = lda.eta
        dic = parentDir + '/LDAmodels/aggregatedModel/dic_' + str(model_name) + '.dict'
        dictionary= Dictionary.load(dic)

    # Get topics dictionaries
    # We cannot just use lda.show_topics as we did not train the aggregated lda model
    topics=sortedDistribution(lda.eta, dictionary)
    for t in topics[451:len(topics)]:
        print(topics.index(t),':')
        print(t)
        print('----- \n')

    # Get per-document topic distribution for unseen data
    # get unseen documents
    categoriesList = ['AOCS', 'com', 'environment', 'OBDH', 'Power', 'prop', 'thermal']

    for category in categoriesList:
        print('\n\n -- ', category ,'--')
        requirementsList = loadRequirements(category)
        print(len(requirementsList), ' requirements to verify')
        acronyms, exp, ecssMultiwords, ecssMultiwordsWithAcronyms = loadAcronymsMW()
        stopset = loadStopset()

        for req in requirementsList:
            print('\n -------')
            print(requirementsList.index(req)+1, '\n')
            req = NLPPipe(req[0], acronyms, exp, ecssMultiwords, ecssMultiwordsWithAcronyms, stopset)
            # transform requirement to bow according to aggregated model dictionary
            unseen_doc = dictionary.doc2bow(req)
            vector=lda[unseen_doc]
            vector=sorted(vector, key=itemgetter(1), reverse=True)
            print(vector)
            for item in vector[0:2]:
                print(item)
                print(topics[item[0]], '\n')
    return

'''---------------------------
------------ MAIN ------------
---------------------------'''
if __name__ == "__main__":
    main()






