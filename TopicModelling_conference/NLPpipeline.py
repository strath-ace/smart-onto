# This Source Code Form is subject to the terms of the Mozilla Public ---------------------
# License, v. 2.0. If a copy of the MPL was not distributed with this ---------------------
# file, You can obtain one at http://mozilla.org/MPL/2.0/. */ -----------------------------
# ---------------- Copyright (C) 2019 University of Strathclyde and Author ----------------
# -------------------------------- Author: Audrey Berquand --------------------------------
# ------------------------- e-mail: audrey.berquand@strath.ac.uk --------------------------


'''
NLPpipeline.py gathers all methods used to pre-process the corpora for the Topic Modeling study.

Main methods -----------------------------------------------------------------------------------------------------
1. corpusProcessing: Application of NLP Pipeline to all documents contained in corpus

2. replaceMultiwords: Find all multiwords in a list of tokens, multiwords are either extracted from the ECSS glossary
                      or find through collocations
3. NLPPipe: Natural Language Processing steps

Minor methods ----------------------------------------------------------------------------------------------------
1. corpusInsight: Provides some information on document corpus: number of tokens, average tokens per document/sentence
                 and dictionary size

2. replacementAction:  Replace tokens in a list of tokens by the equivalent multi word, provided the multi word is known
                      to be within the tokens list

3. replace_acronyms: Search for acronyms within tokens, expand if acronyms are found

4. tf-idf: Generates the tf-idf ranking of each corpus dictionary item, used to filter out words with lowest tf-idf

'''

import itertools
import nltk
import json
import re, os
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.collocations import *
from os import listdir
from os.path import isfile, join

fileDir = os.path.dirname(os.path.abspath(__file__))  #
parentDir = os.path.dirname(fileDir)  # Directory of the Module directory

# ------------------------------------------------------------------------------------------------------------
#                                       METHOD
# ------------------------------------------------------------------------------------------------------------

def corpusInsight(documents):
    '''
    Provides some information on document corpus: number of tokens, average tokens per document/sentence and dictionary
    size
    Input: list of tokenized documents
    '''

    lenght = []
    for d in documents:
        lenght.append(len(d))
    print('Total Number of tokens:', sum(lenght))
    print('Average number of tokens per document:', round(np.mean(lenght), 0))

    allTokens = list(itertools.chain.from_iterable(documents))
    dic = list(set(allTokens))
    #print('Dictionary size:', len(dic))

    return

def replacementAction(multiword, tokens):
    '''
    Replace tokens in a list of tokens by the equivalent multi word,
    provided the multi word is known to be within the tokens list

    Input: multiword, list of tokens
    Output: new list of tokens where the tokens of interest have been replaced by the multi word
    '''
    new_token = '_'.join(multiword)
    wordIndex = tokens.index(multiword[0])
    tokens[wordIndex] = new_token
    indices = []
    for item in multiword[1:len(multiword)]:
        indices.append(tokens.index(item))
    tokens = [v for i, v in enumerate(tokens) if i not in indices]
    return tokens

def replaceMultiwords(tokens, stopset):
    '''
    Find all multiwords in a list of tokens
    multiwords are either extracted from the ECSS glossary or find through collocations
    Input: list of tokens
    Output: new list of tokens including multiwords
    '''

    #----------------------------------------------------------------------------------
    #                  USING ECSS GLOSSARY & ADDITIONAL VALIDATED TERMS
    #----------------------------------------------------------------------------------

    # Get ECSS multiwords + additional validated terms
    inputFiles=['./Inputs4NLP/ecss_2grams.txt', './Inputs4NLP/ecss_3grams.txt', './Inputs4NLP/ecss_4grams.txt',
                    './Inputs4NLP/ecss_6grams.txt','./Inputs4NLP/ecss_9grams.txt', './Inputs4NLP/spacemissiondesign_ngrams.txt']

    ecssMultiwords= []
    wordsChanged = []

    for file in inputFiles:
        with open(file, 'r') as input:
            words = input.read().split('\n')
            words = [x for x in words if x]
            for w in words:
                ecssMultiwords.append(word_tokenize(w))

    # Find and replace within corpus:
    for word in ecssMultiwords:
        # Look if an ecss multiword can be found in the tokens
        if word[0] in tokens:
            i = 1
            wordIndex = tokens.index(word[0])
            if wordIndex != len(tokens) - 1 and len(word) != 1:
                while i <= len(word)-1:
                    if word[i] == tokens[wordIndex+i] and wordIndex + i <= len(tokens)-1:
                        ecssWords = True
                        i = i + 1
                    else:
                        ecssWords = False
                        break

                # If a multiword has been found, replace within tokens
                if ecssWords == True:
                    tokens = replacementAction(word, tokens)
                    wordsChanged.append(word)

    if wordsChanged:
        print(len(wordsChanged), ' ecss multiwords found and replaces: ', wordsChanged)


    #----------------------------------------------------------------------------------
    #                   FINDING NEW MULTIWORDS WITH NLTK COLLOCATIONS
    #----------------------------------------------------------------------------------

    # Remove stopwords
    tokens = [i for i in tokens if not i in stopset]

    # Add collocation analysis over tokens: there might be some multiwords other than those
    # Trigrams
    trigramsCount = []
    trigram_measures = nltk.collocations.TrigramAssocMeasures()
    finder = TrigramCollocationFinder.from_words(tokens)
    finder.apply_freq_filter(10)
    trigrams = finder.nbest(trigram_measures.likelihood_ratio, 5)
    # Replace trigrams in tokens
    if trigrams:
        for word in trigrams:
            tokens = replacementAction(word, tokens)
            trigramsCount.append(word)

    # Bigrams
    bigramsCount = []
    bigram_measures = nltk.collocations.BigramAssocMeasures()
    finder = BigramCollocationFinder.from_words(tokens)
    finder.apply_freq_filter(10)
    bigrams = finder.nbest(bigram_measures.likelihood_ratio, 3)
    # Replace trigrams in tokens
    if bigrams:
        for word in bigrams:
            tokens = replacementAction(word, tokens)
            bigramsCount.append(word)

    '''
    if trigramsCount:
        print(len(trigramsCount), ' trigrams found and replaces', trigramsCount)
    if bigramsCount:
        print(len(bigramsCount), ' bigrams found and replaces', bigramsCount)
    '''

    return tokens

def replace_acronyms(req):
    '''
    Search for acronyms within tokens, expand if acronyms are found
    Input: tokens
    Outputs: tokens with expanded acronyms when applicable
    '''
    # Load acronyms list, manually defined and validated
    acronymsList = []
    with open(parentDir + '/TopicModeling\Inputs4NLP/acronyms.txt', 'r',
              encoding="utf-8") as inputFile:
        acLine = inputFile.read().split('\n')
        for line in acLine:
            if line:
                acronymsList.append(line.split(', '))

    acronyms = [word_tokenize(x[0]) for x in acronymsList]
    acronyms = list(itertools.chain.from_iterable(acronyms))
    expansions = [word_tokenize(x[1]) for x in acronymsList]

    for word in req:
        if word in acronyms:
            # Replace by acronym expansion
            expansionToUse = expansions[acronyms.index(word)]
            start_index = req.index(word)
            req[req.index(word)] = expansionToUse[0]
            i = 1
            while i <= len(expansionToUse) - 1:
                req.insert(start_index + 1, expansionToUse[i])
                start_index = start_index + 1
                i = i + 1
    return req

def tf_idf(corpus):
    '''
    Generates the tf-idf ranking of each corpus dictionary item, used to filter out words with lowest tf-idf
    Input: tokens
    Outputs: .txt file with tf-idf ranking per words, starting from lowest tf-idf
    '''

    # corpus is provided as list of tokens,
    # whereas tfidf needs it as a list of sentences.
    sen = []
    for c in corpus:
        sen.append(" ".join(c))

    vectorizer = TfidfVectorizer()
    tfidf_vectorizer_vectors = vectorizer.fit_transform(sen)

    terms = vectorizer.get_feature_names()
    sums = tfidf_vectorizer_vectors.sum(axis=0)

    data = []
    for col, term in enumerate(terms):
        data.append((term, sums[0, col]))

    pd.set_option('display.max_rows', 4000)
    ranking = pd.DataFrame(data, columns=['term', 'rank'])
    ranked = ranking.sort_values('rank', ascending=True)

    with open(parentDir + '/TopicModeling/Inputs4NLP/corpusTFIDFAnalysis.txt', 'w', encoding="utf-8") as f:
        f.write(ranked.to_string(header=True, index=False))

    '''
    # Save in another format
    terms = ranked['term'].tolist()
    f = open(parentDir + '/TopicModeling/Inputs4NLP/corpusTFIDFAnalysis.txt', mode="w", encoding="utf-8")

    for x in terms:
        x=x.replace(" ", "")
        f.write(str(x))
        f.write('\n')
    f.close()

    pd.set_option('display.max_rows', 5000)
    print(ranked)
    '''

    return

def NLPPipe(sentences):
    '''
    Natural Language Processing steps
    Input: sentences
    Outputs: processed list of tokens
    '''

    # Initialisation ---------------------------------------------------------------------------------------------------
    # create and update English stop words list
    with open(parentDir + "/TopicModeling/Inputs4NLP/non_character_words.txt", encoding="utf-8") as Punctuation:
        filterPunctuation = word_tokenize(Punctuation.read())
    with open(parentDir + "/TopicModeling/Inputs4NLP/wiki_common_words.txt", encoding="utf-8") as wikiCommonWords:
        filterCommonWords = word_tokenize(wikiCommonWords.read())
    stopset = stopwords.words('english')

    for i in filterPunctuation:
        stopset.append(i)
    for i in filterCommonWords:
        stopset.append(i)

    # Initialise Lemmatizer
    wnl = WordNetLemmatizer()

    # Start Preprocessing ----------------------------------------------------------------------------------------------
    # Lower case
    tokens = sentences.lower()

    # Tokenize
    tokens = word_tokenize(tokens)

    # Trim
    tokens = [word.strip() for word in tokens]

    # Remove tokens that are only numbers
    tokens = [x for x in tokens if re.findall('[a-zA-Z]', x)]

    # Remove two tokens cannot remove otherwise
    tokens = [i for i in tokens if i not in ['\\mathbf', '\\displaystyle']]

    # Remove urls
    tokens = [re.sub(r'www.*[\r\n]*', '', token) for token in tokens]

    # Remove non English/number characters:
    tokens = [re.sub('[^A-Za-z0-9_\-/]+', '', token) for token in tokens]

    # Remove punctuation
    tokens = [i for i in tokens if i not in filterPunctuation]

    # Additional cleaning - for wikipedia
    tokens = [i.replace("'", "") for i in tokens]
    tokens = [i.replace("\\", "") for i in tokens]
    tokens = [i.replace("title=", "") for i in tokens]
    tokens = [re.sub('_$', '', i) for i in tokens]
    tokens = [re.sub('-', '_', i) for i in tokens]
    tokens = [''.join(s for s in i if not s.isdigit()) for i in tokens]

    # Remove Empty tokens
    tokens = [x for x in tokens if x]

    # Expand acronyms
    tokens = replace_acronyms(tokens)

    # Replace Multiwords + Bigrams/Trigrams
    tokens = replaceMultiwords(tokens, stopset)

    # Remove stop words from tokens
    tokens = [i for i in tokens if not i in stopset]

    # Lemmatization - currently based on wordnet
    tokens = [wnl.lemmatize(word) for word in tokens]

    return tokens

def corpusProcessing(filepath):
    '''
    Corpus preprocessing: Application of NLP Pipeline to all documents contained in corpus
    Input: directory where .json files are stored
    Output: list of pre-processed documents
    '''
    # ------------------------------------------------------------------------------------------------------------
    #                                    LOAD INPUT + NLP PRE-PROCESSING
    # ------------------------------------------------------------------------------------------------------------
    # Load all corpus .json files into a list, 'doc_set'
    # one doc_set element = one page content
    doc_set = []

    # get list of .json files
    for path in [filepath]:
        doc_list = [f for f in listdir(path) if isfile(join(path, f))]
        for doc in doc_list:
            with open(filepath + doc, 'r') as infile:
                file = json.load(infile)
            doc_set.append(file['content'])

    print('\n Starting NLP Pipeline')
    doc_preprocessed = []
    c = 0
    for i in doc_set:
        if c == 100:
            print('Analysis document', doc_set.index(i)+1, '/', len(doc_set))
            c = 0
        tokens = NLPPipe(i)

        # add tokens to document list
        doc_preprocessed.append(tokens)
        c = c + 1

    # TF-IDF filtering: for each token, measure tf-idf, the lower the tf-idf the less interesting the word is,
    # and should therefore be filtered by being added to the stopword list.
    tf_idf(doc_preprocessed)
    corpusInsight(doc_preprocessed)

    print('Corpus Preprocessed !\n')
    return doc_preprocessed
