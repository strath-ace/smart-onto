
# This Source Code Form is subject to the terms of the Mozilla Public ---------------------
# License, v. 2.0. If a copy of the MPL was not distributed with this ---------------------
# file, You can obtain one at http://mozilla.org/MPL/2.0/. */ -----------------------------
# ---------------- Copyright (C) 2021 University of Strathclyde and Author ----------------
# -------------------------------- Author: Audrey Berquand --------------------------------
# ------------------------- e-mail: audrey.berquand@strath.ac.uk --------------------------

'''
Set of methods called by functions
'''

import os
from os import listdir
from os.path import isfile, join
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

fileDir = os.path.dirname(os.path.abspath(__file__))  #
parentDir = os.path.dirname(fileDir)  # Directory of the Module directory

def cleanPreviousOutputs(targetDirectory):
    '''
    DESCRIPTION: clear previous Outputs (only .json files)
    INPUT: target directory to "clean"
    OUTPUT: target directory empty from .json files '''
    # clear previous Outputs, only remove .json files, e.g., Extracted Text with Tika and NLP Pipeline Outputs
    documents = [f for f in listdir(targetDirectory) if (isfile(join(targetDirectory, f)) and f.endswith('.json'))]
    for d in documents:
        os.remove(targetDirectory+d)
    return

def loadStopset():
    '''
    Generate list of stop words to be removed while processing requirements
    :return: list of stop words
    '''

    # Import Stop Words
    with open(parentDir + "/NLPInputs/non_character_words.txt", encoding="utf-8") as Punctuation:
        filterPunctuation = word_tokenize(Punctuation.read())

    # Common words
    with open(parentDir + "/NLPInputs/common_words.txt", encoding="utf-8") as CommonWords:
        filterCommonWords = word_tokenize(CommonWords.read())

    # Manually validated extra common words -specific to study corpora
    with open(parentDir + "/NLPInputs/corpora_common_words.txt", encoding="utf-8") as CommonWords:
        filterAdditional = word_tokenize(CommonWords.read())

    stopset = stopwords.words('english')

    for i in filterPunctuation:
        stopset.append(i)

    for i in filterCommonWords:
        stopset.append(i)

    for i in filterAdditional:
        stopset.append(i)

    return stopset

def loadAcronymsMW():
    '''
    Generate list of acronyms and multi-words to be used for processing requirements.
    This list of acronyms is based on the ECSS acronyms and glossary of terms lists and was manually curated by us.
    :return: list of acronyms, their expansion and multi-words
    '''

    # Load acronyms list, manually defined and validated
    acronymsList = []
    with open(parentDir + '/NLPInputs/ecss_acronyms.txt', 'r',
              encoding="utf-8") as inputFile:
        acLine = inputFile.read().split('\n')
        for line in acLine:
            if line:
                acronymsList.append(line.split(' | '))

    acronyms = [x[0] for x in acronymsList]
    expansions = [word_tokenize(x[1]) for x in acronymsList]
    exp = []
    for item in expansions:
        item = ' '.join(item)
        item = item.replace("-", "_")
        item = item.replace(" ", "_")
        exp.append(item)

    # Load multiwords
    # Get ECSS multiwords + additional validated terms + acronym expansions
    inputFiles = [parentDir + '/NLPInputs/ecss_2grams.txt',
                  parentDir + '/NLPInputs/ecss_3grams.txt',
                  parentDir + '/NLPInputs/ecss_4grams.txt',
                  parentDir + '/NLPInputs/ecss_5grams.txt',
                  parentDir + '/NLPInputs/ecss_6grams.txt',
                  parentDir + '/NLPInputs/ecss_9grams.txt',
                  parentDir + '/NLPInputs/spacemissiondesign_ngrams.txt',
                  parentDir + '/NLPInputs/ecss_manuallyValidatedTerms.txt']

    ecssMultiwords = []
    for file in inputFiles:
        with open(file, 'r') as input:
            words = input.read().split('\n')
            words = [x for x in words if x]
            for w in words:
                ecssMultiwords.append(word_tokenize(w))

    ecssMultiwords = ecssMultiwords + expansions

    # Separate multi words including acronyms
    ecssMultiwordsWithAcronyms = []
    with open(parentDir + '/NLPInputs/ecss_manuallyMWIncludingAcronym.txt', 'r') as input:
        words = input.read().split('\n')
        words = [x for x in words if x]
        for w in words:
            ecssMultiwordsWithAcronyms.append(word_tokenize(w))

    return acronyms, exp, ecssMultiwords, ecssMultiwordsWithAcronyms

def acronymExpansion(tokens, listReplacements, acronyms, exp):
    '''
    Search for acronyms within tokens, expand if acronyms are found
    Input: tokens, listReplacements: list all replacements done so far within the document
    Outputs: tokens with expanded acronyms when applicable, new replacements done added to list
    '''

    for word in acronyms:
        if word in tokens:
            # Replace by acronym expansion
            expansionToUse = exp[acronyms.index(word)]
            tokens[tokens.index(word)] = expansionToUse
            listReplacements.append(word)
    return tokens, listReplacements

def replaceMultiwords(tokens, listReplacements, ecssMultiwords):
    '''
    Find all multiwords in a list of tokens
    the multiwords list is based on the ECSS glossary and on the additional multiwords found in the Wiki corpus
    Input: list of tokens, listReplacements: list all replacements done so far within the document
    Output: new list of tokens including multiwords, new replacements done added to list
    '''

    #----------------------------------------------------------------------------------
    #                  USING ECSS GLOSSARY & ADDITIONAL VALIDATED TERMS
    #----------------------------------------------------------------------------------


    # Find and replace within corpus:
    for word in ecssMultiwords:
        # Look if an ecss multiword can be found in the tokens
        if word[0] in tokens:
            i = 1
            wordIndex = tokens.index(word[0])
            if wordIndex != len(tokens) - 1 and len(word) != 1:
                while i <= len(word)-1:
                    if word[i] == tokens[wordIndex+i] and wordIndex + i < len(tokens)-1:
                        ecssWords = True
                        i = i + 1
                    else:
                        ecssWords = False
                        break

                # If a multiword has been found, replace within tokens
                if ecssWords == True:
                    tokens = replacementAction(word, tokens)
                    listReplacements.append('_'.join(word))

    return tokens, listReplacements

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


