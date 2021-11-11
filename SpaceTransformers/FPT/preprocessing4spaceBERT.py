# This Source Code Form is subject to the terms of the Mozilla Public ---------------------
# License, v. 2.0. If a copy of the MPL was not distributed with this ---------------------
# file, You can obtain one at http://mozilla.org/MPL/2.0/. */ -----------------------------
# ---------------- Copyright (C) 2021 University of Strathclyde and Author ----------------
# -------------------------------- Author: Audrey Berquand --------------------------------
# ------------------------- e-mail: audrey.berquand@strath.ac.uk --------------------------

import json
import os
import re

from tqdm import tqdm
from nltk import tokenize, pos_tag
from nltk.tokenize import word_tokenize
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters

# Goal: Extract abstract part of acta astronautica journal papers
# Clean up so we end up with one txt file with each line being a sentence

# ---------------------------------------------------------
#                           METHODS
# ---------------------------------------------------------
def solveParsingError(item):
    exceptions = ['fi ber','fi delity','fi eld','fi elds', 'fi eld-of-view', 'fi gure','fi gures','fi lters','fi lter','fi ltering','fi nal','fi nally','fi nancial','fi ner','fi nish','fi nished',
                  'fi nishes','fi nishing','fi nite','fi ring','fi rm','fi rmly','fi rst','fi t','fi ts','fi tting',
                  'fi xed','fi xing']

    if any(ext in item for ext in exceptions):
        item = item.replace('fi ', 'fi')
    else:
        item = item.replace(' fi ', 'fi')

    return item

def cleanSentence(sen):
    elem2replace = ['-\n','{', '}', '\t', '�', '†', '·', '\xa0', '\displaystyle', '[...]', '...', '[citation needed]',
                    '- ', " •  ", 'fffi', "-• ", "• ", '[clarification needed]',"-", '© 2018 Elsevier Masson SAS.',
                    'All rights reserved.', '© 2017 Elsevier Masson SAS.', '© 2019 Elsevier Masson SAS.',
                    'Published by Elsevier Masson SAS.', '© 2016 Elsevier Masson SAS.','© 2017 The Authors.',
                    '© 2019 The Authors.','© 2018 The Authors.', '© 2016 The Authors.', 'Published by Elsevier Ltd.']

    for i in elem2replace:
        sen = sen.replace(i, '')

    sen = sen.replace('\n', ' ')
    sen = sen.replace('�', ' ')
    sen = sen.replace(" .", '.')
    sen = sen.replace("    ", " ")
    sen = sen.replace("  ", " ")
    sen = sen.replace("  ,", ",")
    sen = sen.replace("  ’", "’")
    sen = sen.replace("  :", ":")
    sen = sen.replace("\\'", "'")
    sen=sen.lstrip()

    return sen

def extract_content(content, title):
    punkt_param = PunktParameters()
    punkt_param.abbrev_types = set(['Fig', 'i.e', 'ref', 'e.g', 'etc', 'fig', 'Ref', 'al', 'bil','Eq'])
    sentence_splitter = PunktSentenceTokenizer(punkt_param)
    sentences = sentence_splitter.tokenize(content)
    #sentences = tokenize.sent_tokenize(content)

    filteringKeywords = ['https', 'authors', 'publisher', 'doi', ' http://', 'arXiv:', 'elsevier', 'Acta Astronautica',
                         'Advances in Space Research', 'by Elsevier Ltd', 'Aerospace Science and Technology ',
                         'Elsevier', 'Corresponding author', 'IEEE', 'EncodePages',
                         'FEFF', 'Remote Sens.', 'Available online', "Proceedings", "AIAA", "J.", 'in:', "Conference",
                         "Aeronaut.", "Aircr.", "Appl.", "version", "Aerosp.", 'AIAA',
                         'American Institute of Aeronautics and Astronautics',
                         'COSPAR', '=', 'this book', 'protective laws', 'Springer', 'Editor', 'Publisher', 'http',
                         'exclusive use', 'Copyrigh', 'purchase of the work', 'trademark', 'org/', 'Œg',
                         'exclusive use', 'Copyrigh', 'purchase of the work', 'trademark', 'org/', 'Œg',
                         '.I3', '?1', 'A.e', 'this issue', 'ID:', 'See ', 'Chapter ', '+', '[edit]', 'Wikipedia',
                         '^', '|', 'external link', 'needing additional references', 'using this site',
                         'Creative Commons Attribution-ShareAlike', 'Wikidata', 'Bibcod',
                         'arXiv:', 'Website', 'Retrieved', 'Archived from', 'Please help', 'University', '[England]',
                         'Press', 'Proceedings', 'IEEE Transactions', 'External links',
                         'New York', 'Toronto', 'Patent', '\mathbf', 'Edit links', 'Accessed', 'Find sources',
                         'Submitted manuscript', 'Vol.', '/Namespace', 'PDF', 'Dr.', 'He ', ' he ', ' his ', ' her ',
                         'She ', ' she ', 'Eng.', 'UNIVERSITY', 'Authorized licensed', 'They ', ' they ',
                         'Institute of Technology', 'Acta Astronaut.', 'IAC', 'vol.', 'pp.', 'Space Pol.', 'Int.',
                         'IAA.', 'received the M.Sc.', 'JANUARY', 'FEBRUARY', 'MARCH', 'APRIL', 'MAY', 'JUNE', 'JULY',
                         'AUGUST', 'SEPTEMBER', 'OCTOBER', 'NOVEMBER', 'DECEMBER', ' his ', ' her ', 'His ', 'Her ',
                         ' he ', ' she ', 'He ', 'She ', 'E-mail address:',
                         'Declaration of Competing Interest There is no competing interest to be declared.',
                         'Conflict of interest statement No conflict', 'List of notations', 'Keywords', '@', '.pdf',
                         ' Ed.', ' ed.', 'Digital Object Identifier', ' DOI', 'Index Terms—', 'work was supported by', 'In: ',
                         'Acknowledgement', 'acknowledgement', 'special thanks', 'prepaid basis', 'grants ',
                         'conflict of interest', 'competiting interest', 'grateful', 'reviewer', 'would like to thank',
                         'standard mail', 'additional terms may apply', 'Journal of', 'Encyclopedia.', 'Green and Co.',
                         'Edition', 'Li–CrO2LiCrO23', 'ZnAg2O1.851.50', '\text', 'Sky Telesc', 'Rev', 'J Earth  Syst',
                         ':', 'Wikimedia', 'ffiffi', ' ffi ', 'edn.', 'Institute of', 'Italy.', 'Courtesy', 'A B S T R A C T']

    content2export = []

    for sen in sentences:

        # Replace special characters
        sen=cleanSentence(sen)

        # Remove references
        sen = re.sub('\[[0-9]+\]', '', sen)
        sen = re.sub('\[[a-z]+\]', '', sen)
        sen = re.sub('\[[A-Z]+\]', '', sen)

        #Remove spaces at the start of sentences
        sen = sen.strip()

        # Parser introduces spaces around syllabus "fi", fix it
        if ' fi ' in sen:
            sen = solveParsingError(sen)

        # check if sentence at least 5 tokens
        if len(word_tokenize(sen)) > 5:

            # remove sentence that don't start with a capital letter:
            p0 = re.compile('^[A-Z]')
            # remove table of reference (finishing by number)
            p01 = re.compile('[0-9]\.$')
            # remove sentences that are only capital letters
            p02 = re.compile('[a-z]')

            if re.findall(p0, sen) and not re.findall(p01, sen) and re.findall(p02, sen):
                # don't address sentences including no-no words
                if not any(word in sen for word in filteringKeywords):

                    #check not mostly numbers
                    countNbs = len([c for c in sen if c.isdigit() == True])
                    p1 = countNbs / len(sen)

                    #or special characters..
                    countSpecialChar = len([c for c in sen if c.isalnum() == False])
                    p2 = countSpecialChar / len(sen)

                    if p1 < 0.3 and p2 < 0.3:
                        #specific to publications: remove references format: year
                        p3 = re.compile('\([0-9]{4}\)')
                        # remove reference lines                                                                         ###(formatting) e.g.  1. Introduction
                        p4 = re.compile('^[A-Z]\. ')
                        p5 = re.compile('^\[[0-9].*\]+')
                        p6 = re.compile('[0-9]{4}.$')

                        if not re.findall(p3, sen) and not re.findall(p4, sen) and not re.findall(p5, sen) and not \
                                re.findall(p6, sen):

                            # checks that question mark is at the end of sentence
                            p7 = re.compile('\?')
                            p8 = re.compile('\?$')

                            if not re.findall(p7, sen):
                                content2export.append(sen)
                            elif re.findall(p8, sen):
                                content2export.append(sen)



    if len(content2export) == 0:
        #print("no content found for ", fileName + title)
        print("no content found for ", title)

    return content2export

def extract_abstract(content, title, journal, fileName):
    #print("\n-----------------------------------------------")
    #print(f"Extracting abstract from {title}:")
    #print("-----------------------------------------------")
    punkt_param = PunktParameters()
    punkt_param.abbrev_types = set(['Fig', 'i.e', 'ref', 'e.g', 'etc', 'fig', 'Ref', 'al', 'bil', 'Eq', 'c.g', 'v.s',
                                    'vs', 'ie', 'sq', 'et', 'U.S', 'p', 'Dr','deg', 'Lat', 'Long', 'Geog',
                                    'lat','long', "eg", "w.r.t"])

    # Sentences that contain the following words should not be extracted
    filteredSen = ['Acta Astronautica', 'Email', 'doi', 'https', '@', 'ghts reserved', 'Published by Elsevier Ltd',
                   'Keywords', 'Corresponding author', 'Manuscript received', 'Date of publication', ' Grant ',
                   ' grant ', 'Digital Object Identifier', 'Doi', 'Color versions', 'funding',
                   'work was supported', ' is with ', ' are with ', 'Article history:', 'Accepted', 'elsevier']

    sentence_splitter = PunktSentenceTokenizer(punkt_param)
    sentences = sentence_splitter.tokenize(content)

    sentences = [sen.replace('\n', ' ') for sen in sentences]

    # Identify journal of paper, different papers have different format for the abstract
    if 'ActaAstronautica' in journal:
        # Get firstSentence of abstract
        start = [x for x in sentences if 'A B S T R A C T' in x]

        if not start:
            abstract = []
            txt_file = open("abstractMissing.txt", "a", encoding='utf-8')
            txt_file.write('Acta Astronautica: ' + fileName + ': ' + title + '\n')
            txt_file.close()
            return abstract

        indexStart = sentences.index(start[0])

        # Clean up 1st sentence: get only works after A B S T R A C T
        pattern = re.compile('A B S T R A C T \s*([^.]+|\S+)', re.IGNORECASE)
        match = pattern.search(sentences[indexStart])
        firstSentence = match.group(1)

        # Get lastSentence of abstract
        end = [y for y in sentences if 'Introduction' in y]
        # In some rare cases, not Introduction but "Objective" or "Motivation"
        if not end:
            alternativeEnd = ['Objectives', 'Motivations', 'Background', 'Purpose of this paper',
                              'Recap', 'Space and security in Europe and ESA', 'Historical perspective',
                              'Set-up and methodology', 'Overall dynamics', 'The soviet lunar program N1-L3',
                              'Context and motivation', 'The Philae landing – a bounce into the unknown',
                              'Can we be alone?',
                              'Geometric Brownian Motion (GBM) is key', 'Rosetta mission']
            end = [y for y in sentences for x in alternativeEnd if x in y]
            if not end:
                print("New alternative end needed")

        indexEnd = sentences.index(end[0])

    elif 'AdvanceInSpaceResearch' in journal:
        # Get firstSentence of abstract
        start = [x for x in sentences if 'Abstract' in x]

        if not start:
            abstract = []
            txt_file = open("abstractMissing.txt", "a", encoding='utf-8')
            txt_file.write('AdvanceInSpaceResearch: ' + fileName + ': ' + title + '\n')
            txt_file.close()
            return abstract

        indexStart = sentences.index(start[0])
        pattern = re.compile('� [0-9]+')
        start = word_tokenize(start[0])
        cut = start.index('Abstract')
        firstSentence = start[cut + 1:len(start)]
        firstSentence=' '.join(firstSentence)

        # Get lastSentence of abstract
        end = [y for y in sentences if pattern.search(y)]
        if not end:
            end = [y for y in sentences if 'Published by Elsevier Ltd on behalf of COSPAR' in y]
        indexEnd = sentences.index(end[0])
        i = 0
        while indexEnd < indexStart:
            indexEnd = sentences.index(end[i + 1])

    elif 'AerospaceScienceAndTechnology' in journal:
        # Get firstSentence of abstract
        start = [x for x in sentences if 'a b s t r a c t' in x]
        if not start:
            abstract = []
            txt_file = open("abstractMissing.txt", "a", encoding='utf-8')
            txt_file.write('AerospaceScienceAndTechnology: '+ fileName + ': ' + title + '\n')
            txt_file.close()
            return abstract
        firstSentence = start[0].split('\n\n')

        if not any(word in firstSentence[-1] for word in filteredSen):
            firstSentence = firstSentence[-1]
        else:
            firstSentence = ''


        # Get lastSentence of abstract
        end = [y for y in sentences if 'Introduction' in y]
        if not end:
            alternativeEnd = ['General context',  'Motivation', 'Background and introduction',
                              'About the photon sieve','Instruction']
            end = [y for y in sentences for x in alternativeEnd if x in y]
            if not end:
                print(f"New alternative end needed for {fileName} : {title} ")
                return

        indexStart = sentences.index(start[0])
        indexEnd = sentences.index(end[0])

    elif 'TransactionsGeoscienceRS' in journal:
        # Get firstSentence of abstract
        start = [x for x in sentences if 'Abstract—' in x]
        if not start:
            abstract = []
            txt_file = open("abstractMissing.txt", "a", encoding='utf-8')
            txt_file.write(title)
            txt_file.close()
            return abstract
        indexStart = sentences.index(start[0])

        # Clean up 1st sentence: get only works after Abstract-
        pattern = re.compile('Abstract—\s*([^.]+|\S+)', re.IGNORECASE)
        match = pattern.search(sentences[indexStart])
        firstSentence = match.group(1)

        # Get lastSentence of abstract
        end = [y for y in sentences if 'Index Terms—' in y]
        if not end:
            print("New alternative end needed")

        indexEnd = sentences.index(end[0])

    else:
        # Get firstSentence of abstract
        # Get lastSentence of abstract
        print("Abstract extraction from this journal has not been developed yet")
        return

    # Get sentences in between
    abstract = [firstSentence]

    for item in range(indexStart + 1, indexEnd - 1):
        sen = sentences[item]
        #check sentence does not contained "forbidden words"
        if not any(word in sen for word in filteredSen):
            abstract.append(sen)
    abstract=[cleanSentence(sen) for sen in abstract]

    if not abstract:
        print(f"Missing Abstract for {fileName}: {title}")
        abstract = []

    return abstract

# ---------------------------------------------------------
#                           MAIN
# ---------------------------------------------------------

if __name__ == "__main__":
    # TODO: path to the files for pre-processing
    path = "D:/SpaceRoberta/"

    inputPath = [path + 'Acta Astronautica/',
                 path + 'Wiki/',
                 path + 'Advances in Space Research/',
                 path + 'AerospaceScienceAndTechnology/',
                 path + 'Books_Original/']

    #"D:/SpaceRoberta/Books_Original/"
    # Collect corpus
    # For each document of corpus, export content, one sentence = one line
    txt_file = open("newExtraction.txt", "a", encoding='utf-8')

    totalSentences=0
    ## For every folder containing files
    for path in tqdm(inputPath):
        file2extract=[]
        count = 0
        ## for every file in that folder
        for file in os.listdir(path):
            ## if file endswith ".json"
            if file.endswith(".json"):
                ## ad to the list of files
                file2extract.append(file)

        print("\n -------------------")
        print(path)
        print(len(file2extract), ' files detected \n')

        for file in tqdm(file2extract, position=0, leave=True):

            fileName = file.replace('.json', '')
            with open(path + file, 'r') as infile:
                input = json.load(infile)

            # Extract abstract part and clean
            if 'ActaAstronautica' in path:
                title = input["metadata"]["dc:title"]
                content2export = extract_abstract(input["content"], title, 'ActaAstronautica', fileName)
                #content2export = extract_content(input["content"], title)

            if 'AdvanceInSpaceResearch' in path:
                title = input["metadata"]["dc:title"]
                content2export = extract_abstract(input["content"], title, 'AdvanceInSpaceResearch', fileName)
                #content2export = extract_content(input["content"], title)

            if 'AerospaceScienceAndTechnology' in path:
                title = input["metadata"]["dc:title"]
                content2export = extract_abstract(input["content"], title, 'AerospaceScienceAndTechnology', fileName)
                #content2export = extract_content(input["content"], title)

            # Extract Book Content
            if 'Books' in path:
                #content2export=extract_book(input["content"], fileName)
                content2export = extract_content(input["content"], fileName)

            # If input a wikipedia page, export the whole page
            if 'Wiki' in path:
                title = input["metadata"]["dc:title"]
                #content2export = extract_wiki(input["content"], title)
                content2export = extract_content(input["content"], title)

            # Export each abstract line to txt file
            if content2export:
                for item in content2export:
                    if item:
                        txt_file.write(item)
                        txt_file.write('\n')
                        count=count+1

        totalSentences=totalSentences+count
        print("\n --> Extraction Completed - ", count, " sentences in total for", path, '\n')

    print("\n", totalSentences, ' sentences extracted from whole corpus.')
    txt_file.close()
