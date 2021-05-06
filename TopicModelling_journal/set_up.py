# This Source Code Form is subject to the terms of the Mozilla Public ---------------------
# License, v. 2.0. If a copy of the MPL was not distributed with this ---------------------
# file, You can obtain one at http://mozilla.org/MPL/2.0/. */ -----------------------------
# ---------------- Copyright (C) 2021 University of Strathclyde and Author ----------------
# -------------------------------- Author: Audrey Berquand --------------------------------
# ------------------------- e-mail: audrey.berquand@strath.ac.uk --------------------------

from pip._internal import main

def install(package):
    main(['install', package])

TM_packages = {'selenium': {'Version': '3.141.0'},
                'ipython': {'Version': '7.8.0'},
                'pyldavis': {'Version': '2.1.2'},
                'gensim':{'Version': '3.7.1'}}

list_packages = []
for k in TM_packages.keys():
    list_packages.append(k)

for package in list_packages:
    print('\nInstalling', package, '...')
    install(package)

# Natural Language ToolKit
import nltk
nltk.download('all')

print('\n\n LDA Topic Modelling ready to roll!')

