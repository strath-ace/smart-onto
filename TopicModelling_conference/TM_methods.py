# This Source Code Form is subject to the terms of the Mozilla Public ---------------------
# License, v. 2.0. If a copy of the MPL was not distributed with this ---------------------
# file, You can obtain one at http://mozilla.org/MPL/2.0/. */ -----------------------------
# ---------------- Copyright (C) 2019 University of Strathclyde and Author ----------------
# -------------------------------- Author: Audrey Berquand --------------------------------
# ------------------------- e-mail: audrey.berquand@strath.ac.uk --------------------------

import os
from os import listdir
from os.path import isfile, join

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

