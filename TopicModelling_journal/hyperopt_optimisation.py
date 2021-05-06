# This Source Code Form is subject to the terms of the Mozilla Public ---------------------
# License, v. 2.0. If a copy of the MPL was not distributed with this ---------------------
# file, You can obtain one at http://mozilla.org/MPL/2.0/. */ -----------------------------
# ---------------- Copyright (C) 2021 University of Strathclyde and Author ----------------
# -------------------------------- Author: Audrey Berquand --------------------------------
# ------------------------- e-mail: audrey.berquand@strath.ac.uk --------------------------

'''
Define the optimised weights for the weighted sum of pre-trained models
(method compared to JS Divergence in paper)
'''

import sys
import pickle
import time
import numpy as np
from TopicModelling_journal.categorisation import *
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, plotting
from hyperopt.plotting import main_plot_vars
from hyperopt import base
sys.path.append('../')

'''---------------------------
---------- FUNCTIONS ---------
---------------------------'''
def objFct(hyper_pmt):
    ''' Define objective function to optimise '''

    start = time.time()
    print('Input params:', hyper_pmt)

    semi=False
    update=False

    # Semisupervised models
    # Format = [model_name, semi, update, weight, model folder]
    modelsList_Semi = [['wiki_22_10', True, False, hyper_pmt[0], 'wiki_semisupervised'], ['wiki_22_11', True, False, hyper_pmt[0], 'wiki_semisupervised'],
                  ['wiki_22_12', True, False, hyper_pmt[0], 'wiki_semisupervised'], ['wiki_22_13', True, False, hyper_pmt[0], 'wiki_semisupervised'],
                  ['wiki_22_14', True, False, hyper_pmt[0], 'wiki_semisupervised'], ['wiki_22_15', True, False, hyper_pmt[0], 'wiki_semisupervised'],
                  ['wiki_22_16', True, False, hyper_pmt[0], 'wiki_semisupervised'], ['wiki_22_17', True, False, hyper_pmt[0], 'wiki_semisupervised'],
                  ['wiki_22_18', True, False, hyper_pmt[0], 'wiki_semisupervised'], ['wiki_22_19', True, False, hyper_pmt[0], 'wiki_semisupervised'],
                  ['model_30_3', True, False, hyper_pmt[1], 'reports_semisupervised'], ['reports_30_31', True, False, hyper_pmt[1], 'reports_semisupervised'],
                  ['reports_30_32', True, False, hyper_pmt[1], 'reports_semisupervised'], ['reports_30_33', True, False, hyper_pmt[1], 'reports_semisupervised'],
                  ['reports_30_34', True, False, hyper_pmt[1], 'reports_semisupervised'], ['reports_30_35', True, False, hyper_pmt[1], 'reports_semisupervised'],
                  ['reports_30_36', True, False, hyper_pmt[1], 'reports_semisupervised'], ['reports_30_37', True, False, hyper_pmt[1], 'reports_semisupervised'],
                  ['reports_30_38', True, False, hyper_pmt[1], 'reports_semisupervised'], ['reports_30_39', True, False, hyper_pmt[1], 'reports_semisupervised'],
                  ['model_24_41', True, False, hyper_pmt[2], 'books_semisupervised'], ['books_24_2', True, False, hyper_pmt[2], 'books_semisupervised'],
                  ['books_24_3', True, False, hyper_pmt[2], 'books_semisupervised'], ['books_24_4', True, False, hyper_pmt[2], 'books_semisupervised'],
                  ['books_24_5', True, False, hyper_pmt[2], 'books_semisupervised'], ['books_24_6', True, False, hyper_pmt[2], 'books_semisupervised'],
                  ['books_24_7', True, False, hyper_pmt[2], 'books_semisupervised'], ['books_24_8', True, False, hyper_pmt[2], 'books_semisupervised'],
                  ['books_24_9', True, False, hyper_pmt[2], 'books_semisupervised'], ['books_24_10', True, False, hyper_pmt[2], 'books_semisupervised']]

    # Unsupervised models
    # Format = [model_name, semi, update, weight, model folder]
    modelsList_Unsup = [['wiki_22_10', semi, update, hyper_pmt[0], 'wiki_unsupervised'], ['wiki_22_11', semi, update, hyper_pmt[0], 'wiki_unsupervised'],
                  ['wiki_22_12', semi, update, hyper_pmt[0], 'wiki_unsupervised'],
                  ['wiki_22_13', semi, update, hyper_pmt[0], 'wiki_unsupervised'],
                  ['wiki_22_14', semi, update, hyper_pmt[0], 'wiki_unsupervised'],
                  ['wiki_22_15', semi, update, hyper_pmt[0], 'wiki_unsupervised'],
                  ['wiki_22_16', semi, update, hyper_pmt[0], 'wiki_unsupervised'],
                  ['wiki_22_17', semi, update, hyper_pmt[0], 'wiki_unsupervised'],
                  ['wiki_22_18', semi, update, hyper_pmt[0], 'wiki_unsupervised'],
                  ['wiki_22_19', semi, update, hyper_pmt[0], 'wiki_unsupervised'],
                  ['model_30_21', semi, update, hyper_pmt[1], 'reports_unsupervised'],
                  ['reports_30_22', semi, update, hyper_pmt[1], 'reports_unsupervised'],
                  ['reports_30_23', semi, update, hyper_pmt[1], 'reports_unsupervised'],
                  ['reports_30_24', semi, update, hyper_pmt[1], 'reports_unsupervised'],
                  ['reports_30_25', semi, update, hyper_pmt[1], 'reports_unsupervised'],
                  ['reports_30_26', semi, update, hyper_pmt[1], 'reports_unsupervised'],
                  ['reports_30_27', semi, update, hyper_pmt[1], 'reports_unsupervised'],
                  ['reports_30_28', semi, update, hyper_pmt[1], 'reports_unsupervised'],
                  ['reports_30_29', semi, update, hyper_pmt[1], 'reports_unsupervised'],
                  ['reports_30_30', semi, update, hyper_pmt[1], 'reports_unsupervised'],
                  ['books_24_34', semi, update, hyper_pmt[2], 'books_unsupervised'],
                  ['books_24_2', semi, update, hyper_pmt[2], 'books_unsupervised'],
                  ['books_24_3', semi, update, hyper_pmt[2], 'books_unsupervised'],
                  ['books_24_4', semi, update, hyper_pmt[2], 'books_unsupervised'],
                  ['books_24_5', semi, update, hyper_pmt[2], 'books_unsupervised'],
                  ['books_24_6', semi, update, hyper_pmt[2], 'books_unsupervised'],
                  ['books_24_7', semi, update, hyper_pmt[2], 'books_unsupervised'],
                  ['books_24_8', semi, update, hyper_pmt[2], 'books_unsupervised'],
                  ['books_24_9', semi, update, hyper_pmt[2], 'books_unsupervised'],
                  ['books_24_10', semi, update, hyper_pmt[2], 'books_unsupervised']]

    categoriesList = ['AOCS', 'com', 'environment', 'OBDH', 'Power', 'prop', 'thermal']
    scores, precisionScores, mrrScores = combinedCategorisation(modelsList_Unsup, categoriesList)
    area = np.trapz(precisionScores)
    print('Area: ', area)

    #print('Computation Time:', round((time.time() - start), 2), 'sec')
    return {'loss': -area, 'status': STATUS_OK}

def main():
    ''' Run Optimisation'''

    # Find the optimised weights combination to balance the wiki, book and reports corpora
    trials = Trials()
    space=[hp.uniform('w1', 0, 1), hp.uniform('w2', 0, 1), hp.uniform('w3', 0, 1)]
    global param_list
    param_list = ('w1', 'w2', 'w3')
    best = fmin(fn=objFct, space=space, algo=tpe.suggest, max_evals=100, trials=trials)

    # Output
    print("Best weights combination:", best)
    print("trials summary:")
    for trial in trials.trials:
        print(trial)

    # Save trials
    filename = 'TM_unsupervised_Opt.pkl'
    pickle.dump(trials, open(filename, "wb"))

    # Plot
    #domain = base.Domain(objFct, space)
    #main_plot_vars(trials, bandit=domain)
    plotting.main_plot_history(trials)
    plotting.main_plot_histogram(trials)

    return

'''---------------------------
------------ MAIN ------------
---------------------------'''
if __name__ == "__main__":
    main()






