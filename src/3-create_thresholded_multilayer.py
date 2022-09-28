import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle

from pymnet import *
from itertools import combinations

sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})

#Prepare the paths
savepath = "/m/cs/scratch/networks/trianaa1/Paper2/multilayer/results"
file = open(f'{savepath}/FC_jussi_corr','rb')
fc_file = pickle.load(file)
file = open(f'{savepath}/ISFC_jussi_corr','rb')
isfc_file = pickle.load(file)

#Threshold the network
FC_thr = 0.5
ISFC_thr = 0.1

fc_thresholded = np.where(fc_file > FC_thr, 1, 0)
isfc_thresholded = {}
for key in isfc_file.keys():
    isfc_thresholded[key] = np.where(isfc_file[key] > ISFC_thr, 1, 0)

#Defining the ROIs and subjects    
n_rois = 14
n_sub = fc_file.shape[0]
    
rois = list(range(0,n_rois))
subjects = list(range(0,n_sub))
    
brain = MultilayerNetwork(aspects=1)

#links from FC (inter layers)
for sub in subjects:
    data = np.zeros((n_rois,n_rois))
    data[np.triu_indices(n_rois,1)] = fc_thresholded[sub,:]
    ids = tuple(np.argwhere(data==1))
    for layer in ids:
        brain[sub, sub, layer[0], layer[1]] = 1
        
#links from ISFC (including ISC or interlayer)
for subj_pair in isfc_thresholded.keys():
    data = isfc_thresholded[subj_pair]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
    ids = tuple(np.argwhere(data==1))
    sub = [int(x) for x in subj_pair.split('-')]
    for layer in ids:
        brain[sub[0], sub[1], layer[0], layer[1]]
        
pickle.dump(brain, open(f'{savepath}/multilayer_net',"wb"))
    
