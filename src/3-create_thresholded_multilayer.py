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

#Defining the ROIs and subjects    
n_rois = 14
n_sub = fc_file.shape[0]

#Threshold the network by density
FC_thr = 0.2
ISFC_thr = 0.2

no_links_keep = round(fc_file.shape[1]*FC_thr)
fc_thresholded = fc_file
for row in range(n_sub):
    sub_thr = np.argsort(fc_file[row,:])[-no_links_keep:][0]
    fc_thresholded[row,:] = np.where(fc_thresholded[row,:] >= fc_thresholded[row,sub_thr], 1, 0)
    
no_links_keep = round(n_rois*n_rois*ISFC_thr)
isfc_thresholded = {}
for key in isfc_file.keys():
    pair_thr = -np.sort(-isfc_file[key].flatten())[0:no_links_keep][-1]
    isfc_thresholded[key] = np.where(isfc_file[key] >= pair_thr, 1, 0)

'''#Threshold by absolute threshold
fc_thresholded = np.where(fc_file > FC_thr, 1, 0)
isfc_thresholded = {}
for key in isfc_file.keys():
    isfc_thresholded[key] = np.where(isfc_file[key] > ISFC_thr, 1, 0)'''

#Start the multilayer    
rois = list(range(0,n_rois))
subjects = list(range(0,n_sub))
    
brain = MultilayerNetwork(aspects=1)

#Model 1
#links from FC (intra layers)
for sub in subjects:
    data = np.zeros((n_rois,n_rois))
    data[np.triu_indices(n_rois,1)] = fc_thresholded[sub,:]
    ids = tuple(np.argwhere(data==1))
    for node in ids:
        brain[node[0], node[1], sub, sub] = 1
        
#links from ISFC (interlayer)
for subj_pair in isfc_thresholded.keys():
    data = isfc_thresholded[subj_pair]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
    ids = tuple(np.argwhere(data==1))
    sub = [int(x) for x in subj_pair.split('-')]
    for node in ids:
        brain[node[0], node[1], sub[0], sub[1]] = 1
        
pickle.dump(brain, open(f'{savepath}/multilayer_net_model1_densitythr_fc{str(FC_thr)}_isfc{str(ISFC_thr)}',"wb"))

'''#Model 2
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
        brain[sub[0], sub[1], layer[0], layer[1]] = 1
        
pickle.dump(brain, open(f'{savepath}/multilayer_net_model2_absolutethr_fc{str(FC_thr)_isfc{str(FC_thr)',"wb"))
'''    
