import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from itertools import combinations

from nltools.data import Adjacency
from nltools.stats import isfc
from nilearn.connectome import ConnectivityMeasure

from nilearn import plotting
from nilearn import image

from sklearn.metrics import pairwise_distances

from functions import *
from pymnet import *

#Prepare the paths
metadata_dir = "/m/nbe/scratch/heps/trianaa1/behavioral_data/jussi.xlsx"
ts_dir = "/m/nbe/scratch/heps/trianaa1/multilayer/results/ts-neurovault-set1/"
atlas="/m/nbe/scratch/heps/trianaa1/rois/group_roi_mask-neurovault-set1-2mm.nii"
savepath = "/m/nbe/scratch/heps/trianaa1/multilayer/results/"

#Prepare the data from the subjects to be read
metadata = pd.read_excel(metadata_dir, sheet_name='Sheet1', header=0, index_col=0, usecols="A,B")
metadata.dropna(inplace=True)
metadata["id"] = metadata.index

#Read the averaged ROI time-series data
data = []
for index, row in metadata.iterrows():
    sub_data = []
    sub_data.append(pd.read_csv(os.path.join(ts_dir, f'{row["id"]}_neurovault-set1_averaged-ROI-ts.csv'), header=None))
    sub_data = pd.concat(sub_data)
    data.append(sub_data.values)
data = np.array(data)

n_subs, n_ts, n_nodes = data.shape


#Calculate the FC correlation matrices
correlation_measure = ConnectivityMeasure(kind='correlation')

FC_corr = []
for subject in range(n_subs):
    sub_data = data[subject,:,:]
    correlation_matrix = correlation_measure.fit_transform([sub_data])[0]
    FC_corr.append(correlation_matrix[np.triu_indices(n_nodes,1)])
FC_corr = np.array(FC_corr)
    
pickle.dump(FC_corr, open(f'{savepath}/FC_jussi_corr',"wb"))

#Calculate the ISFC similarity matrices (includes the ISC)
subjects = list(range(0,n_subs))
ids = list(combinations(subjects, r=2))
isfc_output = {}
pair_data = []
for pair in ids:
    pair_data.append(pd.DataFrame.from_records(data[pair[0],:,:]))
    pair_data.append(pd.DataFrame.from_records(data[pair[1],:,:]))
    isfc_output[f'{pair[0]}-{pair[1]}'] = isfc(pair_data)[0]
    pair_data.clear()
pickle.dump(isfc_output, open(f'{savepath}/ISFC_jussi_corr',"wb"))

#Construct the multilayer network
#brain = brain_to_multilayer(ISC_sim, FC_sim)
#pickle.dump(brain, open("/m/nbe/scratch/heps/trianaa1/multilayer/results/multilayer_net","wb"))