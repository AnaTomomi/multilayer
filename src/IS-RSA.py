import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from nltools.data import Brain_Data, Adjacency
from nltools.stats import fdr, threshold

from nilearn import plotting
from nilearn import image

from sklearn.metrics import pairwise_distances

from functions import *

#Prepare the paths
metadata_dir = "/m/nbe/scratch/heps/data/data_info.xlsx"
ts_dir = "/m/nbe/scratch/heps/trianaa1/multilayer/results/"
bh_dir = "/m/nbe/scratch/heps/panulaj/prediction/clinical_data/CSV_files/original_clinical_outcome.csv"
atlas="/m/nbe/scratch/heps/trianaa1/masks/group_roi_mask-brainnetome-3mm.nii"

#Prepare the data from the subjects to be read
metadata = prepare_metadata(metadata_dir,'metadata haywarn1')

#Read the averaged ROI time-series data
data = []
for index, row in metadata.iterrows():
    sub_data = []
    sub_data.append(pd.read_csv(os.path.join(ts_dir, 'ts', f'{row["id"]}_brainnetome_averaged-ROI-ts.csv'), header=None))
    sub_data = pd.concat(sub_data)
    data.append(sub_data.values)
data = np.array(data)

n_subs, n_ts, n_nodes = data.shape

#Calculate the similarity matrices
similarity_matrices = []
for node in range(n_nodes):
    similarity_matrices.append(Adjacency(1 - pairwise_distances(data[:, :, node], metric='correlation'), matrix_type='similarity'))
similarity_matrices = Adjacency(similarity_matrices)

# At the end, each instance contains the correlation between all TRs for all subjects (ISC)
similarity_matrices[0].plot(cmap='RdBu_r')

#Behavioral part
metadata["groups"] = 0 #controls
metadata.loc[metadata.group.str.contains('FEP'), 'groups'] = 1 #patients
beh = pairwise_distances(metadata["groups"].to_numpy().reshape(-1, 1), metric='euclidean')

#Finally we start the IS-RSA
isrsa_beh = {}
for node in range(len(similarity_matrices)):
    isrsa_beh[node] = similarity_matrices[node].similarity(beh, metric='spearman', n_permute=1, n_jobs=1 )['correlation']
#For each node, we compute the Spearman correlations between the ISC and the behavioral scores and store them 
#in a dictionary. In the end, we end up with as many correlations as nodes in the parcellations.

#Plotting the results
im2 = roi_to_brain(atlas, isrsa_beh)
plotting.plot_stat_map(im2)
plotting.plot_stat_map(im2, display_mode='z', cut_coords=8, title = "NN", cmap='RdBu_r')

# IS-RSA with hypothesis testing
isrsa_beh_r, isrsa_beh_p = {}, {}
for node in range(len(similarity_matrices)):
    if node==0:
        print("Doing node {} of {}...".format(node+1, len(similarity_matrices)), end =" ")
    else:
        print("{}..".format(node+1), end = " ")
    stats_beh = similarity_matrices[node].similarity(beh, metric='spearman', n_permute=10000, n_jobs=-1 )
    isrsa_beh_r[node] = stats_beh['correlation']
    isrsa_beh_p[node] = stats_beh['p']
    
#So, we have run some non-parametric permutations by shuffling the behavioral matrix in columns and rows simultaneously.
#But, because we also have the distribution, we have the p-values of each happening, so the only thing left to do is...
#Correct for multiple comparisons!

fdr_thr = fdr(pd.Series(isrsa_beh_p).values)
print(f'FDR Threshold: {fdr_thr}')
