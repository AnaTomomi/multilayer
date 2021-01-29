import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

from nltools.data import Adjacency

from nilearn import plotting
from nilearn import image

from sklearn.metrics import pairwise_distances

from functions import *
from pymnet import *

#Prepare the paths
metadata_dir = "/m/nbe/scratch/heps/data/data_info.xlsx"
ts_dir = "/m/nbe/scratch/heps/trianaa1/multilayer/results/"
bh_dir = "/m/nbe/scratch/heps/panulaj/prediction/clinical_data/CSV_files/original_clinical_outcome.csv"
atlas="/m/cs/scratch/networks/trianaa1/Atlas/Brainnetome/Brainnetome/BN_Atlas_246_3mm.nii.gz"

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

#Calculate the ISC similarity matrices
ISC_sim = []
for node in range(n_nodes):
    ISC_sim.append(Adjacency(1 - pairwise_distances(data[:, :, node], metric='correlation'), matrix_type='similarity'))
ISC_sim = Adjacency(ISC_sim)

pickle.dump(ISC_sim, open("/m/nbe/scratch/heps/trianaa1/multilayer/results/ISC_sim","wb"))

#Calculate the FC similarity matrices
FC_sim = []
for subject in range(n_subs):
    FC_sim.append(Adjacency(1 - pairwise_distances(data[subject, :, :], metric='correlation'), matrix_type='similarity'))
FC_sim = Adjacency(FC_sim)

pickle.dump(FC_sim, open("/m/nbe/scratch/heps/trianaa1/multilayer/results/FC_sim","wb"))