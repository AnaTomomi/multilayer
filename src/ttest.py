import os
import numpy as np
import pandas as pd

from brainiak.isc import isc

from scipy.spatial.distance import squareform
from scipy.stats import ttest_ind, rankdata

from nltools.data import Brain_Data
from nltools.mask import roi_to_brain, expand_mask
from nltools.stats import fdr, threshold

from nilearn.plotting import plot_glass_brain

#Prepare the paths
ts_dir = "/m/nbe/scratch/heps/trianaa1/multilayer/results/ts-jussi"
figpath = "/m/nbe/scratch/heps/trianaa1/multilayer/results/figures"
metadata = pd.read_excel("/m/nbe/scratch/heps/trianaa1/behavioral_data/jussi.xlsx")
mask = Brain_Data('/m/nbe/scratch/heps/trianaa1/masks/group_roi_mask-brainnetome-2mm.nii')

data = []
for index, row in metadata.iterrows():
    sub_data = []
    sub_data.append(pd.read_csv(os.path.join(ts_dir, f'{row["subject"]}_brainnetome_averaged-ROI-ts.csv'), header=None))
    sub_data = pd.concat(sub_data)
    data.append(sub_data.values)
    print(row["subject"])

isc_output = isc(data, pairwise=True, summary_statistic=None, tolerate_nans=True)

#Take indices
metadata["bin"] = 0
metadata.loc[metadata.group.str.contains('patient'), 'bin'] = 1 
groups = metadata[["bin"]].to_numpy()
pat_ids = np.where(groups == 1)[0]
con_ids = np.where(groups == 0)[0]

t, p ={}, {}
for node in range(isc_output.shape[1]):
    mat = squareform(isc_output[:,node])
    pat = mat[pat_ids[0]:pat_ids[-1]+1, pat_ids[0]:pat_ids[-1]+1]
    pat = np.tril(pat,-1).flatten()
    con = mat[con_ids[0]:con_ids[-1]+1, con_ids[0]:con_ids[-1]+1]
    con = np.tril(con,-1).flatten()
    t[node], p[node] = ttest_ind(pat,con,equal_var=False)

fdr_thr = fdr(pd.Series(p).values)
bonferroni = 0.05/len(p)
print(f'FDR Threshold: {fdr_thr} \n Bonferroni Threshold: {bonferroni}')

t_brain = roi_to_brain(pd.Series(t), expand_mask(mask))
p_brain = roi_to_brain(pd.Series(p), expand_mask(mask))

plot_glass_brain(threshold(t_brain, p_brain, thr=bonferroni).to_nifti(), colorbar=True)