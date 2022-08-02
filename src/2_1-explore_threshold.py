import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle

sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})

#Prepare the paths
savepath = "/m/nbe/scratch/heps/trianaa1/multilayer/results/"
file = open(f'{savepath}/FC_jussi_corr','rb')
fc_file = pickle.load(file)
file = open(f'{savepath}/ISFC_jussi_corr','rb')
isfc_file = pickle.load(file)

thr = np.linspace(0,1,21)

#FC
fc = fc_file.flatten()

#ISC
isc = []
for key in isfc_file.keys():
    isc = np.append(isc, np.diagonal(isfc_file[key]))

#ISFC
isfc = []
isfc_isc = []
for key in isfc_file.keys():
    isfc = np.append(isfc,isfc_file[key][~np.eye(isfc_file[key].shape[0],dtype=bool)])
    isfc_isc = np.append(isfc_isc,isfc_file[key].flatten())

#No. of links
thresholded = np.zeros((3,21))
for i,t in enumerate(thr):
    thresholded[0,i] = len(fc[fc>t])
    thresholded[1,i] = len(isc[isc>t])
    thresholded[2,i] = len(isfc[isfc>t])

#Histograms and theshold
fig = plt.figure(figsize=(12,10))
gs = gridspec.GridSpec(2, 2)
ax0 = plt.subplot(gs[0,0:2])
ax1 = plt.subplot(gs[1,0])
ax2 = plt.subplot(gs[1,1])

fc_ax = sns.histplot(fc.flatten(), binwidth=0.1, binrange=(0,1), stat='probability', color='tab:blue', alpha=0.5, ax=ax0, label='FC')
isc_ax = sns.histplot(isc.flatten(), binwidth=0.1, binrange=(0,1), stat='probability', color='tab:orange', alpha=0.5, ax=ax0, label='ISC')
ax0 = sns.histplot(isfc.flatten(), binwidth=0.1, binrange=(0,1), stat='probability', color='tab:green', alpha=0.5, ax=ax0, label='ISFC')
ax0.set_xlabel('correlation value')
ax0.set_ylabel('probability')
plt.legend(handles=[fc_ax,isc_ax,ax0])

fc_ax=sns.lineplot(x=thr, y=thresholded[0,:], color='tab:blue', ax=ax1, label="FC")
isc_ax=sns.lineplot(x=thr, y=thresholded[1,:], color='tab:orange', ax=ax1, label="ISC")
ax1 = sns.lineplot(x=thr, y=thresholded[2,:], color='tab:green', ax=ax1, label="ISFC")
ax1.set_xlabel("threshold")
ax1.set_ylabel("Number of links")
plt.legend(handles=[fc_ax,isc_ax,ax1])

sns.lineplot(x=thr, y=thresholded[0,:]/thresholded[0,0], color='tab:blue', ax=ax2)
sns.lineplot(x=thr, y=thresholded[1,:]/thresholded[1,0], color='tab:orange', ax=ax2)
ax2 = sns.lineplot(x=thr, y=thresholded[2,:]/thresholded[2,0], color='tab:green', ax=ax2)
ax2.set_xlabel("threshold")
ax2.set_ylabel("% of links")

plt.show()

# Based on the model of layers=subjects, nodes=ROIs
n_sub = fc_file.shape[0]

intra_layer = np.zeros((n_sub,21))
for i,t in enumerate(thr):
    intra_layer[:,i] = (fc_file > t).sum(axis=1)

inter_layer = np.zeros((n_sub,21))
keys = list(isfc_file.keys())
for sub in range(n_sub):
    sub_isfc = []
    key_selection = [x for x in keys if x.startswith(f'{sub}-')] + [x for x in keys if x.endswith(f'-{sub}')]
    for key in key_selection:
        sub_isfc = np.append(sub_isfc,isfc_file[key].flatten())
        for i,t in enumerate(thr):
            inter_layer[sub,i] = (sub_isfc > t).sum()

fig, (ax0, ax1) = plt.subplots(2, figsize=(16,16))
ax0 = sns.heatmap(intra_layer/91, vmin=0, cmap="hot", cbar=True, xticklabels=False, cbar_kws={'label': '% of links'}, ax=ax0)
ax0.set_ylabel('subject')
ax0.set_title('intra-layer')

ax1 = sns.heatmap(inter_layer/12740, vmin=0, cmap="hot", cbar=True, xticklabels=list(np.around(thr,2)), cbar_kws={'label': '% of links'}, ax=ax1)
ax1.set_xlabel('threshold')
ax1.set_ylabel('subject')
ax1.set_title('inter-layer')

#Applying threshold
FC_thr = 0.5
ISFC_thr = 0.1

fc_thresholded = np.where(fc_file > FC_thr, 1, 0)
isfc_thresholded = {}
for key in isfc_file.keys():
    isfc_thresholded[key] = np.where(isfc_file[key] > ISFC_thr, 1, 0)