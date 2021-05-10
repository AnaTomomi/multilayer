import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

from brainiak.isc import isc

from nilearn.plotting import plot_glass_brain, plot_stat_map, view_img, view_img_on_surf

from nltools.data import Adjacency, Brain_Data
from nltools.mask import roi_to_brain, expand_mask
from nltools.stats import fdr, threshold

from sklearn.manifold import TSNE
from sklearn.impute import KNNImputer
from sklearn.metrics import pairwise_distances

from scipy.spatial.distance import squareform
from scipy.stats import ttest_ind, rankdata

from functions import sort_square_mtx, scale_mtx

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

# Curate the behavioral data
data = metadata.set_index(metadata.subject)
data.drop(['group','Sex (1=male)', 'Age BL', 'Age FU','CPZeq_BL', 'CPZeq_FU','subject','FD', 'Realism FU', 'Ident_FU','BPRS10+11', 'BPRS11_FU'], axis=1, inplace=True)
names = list(data.columns)
data = data.to_numpy()
imputer = KNNImputer(n_neighbors=5, weights="distance",add_indicator=False)
data = imputer.fit_transform(data)
data = np.around(data,0)

#plot the ISC matrix to see how it looks
'''for roi in range(243):
    to_plot = squareform(isc_output[:,roi])
    v = max(abs(round(np.amin(to_plot),1)),round(np.amax(to_plot),1))
    corr = round(np.mean(np.corrcoef(to_plot, metadata["FD"], rowvar=False)[-1, :-1]),3)
    isc_embedded = TSNE(n_components=2, metric="precomputed").fit_transform(1 - to_plot) 

    cmap='RdYlBu_r' #'RdBu_r'
    fig = plt.figure(figsize=(12, 9))
    sns.set_style("darkgrid")
    ax = fig.add_subplot(331)
    ax = sns.heatmap(to_plot, cmap=cmap, vmin=-v, vmax=v, ax=ax, square=True)
    ax.set_xlabel("subjects")
    ax.set_ylabel("subjects")
    ax.set_title(f'ISC, corr with FD:{corr}')  

    ax = fig.add_subplot(332)
    ax = sns.scatterplot(x=isc_embedded[:,0], y=isc_embedded[:,1], hue=metadata["group"], cmap=cmap, ax=ax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('TSNE by group')
    
    labels = ['realism', 'identity', 'hallucination', 'delusions', 'functioning', '- symp.','+ symp.', 'ToM']
    for i in range(3,10):
        ax = fig.add_subplot(3,3,i)
        ax = sns.scatterplot(x=isc_embedded[:,0], y=isc_embedded[:,1], hue=data[:,i-3], palette=cmap, ax=ax)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f'TSNE by {labels[i-3]}')
        
        ax.get_legend().remove()
        norm = plt.Normalize(data[:,i-3].min(), data[:,i-3].max())
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        ax.figure.colorbar(sm)

    fig.suptitle(f'ROI {str(roi+1)}')
    fig.tight_layout(pad=1.0)
    fig.savefig(f'{figpath}/{roi}_ISC.png')
    plt.close(fig)

#ISC-RSA
metadata["bin"] = 0
metadata.loc[metadata.group.str.contains('patient'), 'bin'] = 1 
beh = pairwise_distances(metadata["bin"].to_numpy().reshape(-1, 1), metric='euclidean')
beh = 1-beh #group similarity matrix "1" are exactly the same, "0" totally different'''

n_subs = len(metadata)
for feat in range(data.shape[1]):
    print(f'Computing IS-RSA for feature {names[feat]}')
    behav = data[:,feat]
    behav_rank = rankdata(behav)
    
    beh_nn = Adjacency(pairwise_distances(np.reshape(behav_rank, (-1, 1)), metric='euclidean'), matrix_type='distance')
    beh_nn = beh_nn.distance_to_similarity()
    beh_annak = np.zeros((n_subs, n_subs))
    for i in range(n_subs):
        for j in range(n_subs):
            if i < j:
                sim_ij = np.mean([behav_rank[i], behav_rank[j]])/n_subs
                beh_annak[i,j] = sim_ij
                beh_annak[j,i] = sim_ij
            elif i==j:
                beh_annak[i,j] = 1
    beh_annak = Adjacency(beh_annak, matrix_type='similarity')

    isrsa_nn_r, isrsa_nn_p = {}, {}
    isrsa_annak_r, isrsa_annak_p = {}, {}
    for node in range(isc_output.shape[1]):
        print("{}..".format(node), end = " ")
        isc_mat = Adjacency(squareform(isc_output[:,node]), matrix_type='similarity')
        stats_nn = isc_mat.similarity(beh_nn, metric='spearman', n_permute=5000, n_jobs=-1 )
        isrsa_nn_r[node] = stats_nn['correlation']
        isrsa_nn_p[node] = stats_nn['p']
    
        stats_annak = isc_mat.similarity(beh_annak, metric='spearman', n_permute=5000, n_jobs=-1 )
        isrsa_annak_r[node] = stats_annak['correlation']
        isrsa_annak_p[node] = stats_annak['p']
    
    with open(os.path.join(figpath,f'nn_{names[feat]}_r'), 'wb') as handle:
        pickle.dump(isrsa_nn_r, handle)
    with open(os.path.join(figpath,f'nn_{names[feat]}_p'), 'wb') as handle:
        pickle.dump(isrsa_nn_p, handle)
    with open(os.path.join(figpath,f'ak_{names[feat]}_r'), 'wb') as handle:
        pickle.dump(isrsa_annak_r, handle)
    with open(os.path.join(figpath,f'ak_{names[feat]}_p'), 'wb') as handle:
        pickle.dump(isrsa_annak_p, handle)
    
    fdr_thr = fdr(pd.Series(isrsa_nn_p).values)
    print(f'FDR Threshold NN: {fdr_thr}')
    
    fdr_thr = fdr(pd.Series(isrsa_annak_p).values)
    print(f'FDR Threshold AK: {fdr_thr}')

#Check the only significant result
for feat in range(data.shape[1]):
    print(f'{names[feat]} ...............')
    with open(os.path.join(figpath,f'nn_{names[feat]}_r'), 'rb') as handle:
        nn_r = pickle.load(handle)
    with open(os.path.join(figpath,f'nn_{names[feat]}_p'), 'rb') as handle:
        nn_p = pickle.load(handle)
    fdr_thr = fdr(pd.Series(nn_p).values)
    bonferroni = [i for i in list(nn_p.values()) if i <= (0.05/len(nn_p))]
    print(f'FDR Threshold NN: {fdr_thr} \n # Bonferroni NN: {len(bonferroni)}')
    
    with open(os.path.join(figpath,f'ak_{names[feat]}_r'), 'rb') as handle:
        ak_r = pickle.load(handle)
    with open(os.path.join(figpath,f'ak_{names[feat]}_p'), 'rb') as handle:
        ak_p = pickle.load(handle)
    fdr_thr = fdr(pd.Series(ak_p).values)
    bonferroni = [i for i in list(ak_p.values()) if i <= (0.05/len(ak_p))]
    print(f'FDR Threshold AK: {fdr_thr} \n # Bonferroni AK: {len(bonferroni)}')

#Plot the significant result
feat=2
print(f'{names[feat]} ...............')
with open(os.path.join(figpath,f'nn_{names[feat]}_r'), 'rb') as handle:
    nn_r = pickle.load(handle)
with open(os.path.join(figpath,f'nn_{names[feat]}_p'), 'rb') as handle:
    nn_p = pickle.load(handle)
fdr_thr = fdr(pd.Series(nn_p).values)
bonferroni = 0.05/len(nn_p)

nn_r_brain = roi_to_brain(pd.Series(nn_r), expand_mask(mask))
nn_p_brain = roi_to_brain(pd.Series(nn_p), expand_mask(mask))

plot_glass_brain(threshold(nn_r_brain, nn_p_brain, thr=bonferroni).to_nifti())