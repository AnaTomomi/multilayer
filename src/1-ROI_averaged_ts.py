import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from nltools.data import Brain_Data, Adjacency

from nilearn.input_data import NiftiLabelsMasker
from nilearn import plotting

#Define the paths
main_path = '/m/nbe/scratch/braindata/jaalho/psykoosi/ensipsykoosi/uusi_data/baseline/epi/Siemens'
control = '/m/nbe/scratch/braindata/jaalho/psykoosi/ensipsykoosi/uusi_data/baseline/epi/Siemens/controls/*/preprocessed_with_Savitzky-Golay/bramila/'
patient = '/m/nbe/scratch/braindata/jaalho/psykoosi/ensipsykoosi/uusi_data/baseline/epi/Siemens/patients/*/preprocessed_with_Savitzky-Golay/bramila/'

save_dir = "/m/nbe/scratch/heps/trianaa1/multilayer/results/ts-neurovault-set2"
metadata_dir = "/m/nbe/scratch/heps/trianaa1/behavioral_data/jussi.xlsx"
atlas = '/m/nbe/scratch/heps/trianaa1/rois/group_roi_mask-neurovault-set2-2mm.nii'

#Prepare the data from the subjects to be read
metadata = pd.read_excel(metadata_dir, sheet_name='Sheet1', header=0, index_col=0, usecols="A,B")
metadata.dropna(inplace=True)
metadata["id"] = metadata.index

#Prepare the Atlas (mask) that will be used to calculate the averaged ROI time-series
masker = NiftiLabelsMasker(labels_img=atlas, standardize=True) #z-scored
plotting.plot_img(atlas)

''' # Run only if you want to check that the signals look fine
for index, row in metadata.head(1).iterrows():
    fmri_file = os.path.join(data_dir,row["scanner"],"FunImgARWCF",row["id"],"Filtered_4DVolume.nii")
    print(fmri_file)
    
    time_series = masker.fit_transform(fmri_file) #Extract the averaged ROI timie-series
    print(time_series.shape)
    plt.figure()
    ax = sns.heatmap(time_series, cmap='RdBu_r', vmin=-5, vmax=5)
    ax.set_xlabel("ROIs")
    ax.set_ylabel("time")'''


for index, row in metadata.iterrows():
    fmri_file = os.path.join(main_path,f'{row["group"]}/{row["id"]}/preprocessed_with_Savitzky-Golay/epi_preprocessed.nii')
    ts_file = os.path.join(save_dir, f'{row["id"]}_neurovault-set2_averaged-ROI-ts.csv')
    
    if os.path.exists(ts_file):
        print(f'Node time series file for {row["id"]} already exists!')
    else:
        print(f'Creating node time series for {row["id"]}')
        time_series = masker.fit_transform(fmri_file)
        pd.DataFrame(time_series).to_csv(ts_file, index=False, header=False)