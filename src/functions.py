import os
import numpy as np
import pandas as pd

from nilearn import image

def prepare_metadata(metadata_dir,sheet_name):
    #Prepare the data from the subjects to be read
    metadata = pd.read_excel(metadata_dir, sheet_name=sheet_name, header=1, index_col=0, usecols="A,B,G,W")
    metadata.dropna(inplace=True)
    metadata = metadata[metadata['Baseline.1'] != 'N']
    metadata["id"] = metadata.index.str[-3:]
    
    return metadata

def roi_to_brain(atlas, to_convert):
    im = image.load_img(atlas)
    data = image.get_data(im)
    values = np.unique(data)
    replace = dict((k+1,v)for k,v in to_convert.items())

    if values[-1]>len(values):
        print("some ROIs are missing, we'll fix that")
        all_rois = list(range(values[0],values[-1]))
        missing_rois = np.setdiff1d(all_rois,list(values))
    
        for new_roi in missing_rois:
            tail = dict((k+1,v)for k,v in replace.items() if k>=new_roi)
            tail[new_roi] = np.float(0)
            tail = dict(sorted(tail.items()))
            head = dict((k,v)for k,v in replace.items() if k<new_roi)
            replace = {**head, **tail}

    new_data = np.zeros_like(data,dtype=np.float)
    keys = list(replace.keys())
    for key in keys:
        new_data[data==key] = replace[key]

    im2 = image.new_img_like(atlas, new_data)
    
    return im2