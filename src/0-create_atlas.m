clear all
close all
clc

addpath(genpath('/m/nbe/scratch/braindata/shared/toolboxes/bramila/bramila'));
addpath('/m/nbe/scratch/braindata/shared/toolboxes/NIFTI');

savepath = '/m/nbe/scratch/heps/trianaa1/rois/atlas_set2.nii';
d = dir('/m/nbe/scratch/heps/trianaa1/rois/set2');
d = d(3:21,:);

mask = zeros(91,109,91);

for i=size(d,1):-1:1
    nii = load_nii(sprintf('%s/%s',d(i).folder, d(i).name));
    ids = find(nii.img>0);
    if any(mask(ids)> 0)
        overlap = mask(ids);
        x = find(mask(ids),1,'first');
        fprintf('warning! %s overlaps with %s \n',d(i).name, d(overlap(x)).name)
    end
    mask(ids) = i;
end

nii.img = mask;
nii.hdr.dime.cal_max=20;
nii.hdr.dime.cal_min=0;

save_nii(nii,savepath)

%Now let's make the struct with names 
for i=size(d,1):-1:1
    label{i,1} = d(i).name;
end
cfg.labels = label;
cfg.res = 2;
cfg.roimask=savepath;
rois = bramila_makeRoiStruct(cfg);
save('/m/nbe/scratch/heps/trianaa1/rois/atlas_set2.mat', 'rois');