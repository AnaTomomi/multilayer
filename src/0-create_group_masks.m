%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script generates group masks and atlas group masks. The script only%
% uses the brainnetome atlas for now.                                     %
%                                                                         %
% Author: ana.trianahoyos@aalto.fi                                        %
% Created: 21.01.2021                                                     %
% Modified: 28.01.2021 Add fill function so that the group mask has no    %
%                      holes.                                             %
% Modified: 29.04.2021 Add new paths to work with Jussi's prepro data     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

close all 
clear all
clc

addpath(genpath('/m/nbe/scratch/braindata/shared/toolboxes/bramila/bramila'));
addpath('/m/nbe/scratch/braindata/shared/toolboxes/NIFTI');
addpath(genpath('/scratch/cs/networks/trianaa1/Paper1/smoothing-group'));

%d=dir('/m/nbe/scratch/heps/trianaa1/masks/*/');
%d(ismember({d.name}, {'.', '..','.DS_Store','._.DS_Store'})) = [];

d1=dir('/m/nbe/scratch/braindata/jaalho/psykoosi/ensipsykoosi/uusi_data/baseline/epi/Siemens/controls/*/preprocessed_with_Savitzky-Golay/bramila/');
d2=dir('/m/nbe/scratch/braindata/jaalho/psykoosi/ensipsykoosi/uusi_data/baseline/epi/Siemens/patients/*/preprocessed_with_Savitzky-Golay/bramila/');
d = [d1; d2];
d(~ismember({d.name}, {'epi_STD_analysis_mask.nii'}))=[];

beh = readcell('/m/nbe/scratch/heps/trianaa1/behavioral_data/jussi.xlsx');
beh = beh(2:end,1);
count = 1;
for i=1:length(d)
    sub = split(d(i).folder,'/');
    sub = sub{13};
    if isempty(cell2mat(strfind(beh, sub)))
        ids(count) = i;
        count = count+1;
    end
end
d(ids)=[];

group_folder_out = '/m/nbe/scratch/heps/trianaa1/rois/set1';

group_mask = [];

res = 2;
res_str = [num2str(res), 'mm'];
start_from_epi = 1; % set to 1 if you want to read individual ep masks, 0 if only create .mat files
atlas = 'neurovault-set2'; % which atlas to use for producing the group masks, options: 'aal', 'ho', 'brainnetome'
roi_number = '19'; %change to match the number of ROIs used in the parcellation
template = ['/m/cs/scratch/networks/trianaa1/Atlas/MNI_templates/MNI152_T1_' res_str '.nii'];
missing_rois = []; %Missing rois in the original mask
%% reading individual masks, creating group mask

nii_path = [group_folder_out '/groupmask-' atlas '-' res_str '.nii'];

if start_from_epi
    for s = 1:length(d)
        disp(s)
        subject=sprintf('%s/%s',d(s).folder,d(s).name);
        ind_mask = load_nii(subject);
        if s == 1
            group_mask = ind_mask.img;
        else
            group_mask = group_mask .* ind_mask.img;
        end
    end

    save_nii(make_nii(group_mask, [res res res]), nii_path)

% origin fix
    clear cfg
    cfg.target = nii_path;
    cfg.template = template;
    cfg = my_correct_origin(cfg);
else
    group_mask = load_nii(nii_path);
    group_mask = group_mask.img;
end

epi_size = size(group_mask);

%% roi masks

if strcmp(atlas,'brainnetome')
    %roi_mask = load_nii('/m/cs/scratch/networks/trianaa1/Atlas/Brainnetome/Brainnetome/BN_Atlas_246_3mm.nii');
    roi_mask = load_nii('/m/cs/scratch/networks/trianaa1/Atlas/Brainnetome/Brainnetome/BNA-maxprob-thr0-2mm.nii');
elseif strcmp(atlas,'neurovault-set1')
    roi_mask = load_nii('/m/nbe/scratch/heps/trianaa1/rois/atlas_set1.nii');
elseif strcmp(atlas,'neurovault-set2')
    roi_mask = load_nii('/m/nbe/scratch/heps/trianaa1/rois/atlas_set2.nii');
end

roi_mask = roi_mask.img;

roi_mask_size = size(roi_mask);

if any(roi_mask_size < epi_size) % filling roi masks with 0's if epi < roi mask
    roi_mask(roi_mask_size(1) + 1, :, :) = 0;
    roi_mask(:, roi_mask_size(2) + 1, :) = 0;
    roi_mask(:, :, roi_mask_size(3) + 1) = 0;
end

group_roi = roi_mask .* group_mask;

unique_roi_indices = unique(group_roi);
unique_roi_indices = unique_roi_indices(2:end); 
n_rois = max(size(unique_roi_indices));

% Removing "gaps" caused by ROIs that are not present at the given
% probability; needed for picking correct labels later on.
for i = 1:n_rois
    for j = 1:length(missing_rois)
        if unique_roi_indices(i) > missing_rois(j)
            unique_roi_indices(i) = unique_roi_indices(i) - 1;
        end
    end
end

group_roi_path = [group_folder_out '/group_roi_mask-' atlas '-' res_str '.nii'];

save_nii(make_nii(group_roi, [res res res]), group_roi_path);

% origin fix

clear cfg
cfg.target = group_roi_path; 
cfg.template = template;
cfg = my_correct_origin(cfg);

clear cfg
cfg.imgsize = roi_mask_size;
cfg.res = res;
cfg.roimask = group_roi_path; 

% getting correct labels
if strcmp(atlas,'brainnetome')
    mask_name = '/m/cs/scratch/networks/trianaa1/Atlas/brainnetome_MPM_rois_2mm.mat'; %this is only used to get the labels, no use of the maps or centroids.
elseif strcmp(atlas,'neurovault-set1')
    mask_name = '/m/nbe/scratch/heps/trianaa1/rois/atlas_set1.mat'; 
elseif strcmp(atlas,'neurovault-set2')
    mask_name = '/m/nbe/scratch/heps/trianaa1/rois/atlas_set2.mat'; 
end

load(mask_name);
labels = cell(n_rois, 1); 
for i = 1:n_rois
    labels(i) = {rois(unique_roi_indices(i)).label}; %
end

cfg.labels=labels;
rois = my_bramila_makeRoiStruct(cfg);

mask_path = [group_folder_out '/group_roi_mask-' atlas '-' res_str];  
    
save(mask_path, 'rois');
