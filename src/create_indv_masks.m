%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script generates brain masks for the DPARSF A software.            %
% WARNING! The images contain skull and membrane tissues which will be    %
% reflected in the masks. The script uses images that have been slice time%
% corrected, realigned, and normalized.                                   %
%                                                                         %
% Author: ana.trianahoyos@aalto.fi                                        %
% Created: 20.01.2021                                                     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all
close all
clc

addpath('/m/nbe/scratch/braindata/shared/toolboxes/NIFTI');
addpath(genpath('/m/nbe/scratch/braindata/shared/toolboxes/bramila/bramila'));

%Read the paths
d=dir('/m/nbe/scratch/heps/haywarn1/Preprocessing/eregHEPS/pp/baseline/*/FunImgARW/');
d(ismember({d.name}, {'.', '..','.DS_Store','._.DS_Store'})) = [];
output_dir='/m/nbe/scratch/heps/trianaa1/masks';
template = '/m/cs/scratch/networks/trianaa1/Atlas/Brainnetome/Brainnetome/BN_Atlas_246_3mm.nii';

for i=1:size(d,1)
    subdir=dir(sprintf('%s/%s',d(i).folder,d(i).name));
    subdir(ismember({subdir.name}, {'.', '..'})) = [];  
    
    cfg.infile=sprintf('%s/%s',subdir.folder,subdir.name);
    cfg.outpath=sprintf('%s/%s/mask.nii',output_dir,d(i).name);
    
    if (isfield(cfg,'infile'))
        nii=load_nii(cfg.infile);
        data=nii.img;
    else 
        fprintf('ERROR: no input file \n');
    end

    kk=size(data);
    T=kk(4);
    
    fprintf('Computing EPI mask...');

    mask =  ones(kk(1:3));
    for t=1:T  
        temp=squeeze(data(:,:,:,t));
        mask=mask.*(temp>0.1*quantile(temp(:),.98));
    end

    fprintf(' done\n');

    fprintf('...EPI mask size %i voxels\n',nnz(mask)); 
    
    if nnz(mask)==0
       warning('Empty EPI mask!');
    end

    if nnz(mask)>0.5*numel(mask)
       warning('EPI mask has over 50% of total FOV volume!');
    end
    
    ref = load_nii(template); 
    ref.img = double(mask);
    ref.img = double(mask);
    ref.hdr.hist.descrip='EPI mask';
    ref.hdr.dime.cal_max=1;
    ref.hdr.dime.cal_min=0;
    siz = size(mask);
    if length(siz)==4
        ref.hdr.dime.dim(1)=4;
        ref.hdr.dime.dim(5)=siz(4);
    end
    if ~isfolder(sprintf('%s/%s/',output_dir,d(i).name))
        mkdir(sprintf('%s/%s/',output_dir,d(i).name));
    end
        
    save_nii(ref,cfg.outpath);
    fprintf('%s done! \n',d(i).name)
    
    clear cfg
end