%% DATA_PREPROCESSING
% Data is in 3D, but for 2D Unet we need matching 2D images of the input
% and output. Thus, for the inputs we need slices of the CT scan and for
% the output we need the corresponding slice of the manually labeled 3D
% volume.
% This script will generate 2D slices from the 3D volumes. For the input,
% an image with 3 channels is created by adding the previous and next slice
% to the n-th slice. In this way, the model can use information of 3 slices
% to segment the middle slice.For the output, only one channel is needed, 
% i.e. the n-th slice of the labelled volume. 

%% Define input and output paths
input_path_3d_img   = 'C:\Users\u0117721\Documents\PhD\projects\pytorch-3dunet\data\nii';
input_path_3d_msk   = 'C:\Users\u0117721\Documents\PhD\projects\pytorch-3dunet\data\nii';
output_path_2d_img  = 'C:\Users\u0117721\Documents\PhD\projects\Pytorch-UNet\data\imgs_test';
output_path_2d_msk  = 'C:\Users\u0117721\Documents\PhD\projects\Pytorch-UNet\data\masks_test';

sample_names_3d_img = getsubdirs(input_path_3d_img);
n_samples = numel(sample_names_3d_img);

counts = zeros(1,5);
classes = 0:4;
%% Loop over all samples and generate the slices
for s = 1:n_samples
    % Load 3D volumes of the inputs and outputs
    sample_name = sample_names_3d_img{s};
    sample_path_3d_img = fullfile(input_path_3d_img, sample_name, 'sample.nii.gz');
    sample_3d = niftiread(sample_path_3d_img); % Read 3D volume of the inputs

    sample_path_3d_msk = fullfile(input_path_3d_msk, sample_name, 'labels_s.nii.gz');
    mask_3d = niftiread(sample_path_3d_msk); % Read 3D volume of the outputs
    
    % Pad the volumes by one slice to allow adding the previous and next 
    % slice for the first and last layer, respectivley.
    pad_size = 1;               % how many slices to pad
    mask_background_val = 0;    % value to use for the padding of the mask
    sample_3d = padarray(sample_3d, pad_size, 0, 'both');
    mask_3d   = uint8(padarray(mask_3d, pad_size, mask_background_val, 'both'))-1;
    n_slices  = size(sample_3d, 1);
    
    % Loop over all slices
    for n = 2:n_slices-1
       img = cat(3,...
           squeeze(sample_3d(n-1, :, :)),...
           squeeze(sample_3d(n, :, :)),...
           squeeze(sample_3d(n+1, :, :))...
           );
       if length(unique(img)) == 1
           continue
       end
       msk = squeeze(mask_3d(n, :, :));
       
       % add occurence of each label to label count
       counts = counts + hist(double(msk(:)),classes);
       
       % save slices
       file_name = sprintf('%s_%0.3i.png', sample_name, n);
       fprintf('%s \n', file_name)
       img_path  = fullfile(output_path_2d_img, file_name);
       msk_path  = fullfile(output_path_2d_msk, file_name);
       imwrite(img, img_path);
       imwrite(msk, msk_path);
    end
end    

%% Calculate weights
total_count = sum(counts);
weights = 1 - counts/total_count;
disp(weights)


