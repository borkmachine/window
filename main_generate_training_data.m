% the main script for generating training data

imgdir = '/home/chiller/400coryIms_masked/';    % please change this when running the code - This is the directory where the images you want to process will be 
% remember to create these directories before running this code - Output directories for the images, can set to be whatever you want. I think they need to exist before you start the program
outdir_pos = 'cropped_positiveCory/';
outdir_neg = 'cropped_negativeCory/';
imgs = dir([imgdir 'C*.jpg']); % These have to match the filenames you have for the images and the masks which were pre-generated
masks = dir([imgdir '*mask*png']);
uniqueMaskIms = [];
for i=1:length(masks)
    thing = masks(i);
    uniqueMaskIms = cat(1, uniqueMaskIms, str2double(thing.name(23:28)));
end
uMI = unique(uniqueMaskIms);

ims = randsample(uMI, 40); % This file is currently set up to select 40 random images, if you want to change that it happens here (it would make sense to run this on all images in a given directory, and that is not hard to change…just use a dir(folder) somewhere in there)

%%
%tmpNums = [273 411 703 480 601 543 342 204 654 479];
%tmpNums = [ 479];
%parfor i = 1:size(imgs,1)
parfor i=1:length(ims);
    imgname = ['Camera_110732781_Image' sprintf('%06d',ims(i)) '.jpg'];%
    %imread([imgdir imgname])
    %imgname = imgs(i).name;
    disp(imgname);
    generate_training_data(outdir_pos, outdir_neg, imgname, imgdir);
end
