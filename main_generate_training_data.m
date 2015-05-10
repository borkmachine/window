% the main script for generating training data

imgdir = '';    % please change this when running the code
% remember to create these directories before running this code
outdir_pos = 'cropped_positive/';
outdir_neg = 'cropped_negative/';
imgs = dir(imgdir);

for i = 1:size(imgs,1)
    imgname = imgs(i).name;
    generate_training_data(outdir_pos, outdir_neg, imgname, imgdir);
end