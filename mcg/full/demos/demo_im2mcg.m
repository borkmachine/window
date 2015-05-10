
%% Demo to show the results of MCG
clear all;close all;home;
im_name = 'gl_001.jpg';
% Read an input image
I = im2double(imread(fullfile(root_dir, 'demos/','trainIms',im_name)));

tic;
% Test the 'fast' version, which takes around 5 seconds in mean
%[candidates_scg, ucm2_scg] = im2mcg(I,'fast');
toc;

tic;
% Test the 'accurate' version, which tackes around 30 seconds in mean
%[candidates_mcg, ucm2_mcg] = im2mcg(I,'accurate');
[candidates_mcg, ucm2_mcg] = im2mcg(I,'fast');

toc;

%% Show UCM results (dilated for visualization)
% figure;
% subplot(1,3,1)
% imshow(I), title('Image')
% 
% %subplot(1,3,2)
% %imshow(imdilate(ucm2_scg,strel(ones(3))),[]), title('Fast UCM (SCG)')
% 
% subplot(1,3,3)
% imshow(imdilate(ucm2_mcg,strel(ones(3))),[]), title('Accurate UCM (MCG)')


%% Show Object Candidates results and bounding boxes
% Candidates in rank position 11 and 12
% id1 = 11; id2 = 12;
% 
% % Get the masks from superpixels and labels
% mask1 = ismember(candidates_mcg.superpixels, candidates_mcg.labels{id1});
% mask2 = ismember(candidates_mcg.superpixels, candidates_mcg.labels{id2});

for id=1:length(candidates_mcg.labels)
    disp(id)
    %masked = I;
    mask = ismember(candidates_mcg.superpixels, candidates_mcg.labels{id});
    mask = double(cat(3,cat(3, mask, mask), mask));
    
    box = candidates_mcg.bboxes(id,:);
    b2 = [box(2) box(1) box(4)-box(2) box(3)-box(1)];

    %masked(ismember(candidates_mcg.superpixels, candidates_mcg.labels{id})==0,:)=0;
    %    imwrite(I.*mask, ['cropped/mask_' num2str(id) '.jpg'])

    imwrite(imcrop(I.*mask, b2), ['/home/chiller/cropped/crop_' num2str(id) '.jpg'])
end
%%
system(['source ~/caffe_prep; export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/oldcuda/targets/x86_64-linux/lib/; python ~/imnet.py ' num2str(length(candidates_mcg.labels)) ' > detectedWindowBlobs.csv']);
%%
windBlobs = csvread('output.csv');
%%
%windBlobs = windBlobs(:,1);
truth = logical(imread([im_name '_combomask.png']));
for i=1:size(windBlobs,1)
    
    id = windBlobs(i);
    mask = ismember(candidates_mcg.superpixels, candidates_mcg.labels{id});
    mask = double(cat(3,cat(3, mask, mask), mask));
    
    box = candidates_mcg.bboxes(id,:);
    b2 = [box(2) box(1) box(4)-box(2) box(3)-box(1)];
    cropped = imcrop(I.*mask, b2);
    px = numel(cropped)/3;
    comp = (mask(:,:,1) == truth);
    percWindow = double(sum(comp(:)))/px;
    if percWindow > .95
        disp(percWindow)
        imshow(cropped)
        k=waitforbuttonpress;
    end
end
% Bboxes is a matrix that contains the four coordinates of the bounding box
% of each candidate in the form [up,left,down,right]. See folder bboxes for
% more function to work with them

% Show results
% figure;
% subplot(1,3,1)
% imshow(I), title('Image')
% subplot(1,3,2)
% imshow(mask1), title('Candidate + Box')
% hold on
% plot([candidates_mcg.bboxes(id1,4) candidates_mcg.bboxes(id1,4) candidates_mcg.bboxes(id1,2) candidates_mcg.bboxes(id1,2) candidates_mcg.bboxes(id1,4)],...
%      [candidates_mcg.bboxes(id1,3) candidates_mcg.bboxes(id1,1) candidates_mcg.bboxes(id1,1) candidates_mcg.bboxes(id1,3) candidates_mcg.bboxes(id1,3)],'r-')
% subplot(1,3,3)
% imshow(mask2), title('Candidate + Box')
% hold on
% plot([candidates_mcg.bboxes(id2,4) candidates_mcg.bboxes(id2,4) candidates_mcg.bboxes(id2,2) candidates_mcg.bboxes(id2,2) candidates_mcg.bboxes(id2,4)],...
%      [candidates_mcg.bboxes(id2,3) candidates_mcg.bboxes(id2,1) candidates_mcg.bboxes(id2,1) candidates_mcg.bboxes(id2,3) candidates_mcg.bboxes(id2,3)],'r-')
