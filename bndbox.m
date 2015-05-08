%% README
% There are six datasets that need labeling, 
% to select which dataset to proceed, change the following:
dataset_index = 1; % range from 1 to 6
% and then run the code 

% each dataset contains ~1000-2000 photos,
% the program will pause while displaying each photo
% select the best matching region for each window you see in this photo
% (make sure that you do not include any region outside

% if there is no window, or you are done selecting
% click the grey area (outside) to the _left_ side of the image
% the program will then fetch you the next image 

% if you make a mistake while selecting the region, don't worry
% click the grey area (outside) to the _right_ side of the image 
% _all_ regions you selected for the _current_ image will be deleted
% then you can start over

% the program stores the data every time you finish one image
% your hard work won't be missed!!

%% do not attempt to change the following code
path_info = {...
    {'/media/DATA1/plamen/mulford_20150204/20150204-3/right/rightCrop/', 110732692, 2059}, ...
    {'/media/DATA1/plamen/mulford_20150204/20150204-3/left/leftCrop/', 110732781, 2059}, ...
    {'/media/DATA1/plamen/mulford_20150204/20150204-2/right/rightCrop/', 110732692, 3445}, ...
    {'/media/DATA1/plamen/mulford_20150204/20150204-2/left/leftCrop/', 110732781, 3445}, ...
    {'/media/DATA1/plamen/mulford_20150204/20150204-1/right/rightCrop/', 110732692, 1235}, ...
    {'/media/DATA1/plamen/mulford_20150204/20150204-1/left/leftCrop/', 110732781, 1235} ...
};
path = path_info{dataset_index}{1};
magic_num = path_info{dataset_index}(2); magic_num = magic_num{1};
num_img = path_info{dataset_index}(3); num_img = num_img{1};
savepath = sprintf('~/bndbox_%d.mat', dataset_index);

%% main logic
bndboxes = cell(1, num_img);
for i = 1:num_img
    save(savepath, 'bndboxes')
    fn = sprintf('%sCamera_%d_Image%06d.jpg', path, magic_num, i);
    imshow(fn)
    rects = [];
    while 1
        rect = getrect;
        if min(rect(1), rect(2)) < 0
            bndboxes{i} = rects;
            break
        end
        rects = [rects; rect];
        if rect(1) + rect(3) > size(imread(fn), 2)
            rects = [];
        end
    end
end
save(savepath, 'bndboxes')
save([savepath, '_copy'], 'bndboxes')
close all
msg = [...
    'CONGRATULATIONS! You have completed dataset number ', ...
    num2str(dataset_index), '\n', ...
    'Output file is saved as ', savepath, '\n', ...
    'DONT WORRY. Once you change the dataset index, the old file WONT be replaced.\n'];
disp(sprintf(msg))
