function generate_training_data(outdir_pos, outdir_neg, imgname, imgdir)
    % generate region candidates with mcg
    addpath(genpath('mcg/full'));
    I = (imread([imgdir imgname]));
    
    gts = dir([imgdir imgname(1:end-4), '*.png']); % all the ground truths
    ground_truth = zeros(size(I,1), size(I,2));
    for i = 1:size(gts,1)
        gt = imread([imgdir gts(i).name]);
        ground_truth = or(ground_truth, gt);
    end
	%size(ground_truth)
	%size(I)   
%I = imrotate(I, 90); 
%	#imshow(I)    
	
    if numel(ground_truth) > 500000
        I = imresize(I, 0.5);
        ground_truth = imresize(ground_truth, 0.5);
    end
    %imshow(I)
    %figure;
    %imshow(ground_truth)
    [candidates_mcg, ~] = im2mcg(I,'fast');

    % For each candidate, classify according to ground truth, crop and save
    I = im2double(I);
    for id=1:length(candidates_mcg.labels)
        mask = ismember(candidates_mcg.superpixels, candidates_mcg.labels{id});
        mask = double(cat(3,cat(3, mask, mask), mask));

        box = candidates_mcg.bboxes(id,:);
        b2 = [box(2) box(1) box(4)-box(2) box(3)-box(1)];

        percOverlap = calculate_overlap(logical(mask(:,:,1)), logical(ground_truth));
        if percOverlap > 0.95
            imwrite(imcrop(I.*mask, b2), [outdir_pos imgname '-' num2str(id) '_' num2str(percOverlap) '.jpg']);
        else
            imwrite(imcrop(I.*mask, b2), [outdir_neg imgname '-' num2str(id) '_' num2str(percOverlap) '.jpg']);
        end
    end
end
