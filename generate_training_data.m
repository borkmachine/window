function generate_training_data(outdir_pos, outdir_neg, imgname, imgdir)
    % generate region candidates with mcg
    addpath('mcg');
    I = im2double(imread([imgdir imgname]));
    
    gts = dir([imgdir imgname(1:end-4), '*.png']); % all the ground truths
    ground_truth = zeros(size(I,1), size(I,2));
    for i = 1:size(gts,1)
        gt = imread(gts(i).name);
        ground_truth = or(ground_truth, gt);
    end
    
    I = imresize(I, 0.5);
    ground_truth = imresize(ground_truth, 0.5);
    
    [candidates_mcg, ~] = im2mcg(I,'fast');

    % For each candidate, classify according to ground truth, crop and save

    for id=1:length(candidates_mcg.labels)
        mask = ismember(candidates_mcg.superpixels, candidates_mcg.labels{id});
        mask = double(cat(3,cat(3, mask, mask), mask));

        box = candidates_mcg.bboxes(id,:);
        b2 = [box(2) box(1) box(4)-box(2) box(3)-box(1)];

        
        if calculate_overlap(mask(:,:,1), ground_truth) > 0.5
            imwrite(imcrop(I.*mask, b2), [outdir_pos imgname '-' num2str(id) '.jpg']);
        else
            imwrite(imcrop(I.*mask, b2), [outdir_neg imgname '-' num2str(id) '.jpg']);
        end
    end
end