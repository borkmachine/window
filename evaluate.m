% given an img and a mask, run window detection and evaluate
function evaluate(img_dir, imgname, normal_dir)
    addpath('mcg/full');
    install
    I = im2double(imread([img_dir imgname]));
    nmap = imread([normal_dir imgname(1:end-4) '_normalmap.png']);
    nmap = process_normap(nmap);
    
    
    [candidates_mcg, ~] = im2mcg(I,'fast');
    % generate and filter bounding boxes
    for id=1:length(candidates_mcg.labels)
        mask = ismember(candidates_mcg.superpixels, candidates_mcg.labels{id});
        mask = double(cat(3,cat(3, mask, mask), mask));        
        box = candidates_mcg.bboxes(id,:);
        b2 = [box(2) box(1) box(4)-box(2) box(3)-box(1)];
        
        cropped_normal = imcrop(nmap.*mask, b2);
        % filter the ceiling and floor
        if sum(sum(or(cropped_normal > 0.9 , cropped_normal < -0.9))) > 0.1*numel(cropped_normal)
            continue
        end
        
        I = insertObjectAnnotation(I, 'rectangle', b2, 'box')
        
        
    end
    
end