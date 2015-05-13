% given an img and a mask, run window detection and evaluate
function evaluate(img_dir, imgname, normal_dir)
    addpath('mcg/full');
    install
    I = im2double(imread([img_dir imgname]));
    nmap = imread([normal_dir imgname(1:end-4) '_normalmap.png']);
    nmap = process_normap(nmap);
    nmap = zeros(size(nmap));

    [candidates_mcg, ~] = im2mcg(I,'fast');
    save('temp.mat', 'candidates_mcg');
    %load('temp.mat');

    valid_ids = [];
    % generate and filter bounding boxes
    for id=1:length(candidates_mcg.labels)
        mask = ismember(candidates_mcg.superpixels, candidates_mcg.labels{id});
        mask = double(cat(3,cat(3, mask, mask), mask));
        box = candidates_mcg.bboxes(id,:);
        b2 = [box(2) box(1) box(4)-box(2) box(3)-box(1)];

        cropped_normal = imcrop(nmap.*mask, b2);
        cropped_normal = cropped_normal(:,:,1);
        % filter the ceiling and floor
        if sum(sum(or(cropped_normal > 0.9 , cropped_normal < -0.9))) > 0.1 * sum(sum(mask))
            disp('filtered');
            continue
        end

        valid_ids = [valid_ids, id];
        I = insertObjectAnnotation(I, 'rectangle', b2, 'box');
    end
    save('test.mat', 'valid_ids');

    setenv('CAFFE_ROOT', '/home/chiller/caffe');
    setenv('CAFFE_DIST', '/home/chiller/caffe/distribute');
    setenv('PATH', '/home/chiller/caffe/distribute/bin:/usr/local/cuda/bin:/opt/anaconda/bin:$PATH:/share/instsww/pkg/matlab-r2012b/bin');
    setenv('LD_LIBRARY_PATH', '/home/chiller/caffe/distribute/lib:/usr/local/cuda/lib64:/opt/anaconda/lib');
    setenv('PYTHONPATH', '/home/chiller/caffe/distribute/python');
    commandStr = '/usr/bin/python imnet.py';
    [status, commandOut] = system(commandStr);
    if status == 1
        disp('python failed');
        return
    end
    load('outputVec.mat');

    % generate the mask based on the result passed from classifier
    result_mask = zeros(size(I,1), size(I,2));
    for id = idx
        mask = ismember(candidates_mcg.superpixels, candidates_mcg.labels{id});
        box = candidates_mcg.bboxes(id,:);
        b2 = [box(2) box(1) box(4)-box(2) box(3)-box(1)];
        % binary scheme
        % result_mask = or(result_mask, mask);
        % confidence encoded scheme
        result_mask(box(2):box(4), box(1):box(3)) = result_mask(box(2):box(4), box(1):box(3)) + 1;
    end
    result_mask = result_mask / max(max(result_mask));
    figure(2);
    imshow(result_mask);
end
