% given an img and a mask, run window detection and evaluate
function evaluate(img_dir, imgname, normal_dir)
    addpath('mcg/full');
    install
    I = im2double(imread([img_dir imgname]));
    nmap = imread([normal_dir imgname(1:end-4) '_normalmap.png']);
    nmap = process_normap(nmap);
    imtool(nmap);
    
    [candidates_mcg, ~] = im2mcg(I,'fast');
    % generate and filter bounding boxes
    candidates_mcg
    
end