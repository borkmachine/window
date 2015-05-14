imgs = dir('result_Cam*.jpg');
imgdir = '/home/chiller/400coryIms_masked/';

precision = [];
recall = [];

for threshold = 0:0.01:0.99

    % for each image calculate the accuracy
    sum_recall = 0;
    sum_precision = 0;
    for i = 1:size(imgs, 1)
        imgname = imgs(i).name;
        img = im2single(imread(imgname));
        imshow(img)

        masks = dir([imgdir, imgname(8:end-4), '*.png']);
        new_mask = zeros(size(img));
        for j = 1:size(masks, 1)
            maskname = masks(j).name;
            new_mask = or(new_mask, imresize(imread([imgdir maskname]), 0.25));
        end


        img(img > threshold) = 1;
        img(img <= threshold) = 0;


        sum_recall = sum_recall + calculate_overlap(new_mask, logical(img));
        sum_precision = sum_precision + calculate_overlap(logical(img), new_mask);

    end

    avg_recall = sum_recall/size(imgs,1)
    avg_precision = sum_precision/size(imgs,1)
    
    precision = [precision, avg_precision];
    recall = [recall, avg_recall];
end

plot(recall, precision);