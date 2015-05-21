% A simple script that goes through a directory of images and asks a user to draw a polygon on them using ROIPoly. 
% Since pictures sometimes have multiple regions, hitting ‘a’ or ‘A’ will allow for another region to be drawn, 
% this can be repeated as many times as needed. Hitting any other key after a region will go to the next image. 
% Currently this also rotates the images by 90 degrees.


files = dir('*.jpg');
fileID = fopen('windows.txt', 'w');
h=figure;
for i=1:length(files)
    count = 1;
    file = files(i);
    I = imrotate(imread(file.name),90);
    imshow(I)
    BW = roipoly(I);
    if ~isempty(BW)
        imwrite(BW, [file.name '_mask_' num2str(count) '.png'])
    end
    k = waitforbuttonpress;
    while or(h.CurrentCharacter  == 'a',h.CurrentCharacter  == 'A')
        count = count + 1
        disp 'yolo'
        BW = roipoly(I);
        imwrite(BW, [file.name '_mask_' num2str(count) '.png'])

        k = waitforbuttonpress;
    end
end
