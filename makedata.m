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