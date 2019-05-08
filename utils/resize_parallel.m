close all
clear all

%find the images

fold = '/Users/vik748/Google Drive/data/Lars2_081018_clahe_masks2_out' 
d=dir(fullfile(fold,'*.png'));
d = d(not([d.isdir]));
d = d(arrayfun(@(x) x.name(1), d) ~= '.');

% Turn off divide by zero warnings
warning('off','MATLAB:imagesci:tifftagsread:badTagValueDivisionByZero')

%figure
%imshowpair(X,J,'montage')
%title('Original (left) and Contrast Enhanced (right) Image')

%parpool open 12
tic
parfor (count_ind = 1:size(d,1), 12) %404:size(d,1)
%for count_ind = 1:100
    count_ind
    X  = imread(fullfile(d(count_ind).folder,d(count_ind).name));
    
    X = imresize(X, 1/5, 'bicubic', 'Antialiasing', true);

    
    imwrite(X,strcat(d(count_ind).folder,'_800x600_test/',d(count_ind).name,'.png'),'png')
    
end

toc 
  
%Turn warning back on
warning('on','MATLAB:imagesci:tifftagsread:badTagValueDivisionByZero')