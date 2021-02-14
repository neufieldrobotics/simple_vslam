close all
clear all

%find the images

fold = '/data/Stingray/Stingray2_080718_masks_from_model'
outfold = strcat(fold, '_800x600_test')

[parent, newfold, file] = fileparts(outfold)
mkdir(parent, newfold)

d=dir(fullfile(fold,'*.png'));
d = d(not([d.isdir]));
d = d(arrayfun(@(x) x.name(1), d) ~= '.');

% Turn off divide by zero warnings
warning('off','MATLAB:imagesci:tifftagsread:badTagValueDivisionByZero')

%figure
%imshowpair(X,J,'montage')
%title('Original (left) and Contrast Enhanced (right) Image')

numIterations = size(d,1)
ppm = ParforProgressbar(numIterations, 'showWorkerProgress', true, 'parpool', 'local');

%parpool open 12
tic
parfor (count_ind = numIterations) %404:size(d,1)
%for count_ind = 1:100
    count_ind
    X  = imread(fullfile(d(count_ind).folder,d(count_ind).name));
    X = imresize(X, 1/5, 'bicubic', 'Antialiasing', true);
    %X  = rgb2gray(X)
    X = im2bw(X)
    [filepath,name,ext] = fileparts(d(count_ind).name)
    imwrite(X,strcat(outfold,'/',name,'.png'),'png')
    ppm.increment();
    
end

toc / numIterations
delete(ppm);
  
%Turn warning back on
warning('on','MATLAB:imagesci:tifftagsread:badTagValueDivisionByZero')