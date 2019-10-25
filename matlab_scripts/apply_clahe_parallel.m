close all
clear all

%find the images

fold = '/data/cervino_timelapse_5/time_lapse_5_cervino' 
d=dir(fullfile(fold,'*.JPG'));
d = d(not([d.isdir]));
d = d(arrayfun(@(x) x.name(1), d) ~= '.');

% Turn off divide by zero warnings
warning('off','MATLAB:imagesci:tifftagsread:badTagValueDivisionByZero')

%figure
%imshowpair(X,J,'montage')
%title('Original (left) and Contrast Enhanced (right) Image')

%parpool open 12
tic
parfor (count_ind = 1:size(d,1), 32)%404:size(d,1)
%for count_ind = 1:100

    count_ind;
    [X MAP] = imread(fullfile(d(count_ind).folder,d(count_ind).name),'jpg');
    
    X = imresize(X, 1/5, 'bicubic', 'Antialiasing', true);
    
    %RGB = ind2rgb(X,MAP);
    
    LAB = rgb2lab(X);
      
    L = LAB(:,:,1)/100;

    L = adapthisteq(L,'NumTiles',[3 4],'ClipLimit',0.005,'NBins',512,'Distribution','rayleigh');
    
    LAB(:,:,1) = L*100;
    
    J = lab2rgb(LAB);
    
    imwrite(J,strcat(d(count_ind).folder,'_800x600_clahe_test/',d(count_ind).name,'.png'),'png')
    
end

toc 
  
%Turn warning back on
warning('on','MATLAB:imagesci:tifftagsread:badTagValueDivisionByZero')