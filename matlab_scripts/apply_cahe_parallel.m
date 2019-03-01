close all
clear all

%find the images

fold = '../Lars2_081018' 
d=dir(fold);
d = d(not([d.isdir]));
d = d(arrayfun(@(x) x.name(1), d) ~= '.');

% Turn off divide by zero warnings
warning('off','MATLAB:imagesci:tifftagsread:badTagValueDivisionByZero')

%figure
%imshowpair(X,J,'montage')
%title('Original (left) and Contrast Enhanced (right) Image')



%parpool open 12
tic
parfor (count = 1:size(d,1), 12)%404:size(d,1)
    count;
    X = imread(fullfile(d(count).folder,d(count).name));
    
    %RGB = ind2rgb(X,MAP);
    
    LAB = rgb2lab(X);
    
    
    L = LAB(:,:,1)/100;

    L = adapthisteq(L,'NumTiles',[12 16],'ClipLimit',0.005,'NBins',512,'Distribution','rayleigh');
    
    LAB(:,:,1) = L*100;
    
    J = lab2rgb(LAB);
    
    imwrite(J,strcat(d(count).folder,'_clahe/',d(count).name(1:end-4),'.tif'),'tif')
    
end

toc 
  


%Turn warning back on
warning('on','MATLAB:imagesci:tifftagsread:badTagValueDivisionByZero')