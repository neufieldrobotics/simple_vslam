iname = '/home/vik748/data/low_contrast_datasets/skerki_mud/ESC.970622_024806.0590.tif'
[filepath,name,ext] = fileparts(iname)

X = imread(iname);
figure
imshow(X)

figure
X_stretch = imadjust(X,lowhigh);
imshow(X_stretch)
imwrite(X_stretch, fullfile(filepath,strcat(name,'_stretch',ext)))

figure
X_histeq = histeq(X);
imshow(X_histeq)
imwrite(X_histeq, fullfile(filepath,strcat(name,'_histeq',ext)))

figure
X_clahe = adapthisteq(X,'NumTiles',[3 4],'ClipLimit',0.005,'NBins',512,'Distribution','rayleigh');
imshow(X_clahe)
imwrite(X_clahe, fullfile(filepath,strcat(name,'_clahe',ext)))

figure
X_adhisteq = adapthisteq(X,'NumTiles',[3 4],'ClipLimit',0.5,'NBins',512,'Distribution','rayleigh');
imshow(X_adhisteq)
imwrite(X_adhisteq, fullfile(filepath,strcat(name,'_adhisteq',ext)))