%write_mask('test1.png',[3000,4000],[717,1744,1561,3000])
%find_corners('test1.png')

%% Read list of image_names for which masks have to be created
image_fold = '../Lars2_081018_clahe' 
d=dir(image_fold);
d = d(not([d.isdir]));
d = d(arrayfun(@(x) x.name(1), d) ~= '.');
image_names=cell2mat({d.name}');
image_names=string(image_names(:,1:end-4));

%% Read masks and populate the corners table with known mask locations
mask_fold = './Lars2_081018_clahe_masks2_resize_640x480'
mask_out_fold = strcat(mask_fold,'_txt/')
md = dir(mask_fold);
md = md(not([md.isdir]));
md = md(arrayfun(@(x) x.name(1), md) ~= '.');
mask_names=cell2mat({md.name}');
mask_names=string(mask_names(:,1:end-9));
img_size= size(imread(fullfile(md(1).folder,md(1).name)))
corners = NaN(size(image_names,1),4);

for (count = 1:size(md,1))
    count;
    %known_corners(count,:)=find_corners(fullfile(d(count).folder,d(count).name));
    corners(contains(image_names,mask_names(count)),:)=find_corners(fullfile(md(count).folder,md(count).name));
end


%% Interpolate and fill 
corners = fillmissing(corners,'linear',1);
corners = round(corners);
corners(corners<1)=1;
corners(corners>img_size(2))=img_size(2);
x1=corners(:,1);
x1(x1>img_size(1))=img_size(1);
corners(:,1)=x1;

x2=corners(:,3);
x2(x2>img_size(1))=img_size(1);
corners(:,3)=x2;

%% Write masks to file
parfor (count = 1:size(d,1),12)%size(d,1)
    %write_mask(strcat(mask_out_fold,d(count).name(1:end-4),'_mask.png'),img_size,corners(count,:))
    write_mask2text(strcat(mask_out_fold,d(count).name(1:end-4),'.txt'),corners(count,:))
end

%% Require functions
function mask_out = write_mask(filename,size,corners)
out = zeros(size);
out(corners(1):corners(3),corners(2):corners(4))=255;
%imshow(out)
imwrite(out,filename)
end

function to = write_mask2text(filename,corners)
filename
fileID = fopen(filename,'w')
fprintf(fileID,'%d,%d,%d,%d',corners);
fclose(fileID);
end


function corners = find_corners(filename)
a = imread(filename);
x_any = any(a,2); 
y_any = any(a,1);
xs = find(diff(x_any)~=0);
ys = find(diff(y_any)~=0);

if size(ys,2)==1
    if y_any(1)
        ys = [0;ys];
    elseif y_any(size(a,2))    
        ys = [ys;size(a,2)];
    end
elseif size(ys,2)==0;
    ys = [0;size(a,2)];
end

if size(xs,2)==1
    if x_any(1)
        xs = [0;xs];
    elseif x_any(size(a,1))    
        xs = [xs;size(a,1)];
    end
elseif size(xs,2)==0;
    xs = [0;size(a,1)];
end

corners = [xs(1)+1,ys(1)+1,xs(2),ys(2)];
end