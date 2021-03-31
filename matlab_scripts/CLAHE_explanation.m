X = rgb2gray(imread('/home/vik748/data/Lars1_080818/G0285915.JPG'));

L = adapthisteq(X,'NumTiles',[12 16],'ClipLimit',0.005,'NBins',512,'Distribution','rayleigh');

rect = [500 1000 500 500]

figure(1)
imshow(X, 'Border','tight')
title('Raw')
rectangle('Position', [500 1000 500 500], 'LineWidth',2)

figure(2)
imshow(L, 'Border','tight')
title('CLAHE')
rectangle('Position', [500 1000 500 500], 'LineWidth',2)

[A, rect] = imcrop(X, rect);

[B, rect] = imcrop(L,rect)

edges = linspace(0,255,255/5+1);
figure(3)
t = tiledlayout(2,3,'TileSpacing','Compact','Padding','Compact');
nexttile
imshow(A, 'Border','tight')
title('Raw')

nexttile
imshow(B, 'Border','tight')
title('CLAHE')

nexttile
C = histeq(A);
imshow(C, 'Border','tight')
title('Histogram Equalization')

nexttile
imshow(A, 'Border','tight')
ha = histogram(A, edges), ylim([0 6e4]), xlim([0 255]);
colormap gray
cb = colorbar('location','southoutside','TickLabels',{''})
xlabel(cb,'Intensity (bins)');
ylabel('Frequency')

nexttile
imshow(A, 'Border','tight')
ha = histogram(B, edges), ylim([0 6e4]), xlim([0 255]);
colormap gray
cb = colorbar('location','southoutside','TickLabels',{''})
xlabel(cb,'Intensity (bins)');
ylabel('Frequency')

nexttile
imshow(A, 'Border','tight')
ha = histogram(C, edges), ylim([0 6e4]), xlim([0 255]);
colormap gray
cb = colorbar('location','southoutside','TickLabels',{''})
xlabel(cb,'Intensity (bins)');
ylabel('Frequency')


tick_labels = get(gca,'Xticklabel')
ticks = get(gca,'Xtick')
set(gca,'Xticklabel',[])
hbar = colorbar('horiz', 'Ticks', ticks,'TickLabels', tick_labels)


set(hbar,'location','manual')


