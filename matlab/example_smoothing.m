% example: edge-preserving smoothing
% figure 1 in our paper

close all;

RGB = imread('./img_smoothing/girl.jpg');
I_gray = rgb2gray(RGB);

I = double(I_gray) / 255;%imread('./img_smoothing/cat.bmp')
p = I;
r = 8; % try r=2, 4, or 8
eps = 0.02^2; % try eps=0.1^2, 0.2^2, 0.4^2
n = 1;

umgf_q = umgf_guidedfilter(I, p, r, eps, n);
gf_q = guidedfilter(I, p, r, eps);

imwrite(gf_q,'./girl_gf.png');
imwrite(umgf_q,'./girl_umgf.png');
