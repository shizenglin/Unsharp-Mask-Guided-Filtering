% example: detail enhancement
% figure 6 in our paper

close all;

I = double(imread('./img_enhancement/tulips.bmp')) / 255;
p = I;

r = 16;
eps = 0.1^2;
n=1;

umgf_q = zeros(size(I));

umgf_q(:, :, 1) = umgf_guidedfilter(I(:, :, 1), p(:, :, 1), r, eps, n);
umgf_q(:, :, 2) = umgf_guidedfilter(I(:, :, 2), p(:, :, 2), r, eps, n);
umgf_q(:, :, 3) = umgf_guidedfilter(I(:, :, 3), p(:, :, 3), r, eps, n);
umgf_enhanced = (I - umgf_q) * 5 + umgf_q;

gf_q = zeros(size(I));
gf_q(:, :, 1) = guidedfilter(I(:, :, 1), p(:, :, 1), r, eps);
gf_q(:, :, 2) = guidedfilter(I(:, :, 2), p(:, :, 2), r, eps);
gf_q(:, :, 3) = guidedfilter(I(:, :, 3), p(:, :, 3), r, eps);
gf_enhanced = (I - gf_q) * 5 + gf_q;

imwrite(gf_enhanced,'./tulips-enhance-gf.png');
imwrite(umgf_enhanced,'./tulips-enhance-umgf.png');
