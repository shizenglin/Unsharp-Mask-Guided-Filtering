% example: flash/noflash denoising
% figure 8 in our paper
% *** Errata ***: there is a typo in the caption of figure 8, the eps should be 0.02^2 instead of 0.2^2; sig_r should be 0.02 instead of 0.2.

close all;

I = double(imread('./img_flash/cave-flash.png')) / 255;
p = double(imread('./img_flash/cave-noflash.png')) / 255;

r = 8;
eps = 0.02^2;
eps1 = 0.02^2;
n=1;

umgf_q = zeros(size(I));

umgf_q(:, :, 1) = umgf_guidedfilter(I(:, :, 1), p(:, :, 1), r, eps1, n);
umgf_q(:, :, 2) = umgf_guidedfilter(I(:, :, 2), p(:, :, 2), r, eps1, n);
umgf_q(:, :, 3) = umgf_guidedfilter(I(:, :, 3), p(:, :, 3), r, eps1, n);

gf_q = zeros(size(I));
gf_q(:, :, 1) = guidedfilter(I(:, :, 1), p(:, :, 1), r, eps);
gf_q(:, :, 2) = guidedfilter(I(:, :, 2), p(:, :, 2), r, eps);
gf_q(:, :, 3) = guidedfilter(I(:, :, 3), p(:, :, 3), r, eps);

imwrite(gf_q,'./img_flash/cave-noflash-gf.png');
imwrite(umgf_q,'./img_flash/cave-noflash-umgf.png');
