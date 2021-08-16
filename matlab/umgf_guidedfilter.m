function q = umgf_guidedfilter(I, p, r, eps, n)%,I_mask,mean_a,mean_mean_p,I_strc
%   GUIDEDFILTER   O(1) time implementation of guided filter.
%
%   - guidance image: I (should be a gray-scale/single channel image)
%   - filtering input image: p (should be a gray-scale/single channel image)
%   - local window radius: r
%   - regularization parameter: eps

[hei, wid] = size(I);
N = boxfilter(ones(hei, wid), r); % the size of each local patch; N=(2r+1)^2 except for boundary pixels.

mean_I = boxfilter(I, r) ./ N;
mean_mean_I = mean_I;
for i=1:n
    mean_mean_I = boxfilter(mean_mean_I, r) ./ N;
end

mean_p = boxfilter(p, r) ./ N;
mean_mean_p = boxfilter(mean_p, r) ./ N;

mean_Ip = boxfilter(I.*p, r) ./ N;
cov_Ip = mean_Ip - mean_I .* mean_p; % this is the covariance of (I, p) in each local patch.

mean_II = boxfilter(I.*I, r) ./ N;
var_I = mean_II - mean_I .* mean_I;

%mean_pp = boxfilter(p.*p, r) ./ N;
%var_p = mean_pp - mean_p .* mean_p;

a = cov_Ip ./ (var_I+eps); % Eqn. (5) in the paper;%sqrt(var_I.*var_p)

mean_a = boxfilter(a, r) ./ N;

I_mask = I-mean_mean_I;
I_strc = mean_a .* I_mask;
q = I_strc + mean_mean_p; % Eqn. (8) in the paper;
end
