function [ ] = complexinf( prob,range,numsample )
%COMPLEXINF Summary of this function goes here
%   Detailed explanation goes here
iterative = true;
 
w = 1:range/numsample:range;
w = 1:0.01:40;
% w = exp(w);
% w = linspace(min(prob(:)),max(prob(:)),10000);
% w = 1:1:1000;
figure(1);
res = 0;
%  prob = 2.*prob-1;
 ent = -prob.*log2(prob)-(1-prob).*log2(1-prob);
% probcomp = sqrt(prob) + i.*sqrt(1-prob);
% theta = angle(probcomp);
% prob = ent;
% prob = sqrt(prob);
for k= 1 : numel(w)
      im = mod(single(floor((2.^(log2(prob).*(w(k).*ent))))),2);
%          im = prob.*w(k) - floor(prob.*w(k));
%        im = prob>(w(k));
%          im = cos((ent.*prob.*exp(w(k))));
%         im = im.^2;
%        im = exp(j.*w(k).*prob);
%         im = exp(w(k)).*prob - floor(exp(w(k)).*prob);
%        im = exp(j.*floor(exp(w(k)).*(prob)));
%     im = imshow(real(single(reshape(,32,32,3))),[],'InitialMagnification','fit');
     im = real(im);
    im = reshape(im,32,32,3,8);
    im = im.*reshape(2.^(-1:-1:-8),1,1,1,8);
    im = sum(im,4);
    f = imshow(real(single(reshape(im,32,32,3))),[],'InitialMagnification','fit');
    title(['w=',num2str(w(k))]);
    drawnow update;
end

end
