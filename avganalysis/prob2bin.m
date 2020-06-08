function [ mat] = prob2bin( prob,bitnum )
%AVGMATGENERATE Summary of this function goes here
%   Detailed explanation goes here
% mat = logical(eye([numel(prob),bitnum],'like',prob));
% % % prob = 1-prob;
%    prob = prob - min(prob);
%     prob = min(prob,1-prob);
 prob = prob./max(prob);
entall = mean(prob);
entavgprob = - entall.*log2(entall) - (1-entall).*log2(1-entall);
bitcount = -log2(max(prob));
mat = 0;
bitcount = 1;
while numel(find(mat))<2
prob1 = floor(prob.*(2.^(bitcount)));

mat = logical(mod((prob1),2));
bitcount = bitcount +1;
end
fprintf('bitplane: %d ',bitcount-1);
%  mat = (prob>(floor(q)*chunksize)& prob<ceil(q)*chunksize);
% mat = prob >(1- chunksize);

set(groot,'CurrentFigure',1);
im = mat;
im = reshape(im,32,32,3,8);
im = im.*reshape(2.^(-1:-1:-8),1,1,1,8);
im = sum(im,4);
f = imshow(real(single(reshape(im,32,32,3))),[],'InitialMagnification','fit');
drawnow update;
set(groot,'CurrentFigure',2);
im = prob;
f = plot(1:numel(im),im);
drawnow update;

end
function ent1 = entropy(p)
ent1 = -p.*log2(p) - (1-p).*log2(1-p);
ent1(isnan(ent1)) = 0;
end
function inds = firstonebit(prob)
found = zeros(size(prob),'like',prob);
inds = found;
i = 0;
while ~all(found | prob==0)
    f = prob >= 2.^(-i);
    inds(and(f,~found)) = i;
    found(f) = 1;
    i = i+1;
end
end

