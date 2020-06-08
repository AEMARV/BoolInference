function [ sample] = createsample( prob,trynum )
%CREATESAMPLE Summary of this function goes here
%   Detailed explanation goes here
bitnum = 30;
sample = prob.*0;
for i = 1 : trynum
    r = rand([1,bitnum])>0.5;
[~,r] = max(r,[],ndims(r));
r = randperm(bitnum,1);
basis = prob.*2.^(r);
basis = floor(basis);
basis = mod(basis,2);
sample = sample + basis;
im = imshow((single(reshape(sample,32,32,3))/i),[],'InitialMagnification','fit');
drawnow update;
end

end

