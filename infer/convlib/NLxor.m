function [ bin ] = NLxor( bin,dim ,trainind,valind,sampledim,isreverse)
%NLXOR Summary of this function goes here
%   Detailed explanation goes here
assert(sampledim==5);
[bin,prob] = meanfull(bin,dim);
ent = entropy(prob);
I = choosechannel(ent);
bin = swapdim(bin,1,dim);
mask = bin(I,:,:,:,:);
restind = [1:(I-1),I+1:size(bin,1)];

bin1 = bin;
bin1(restind,:,:,:,:) = linoperate(bin1(restind,:,:,:,:),mask,trainind,valind,false,sampledim);
bin(restind,:,:,:,:) = linoperate(bin(restind,:,:,:,:),~mask,trainind,valind,false,sampledim);
bin(restind,:,:,:,:) = or(and(mask,bin1(restind,:,:,:,:)),and(~mask,bin(restind,:,:,:,:)));
bin = swapdim(bin,1,dim);
end
function [bin] = linoperate(bin,mask,trainind,valind,isreverse,sampledim)
inds = randperm(4);
for i =  1: numel(inds)
 bin = convxor(bin,inds(i),sampledim,trainind,valind,isreverse,mask);
end
end
function I = choosechannel(ent)
methodnum =2 ;
switch methodnum
    case 1
        method = 'maxent';
    case 2
        method = 'rand';
end
switch method
    case 'maxent'
        [~,I] = max(ent);
    case 'rand'
        I = randperm(numel(ent),1);
end
end
function [bin] = swapdim(bin,dim1,dim2)
permmat = 1:ndims(bin);
permmat(dim2) = dim1;
permmat(dim1) = dim2;
bin = permute(bin,permmat);

end
function [bin,prob] = meanfull(bin,dim)
iters = 0;
for i = 1 : ndims(bin)
    if i == dim
        continue;
    end
    if iters == 0
        prob = mean(bin,i);
        iters = 1;
    else
        prob = mean(prob,i);
    end
end
end
