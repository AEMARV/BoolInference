function [ bin] = convxor( bin,dim,sampdim,trainind,valind,inverse,mask)
%CONVXOR Summary of this function goes here
%   Detailed explanation goes here

if nargin<7
    mask = or(true,bin);
end
bin = and(bin,mask);
assert(sampdim==5);
if inverse
  bin = reversedim(bin,dim);  
end
binout = bin;
dimsz = size(bin,dim);
iters = 2.^ceil(log2(dimsz));
 [probbin] = meanfull(bin(:,:,:,:,trainind),mask(:,:,:,:,trainind),dim);
minent = entropy(probbin);
minent = sum(minent(:),'omitnan');
for i = 1 : iters
    bin = logical(mod(cumsum(bin,dim),2));
    bin = and(bin,mask);
    probbin = meanfull(bin(:,:,:,:,trainind),mask(:,:,:,:,trainind),dim);
%     [probbin] = mean(and(bin(:,:,:,:,trainind),mask(:,:,:,:,trainind)),sampdim);
    ent = entropy(probbin);
    ent = sum(ent(:),'omitnan');
    if ent <= minent
        binout = bin;
        minent = ent;
    end
end
bin = binout;
if inverse
    bin = reversedim(bin,dim);
end
clear('binout');
end
function [prob] = meanmasked(bin,mask,dim)
if all(mask(:))
    prob = mean(bin,dim);
    return;
end
nanratio = sum(mask,dim)./size(mask,dim);
bin = and(bin,mask);
prob = sum(bin,dim)./(size(bin,dim).*nanratio);
end
function [prob] = meanfull(bin,mask,dim)
mask = single(mask);
mask(mask==0) = nan;
iters = 0;
bin = bin.*mask;
for i = 1 : ndims(bin)
    if i == dim
        continue;
    end
    if iters == 0
        prob = mean(bin,i,'omitnan');
        iters = 1;
    else
        prob = mean(prob,i,'omitnan');
    end
end
clear('bin');
end
function [bin] = swapdim(bin,dim1,dim2)
permmat = 1:ndims(bin);
permmat(dim2) = dim1;
permmat(dim1) = dim2;
bin = permute(bin,permmat);

end
function bin = reversedim(bin,dim)
bin = swapdim(bin,1,dim);
bin(1:size(bin,1),:) = bin(size(bin,1):-1:1,:);
bin = swapdim(bin,1,dim);
end
