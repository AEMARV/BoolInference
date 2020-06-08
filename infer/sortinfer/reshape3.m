function [ bin,newtrainlog,sortinv] = reshape3( bin,trainlog )
%RESHAPE3 Summary of this function goes here
%   Detailed explanation goes here
sz = 64;
bitnum = size(bin,1);
% Sort = randperm(size(bin,1),size(bin,1));
Sort = 1:size(bin,1);
[~,sortinv] = sort(Sort);
bin = bin(Sort,:);
bin = reshape(bin,sz,[]);
newtrainlog = false(1,size(bin,2));
newtrainlog(1:((bitnum).*sum(trainlog))/sz) = true;
end

