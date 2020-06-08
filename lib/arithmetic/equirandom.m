function [ bin,newtrainlog,I] = equirandom(bin,trainlog,eqsize,isrev,I )
%EQUIRANDOM Summary of this function goes here
%   Detailed explanation goes here
blocknum = size(bin,1)/eqsize;
if ~isrev
    if isempty(I)
    I = randperm(size(bin,1),size(bin,1));
    I = 1 : size(bin,1);
    end
    bin = bin(I,:);
    bin = reshape(bin,eqsize,[]);
    newtrainlog = gpuArray.false(1,size(bin,2));
    newtrainlog(1:(sum(trainlog)*blocknum)) = true;
else
    [~,Iinv] = sort(I);
    bin =reshape(bin,[],numel(trainlog));
    bin = bin(Iinv,:);
end

end

