function [ CodeMatrix,CodeMatrixinv ] = bubblewrap( bin,trainind,valind,testind,verbose,method)
%BUBBLEWRAP Summary of this function goes here
%   Detailed explanation goes here
bin = permute(bin,[3,4,1,2,5]);
bin = reshape(bin,[],size(bin,ndims(bin)));
CodeMatrix = [];
CodeMatrixinv = CodeMatrix;
valentall = [];
verbose.exist = true;
method.startmod = inf;
batchsz =1000;
while(true)
    batchind=randperm(numel(trainind),batchsz);
    batchind = trainind(batchind);
    batch = bin(:,batchind);
    [CodeMatrix,CodeMatrixinv,binbatch,~,~] = xorinfermat(gpuArray(batch),gpuArray(CodeMatrix),gpuArray(CodeMatrixinv),method,verbose);
    valbin = single(CodeMatrix) * single(bin(:,valind));
    valbin = logical(mod(valbin,2));
    valprob = mean(valbin,2);
    valent = sum(-valprob.*log2(valprob) - (1-valprob).*log2(1-valprob),'omitnan');
    valentall = cat(1,valentall,valent);
    talk(valent)
    batchprob = mean(binbatch,2);
    figure(1);
    [CodeMatrixinv] = showgenimage(batchprob,CodeMatrixinv,true);
    figure(2);
    imshow(imresize(gather(CodeMatrix),[500,500]),[])
end

end
function [CodeMatrix,CodeMatrixinv] = regularize(CodeMatrix,CodeMatrixinv)
imresponse = logical(mod(sum(CodeMatrix,2),2));
sender = find(imresponse,1,'last');
imresponse(sender) = 0;
CodeMatrix(imresponse,:) = xor(CodeMatrix(sender,:),CodeMatrix(imresponse,:));
CodeMatrixinv(:,sender) = xor(mod(sum(CodeMatrixinv(:,imresponse),2),2),CodeMatrixinv(:,sender));
end
function [] = talk(valent)
fprintf('------Validation Entropy %.4f',valent);
fprintf('\n');
end
function bin = shiftrandom(bin,origsize)
sampnum = size(bin,2);
bin = reshape(bin,[origsize,sampnum]);
sz1 = size(bin,1);
sz2 = size(bin,2);
rand = gpuArray.rand(1,2);
rand = floor(rand.*[sz1,sz2]/2)+1;
bin = circshift(bin,[rand,0,0]);
%  bin(1:rand(1),:,:,:) = 0;
%  bin(:,1:rand(2),:,:) = 0;
bin = reshape(bin,[],sampnum);
end


function [CodeMatrixinv] = showgenimage(prob,CodeMatrixinv,issparse)
imnum = 4;
if ~issparse
    prob = repmat(prob,1,imnum);
    im = gpuArray.rand(size(prob))>1-prob;
    im = mod(single(CodeMatrixinv) * single(im),2);
else
    [~,inds] = sort(prob,1,'descend');
    im  = CodeMatrixinv(:,inds(1:imnum));
end
im = reshape(im,32,32,3,8,imnum);
plotpics(single(im),imnum,4);
end
function [] = plotpics(bin,numpics,binnum)
rsz  = [numpics,binnum];
for i = 1: numpics
    for j = 1 : binnum
    subplot(rsz(2),rsz(1),sub2ind(rsz,i,j));imshow(bin(:,:,:,j,i),[]);
    tname = sprintf('Image: %d , Bin: %d',i,j);
    title(tname);
    end
end
drawnow;
end

