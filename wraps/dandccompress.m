function [ output_args ] = dandccompress( data,mehtod,verbose )
%DANDCCOMPRESS Summary of this function goes here
%   Detailed explanation goes here
labels = data.labels;
data = data.data;
if method.usegpu
%     data = gpuArray(data);
end
  data = gather(tobin(data,true));
% %  data = single(data)./255;
data = reshape(data,[],size(data,ndims(data)));
% data = permute(data,[1,3,2]);
bsz = method.maxbatchsize;
sampnum = size(data,2);
compressor = [];
decompressor = [];
for epoch = 1 : method.epochnum
for i = 1 : ceil(size(data,2)/bsz)
batchind = (((i-1)*bsz)+1) : min(sampnum,i.*bsz);
batchind = find(labels==1);
if method.usegpu
    batch = gpuArray(data(:,batchind));
    compressor = gpuArray(compressor);
    decompressor = gpuArray(decompressor);
else
    batch = data(:,batchind);
end
[compressor,decompressor,batch,prob] = xorinfermatdc(batch,compressor,decompressor,method.trainer,verbose);
data(:,batchind,:) = gather(batch);
end
end
end
function [compressor,decompressor,batch,prob] = xorinfermatdc(batch,compressor,decompressor,method,verbose);
epochnum = 
for i = 1 : epochnum
    
end
end
