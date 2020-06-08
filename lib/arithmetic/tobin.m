function [bin] = tobin(data,grey)
grey = false;
if nargin < 2
    grey = false;
end
% last dimension is the sample dimension
if ~isa(gather(data(1)),'uint8')
    error('other class types may not be supported')
end

bin = gpuArray.zeros([size(data),8],'int8');
for i = 8 :-1: 1
    bin(:,:,:,:,i) = mod(data,2);
    data = floor(data./2)-uint8(bin(:,:,:,:,i)) ;
end
clear('data');
bin = permute(bin,[1,2,3,5,4]);
if grey
    bin = cumsum(bin,4);
    bin = mod(bin,2);
end
bin = logical(bin);
% bin= (bin.*2-1);
end

