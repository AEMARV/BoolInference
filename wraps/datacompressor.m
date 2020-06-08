function [ imdb,compressor,decompressor ] = datacompressor(imdb,method,verbose )
% Converts the data to binary and compresses the binary
%  [ data,compressor,decompressor ] = datacompressor(data,method,verbose )
%==========================================================================
%  Constraints on data
%1: data should be quantized to integers 0:255. if this is not the case one
% can provide a quantizer and override tobin function in the lib folder.
%2: samples of the data must be on the last dimension of the data matrix.
%
%==========================================================================
% Outputs:
% Compressor is a logical square matrix with the number of rows
% equal to the number of bits in the data.
%
% To compress new data based on the current matrix one should multiply the
% matrix with the binary array and perform modular operation by 2.
%--------------------------------------------------------------------------
% Decompressor is a logical square matrix which is the inverse of
% Compressor in GL2n group.
% to decompress data. D is a the decompressor and x is the compressed data.
% y = mod(Dx,2);
%==========================================================================
% Inputs:
% method : a struct to pass option to the function. It can be constructed
% with parsemethod function
%--------------------------------------------------------------------------
% verbose: a struct with options for showing progress to user.
% verbose.exist should be true to show anything at all.
labels = imdb.labels;
labels = labels +1;
data = imdb.data;
if method.tobin
data = gather(tobin(data,false));
else
    data = double(data)./256;
end
data = reshape(data,[],size(data,ndims(data)));
bsz = method.maxbatchsize;
sampnum = size(data,2);
compressor = [];
decompressor = [];
compressor = {};
imdb.compressors = cell(1:numel(unique(labels)));
for curlabel = min(labels):max(labels)
    % batchind = (((i-1)*bsz)+1) : min(sampnum,i.*bsz);
    batchind = find(labels==curlabel);
    if method.usegpu
        batch = gpuArray(data(:,batchind));
%         compressor = gpuArray(compressor);
        decompressor = gpuArray(decompressor);
    else
        batch = data(:,batchind);
    end
    switch method.sccodemethod
        case 'fredkin'
            if method.flatten
                batch = compress_fredkin(batch,method);
            else
                batch = compress_fredkin_conv(batch,method);
            end
        case 'gxor'
            if method.flatten
                batch = compress_gxor_full(batch,method);
            else
                batch = compress_gxor_conv(batch,method);
            end
        case 'gxor-conv2d'
            [batch,net] = gxor_conv_2d_infer(batch,method);
            compressor = cat(1,compressor,{net});
        case 'MCL'
            [compressor,decompressor,batch,prob] = conditionalinfer(batch,compressor,decompressor,method.trainer,verbose);
        case 'lmcl-hyb'
            [batch,net] = h_lmcl(batch,method);
        case 'kl-mean'
            batch = batch ./sum(batch,1);
            compressor = mean(batch(:,1:4500),2);
            imdb.compressors{curlabel} = compressor;
    end
    data(:,batchind) = gather(batch);
end
end

