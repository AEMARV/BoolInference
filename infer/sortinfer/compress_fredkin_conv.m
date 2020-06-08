function [ bin] = compress_fredkin_conv(bin,method )
%CONV_FREDKIN Summary of this function goes here
%   Detailed explanation goes here
valnum = 500;
bin = reshape(bin,32,32,3,8,[]);
bitnum = size(bin);
bitnum = prod(bitnum(1:4));
trainind = 1:size(bin,5)-valnum-1;
trainlog = false(1,size(bin,5));
trainlog(trainind) = true;
valind = size(bin,5)-valnum:size(bin,5);
maxiters = 200;
epochnum = 10000;
depth =0;
for epoch = 1 : epochnum
    
    for dim = 1 : 4
        flag = false;
        done = false;
        for iters = 1 :maxiters
            
            %             bitmask(:) = true;
            
            [bin,newtrainlog] = reshape_per_dim(bin,trainlog,dim);
            bitmask = or(bin(:,1),true);
            %% Initial Sort
%             if ~flag
%                 samps = rand(size(newtrainlog))>(100/numel(newtrainlog));
%                 newtrainlog(samps)=  false;
%                 flag = true;
%             else
%                 newtrainlog(samps)=  false;
%             end
            probs = mean(bin(bitmask,newtrainlog),2);
            [~,I] = sort(probs);
            [~,Iinv] = sort(I);
            bin = bin(I,:);
            
            [bin,done] = sortinfer_1by1(bin,newtrainlog);
            
%             bin = bin(Iinv,:);
            bin = reshape_per_dim_inv(bin,dim);
            % [bin,~] = blockize(bin,trainlog,2,true);
            if done
                break;
            end
            %% verbose
            enttrain = mean(bin(:,:,:,:,trainind),5);
            enttrain = entropy(enttrain);
            enttrain = sum(enttrain(:),'omitnan');
            %             probbittrain = mean(mean(bin(:,:,:,:,trainind),2),1);
            probrest = mean(bin(:,:,:,:,valind),5);
            entrest = entropy(probrest);
            entrest = sum(entrest(:),'omitnan');
            %             probbitval = mean(mean(bin(:,:,:,:,valind),2),1);
            fprintf('Epoch:%d dim:%d Iters:%d%%',epoch,dim,iters);
            fprintf('EntropyTrain %.10f -',enttrain./bitnum);
            fprintf('EntropyVal %.6f -',entrest./bitnum);
            fprintf('Depth %d -',depth);
            %         fprintf('EntropyUpTrain %.6f -',entropy(probbittrain));
            %         fprintf('EntropyUpVal %.6f -',entropy(probbitval));
            fprintf('\n');
            imnum = randperm(size(bin,5),1);
            im = bin(:,:,:,:,imnum);
            im = sum(im.*(2.^reshape((-1:-1:-8),1,1,1,8)),4);
            imshow(double(im(:,:,:)),[],'InitialMagnification','fit');
            drawnow update;
        end
    end
end
end
function [bin,trainlognew] = blockize(bin,trainlog,blocksize,isrev)
blocknum = size(bin,1)/blocksize;
if ~isrev
    bin = reshape(bin,32,32,24,[]);
    
    bin = mat2cell(bin,repmat(blocksize,1,blocknum),repmat(blocksize,1,blocknum),24,ones(1,size(bin,4)));
    bin =reshape(bin,1,[]);
    bin = cell2mat(bin);
    bin = reshape(bin,(blocksize^2)*24,[]);
    trainindnew = 1:sum(trainlog)*(blocknum^2);
    trainlognew = gpuArray.false(1,size(bin,2));
    trainlognew(trainindnew) = true;
else
    bin = rehsape(bin,blocksize,blocksize,24,[]);
    bin = mat2cell(bin,blocksize,blocksize,24,ones(1,size(bin,4)));
    bin =reshape(bin,blocknum,blocknum,1,size(bin,4));
    bin = cell2mat(bin);
    bin =reshape(bin,32,32,3,8,[]);
end
end

function [bin,trainlognew] = reshape_per_dim(bin,trainlog,dim)
trainlogfactor = numel(bin)/(size(bin,ndims(bin)).*(size(bin,dim)));
permutearr = 1:ndims(bin);
permutearr(1) = dim;
permutearr(dim) = 1;
bin = permute(bin,permutearr);
bin = reshape(bin,size(bin,1),[]);
trainlognew = false(1,size(bin,2));
trainlognew(1:sum(trainlog)*trainlogfactor) = true;
end
function [bin] = reshape_per_dim_inv(bin,dim)
sz = [32,32,3,8];
tmp = sz(dim);
sz(dim) = sz(1);
sz(1) = tmp;
bin = reshape(bin,[sz,numel(bin)/prod(sz)]);
permutearr = 1:ndims(bin);
permutearr(1) = dim;
permutearr(dim) = 1;
bin = permute(bin,permutearr);
end
