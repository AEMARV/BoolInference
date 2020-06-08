function [ bin] = compress_gxor_conv( bin,method )
valnum = 500;
bin = reshape(bin,32,32,3,8,[]);
bitnum = size(bin);
bitnum = prod(bitnum(1:4));
trainind = 1:size(bin,5)-valnum-1;
trainlog = false(1,size(bin,5));
trainlog(trainind) = true;
valind = size(bin,5)-valnum:size(bin,5);
maxiters = 1;
epochnum = 10000;
depth =0;
for epoch = 1 : epochnum
    
    for dim = 1 : 4
        [bin,preshape,newtrainlog] = reshapefullfactors(bin,false,[],trainlog);
        for iters = 1 :maxiters
            %             [bin,newtrainlog] = reshape_per_dim(bin,trainlog,dim);
%             [bin,dimpermind] = permuterepdim(bin);
            %             I = randperm(size(bin,1),size(bin,1));
            %             bin(:,:) = bin(I,:);
            %             [~,Iinv] = sort(I);
            
            
            bitmask = or(bin(:,1),true);
            mask = bin(1,:);
            mask(:)= true;
            %             if ~flag
            %                 samps = rand(size(newtrainlog))>(100/numel(newtrainlog));
            %                 newtrainlog(samps)=  false;
            %                 flag = true;
            %             else
            %                 newtrainlog(samps)=  false;
            %             end
            
            [bin,depth,mask,newtrainlog] = gxor(bin,mask,newtrainlog,method);
            
%             [bin] = permuterepdim(bin,dimpermind);
            %              bin(:,:) = bin(Iinv,:);
%             bin = reshape_per_dim_inv(bin,dim);
            if depth ==0
                break;
            end
            
        end
        [bin] = reshapefullfactors(bin,true,preshape);
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
            imnum = 1;
            im = bin(:,:,:,:,imnum);
            im = sum(im.*(2.^reshape((-1:-1:-8),1,1,1,8)),4);
            imshow(double(im(:,:,:)),[],'InitialMagnification','fit');
            drawnow update;
    end
end
end
function [bin,ind] = permutedim(bin,isrev,ind)
if ~isrev
    ind = randperm(ndims(bin)-1,ndims(bin)-1);
    bin = permute(bin,[ind, ndims(bin)]);
    ind = [ind,ndims(bin)];
else
    bin = ipermute(bin,ind);
end
end
function [bin,params,newtrainlog] = reshapefullfactors(bin,isrev,params,trainlog)
if ~isrev
    params = [];
    params.sz = size(bin);
    sz = size(bin);
    factors = [];
    for i = 1 : ndims(bin)-1
        factors = [factors,factor(sz(i))];
    end
    bin = reshape(bin,[factors,sz(end)]);
    [bin,params.factorpermind] = permutedim(bin,false);
    temp= params.factorpermind;
    temp = temp(1:end-1);
    factors = factors(temp);
    numfacs = randperm(numel(factors),1);
    finalrepsize = prod(factors(1:numfacs));
    params.reshapelastsz= size(bin);
    bin = reshape(bin,finalrepsize,[]);
    newtrainlog = gpuArray.false(size(bin(1,:)));
    newtrainlog(1:sum(trainlog)*(numel(newtrainlog)/numel(trainlog))) = true;
else
    bin = reshape(bin,params.reshapelastsz);
    bin = permutedim(bin,true,params.factorpermind);
    bin = reshape(bin,params.sz);
end
end
function [bin,params,newtrainlog] = reshapefactors(bin,isrev,params,trainlog)
if ~isrev   
   
    params = [];
    params.sz = size(bin,1);
    facs = factor(size(bin,1));
    finaldim= prod(facs(rand(size(facs))>0.5));
    if finaldim >2
        bin = reshape(bin,finaldim,[]);
        newtrainlog = bin(1,:);
        newtrainlog(:) = false;
        newtrainlog(1:sum(trainlog)*(params.sz/finaldim)) = true;
    else
        newtrainlog = trainlog;
    end
    
else
    newtrainlog = [];
    bin = reshape(bin,params.sz,[]);
end
end
function [bin,I] = permuterepdim(bin,I)
if nargin <2
    I = randperm(ndims(bin)-1);
    I = [I,ndims(bin)];
    bin= permute(bin,I);
else
    bin = ipermute(bin,I);
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