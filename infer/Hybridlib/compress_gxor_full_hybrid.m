function [ bin ] = compress_gxor_full_hybrid(bin,method )
%COMPRESS_GXOR_FULL_HYBRID Summary of this function goes here
%   Detailed explanation goes here
valnum = 500;
bitnum = size(bin,1);
trainind = 1:size(bin,2)-valnum-1;
trainindlogical = false(1,size(bin,2));
trainindlogical(trainind) = true;
valind = size(bin,2)-valnum:size(bin,2);
maxiters = 100;
epochnum = 100;
for epoch = 1 : epochnum
    thistrain = trainindlogical & (rand(size(trainindlogical))>0);    
    for iters = 1 :maxiters
        bitmask(:) = true(size(bin,1));
        [bin,depth] = gxor_hyb(bin,bitmask,thistrain,method);
        %% verbose
        enttrain = mean(bin(:,trainind),2);
        enttrain = entropy(enttrain);
        enttrain = sum(enttrain(:));
        probbittrain = mean(mean(bin(:,trainind),2),1);
        probrest = mean(bin(:,valind),2);
        entrest = entropy(probrest);
        entrest = sum(entrest,'omitnan');
        probbitval = mean(mean(bin(:,valind),2),1);
        fprintf('Epoch:%d Iters:%d%%',epoch,iters);
        fprintf('EntropyTrain %.10f -',enttrain./size(bin,1));
        fprintf('EntropyVal %.6f -',entrest./size(bin,1));
        fprintf('Depth %d -',depth);
        %fprintf('EntropyUpTrain %.6f -',entropy(probbittrain));
        %fprintf('EntropyUpVal %.6f -',entropy(probbitval));
        fprintf('\n');        
        im = reshape(bin(:,3),[32,32,3,8]);
        im = sum(im.*(2.^reshape((-1:-1:-8),1,1,1,8)),4);
        imshow(double(im(:,:,:)),[],'InitialMagnification','fit');
        drawnow update;
    end
end
end
