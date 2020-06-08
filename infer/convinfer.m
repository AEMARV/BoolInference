function [ bin ] = convinfer( bin)
%CONVINFER Summary of this function goes here
%   Detailed explanation goes here
%% Resize
bin = reshape(bin,32,32,3,8,[]);
%% General Parameters
EpochNum = 100;
valnum = 500;
sampdim = ndims(bin);
trainind = 1:size(bin,sampdim)-valnum-1;
valind = size(bin,sampdim)-valnum:size(bin,sampdim);
isreverse = false;
Numbits = numel(bin)./size(bin,sampdim);
rev = false;
enttrain = 0;
entprev = -1;
%% TRAIN
for epoch = 1 : EpochNum
    %      isreverse = ~isreverse;
%     bin = convxor(bin,1,sampdim,trainind,valind,isreverse);
% %     %     bin = convxor(bin,1,sampdim,trainind,valind,~isreverse);
% %     
%     bin = convxor(bin,2,sampdim,trainind,valind,isreverse);
% %     %     bin = convxor(bin,2,sampdim,trainind,valind,~isreverse);
% %     
%     bin = convxor(bin,3,sampdim,trainind,valind,isreverse);
%      bin(:,:,1:3,:,:) = bin(:,:,randperm(3),:,:);
%     bin = convxor(bin,3,sampdim,trainind,valind,~isreverse);
% %     
%      bin = convxor(bin,4,sampdim,trainind,valind,isreverse);
%     bin(:,:,:,1:8,:) = bin(:,:,:,randperm(8),:);
%     bin = convxor(bin,4,sampdim,trainind,valind,~isreverse);
%       bin = NLxor(bin,1,trainind,valind,sampdim,rev);
%      bin = NLxor(bin,2,trainind,valind,sampdim,rev);
%       bin = NLxor(bin,3,trainind,valind,sampdim,rev);
% if entprev ==enttrain
      bin = NLxor(bin,4,trainind,valind,sampdim,rev);
      bin = NLxor(bin,3,trainind,valind,sampdim,rev);
% end
%     rev = ~rev;
    %% Verbose
    entprev = enttrain;
    probtrain = mean(bin(:,:,:,:,trainind),sampdim);
    enttrain = entropy(probtrain);    
    enttrain = sum(enttrain(:),'omitnan');
    
    % val
    probval = mean(bin(:,:,:,:,valind),sampdim);
    entval = entropy(probval);
    entval = sum(entval(:));
    
    % printing
    fprintf('Epoch %d/%d ',epoch,EpochNum);
    fprintf('EntropyTrain %.10f -',enttrain./Numbits);
    fprintf('EntropyVal %.10f -',entval./Numbits);
    fprintf('\n');
    imshow(double(bin(:,:,:,1,randperm(size(bin,sampdim),1))));
end

end

