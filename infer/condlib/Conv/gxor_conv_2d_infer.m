function [ bin,net] = gxor_conv_2d_infer(bin,method )
%GXOR_CONV Summary of this function goes here
%   Detailed explanation goes here
%% Initial Params
valnum = 500;
origsize = size(bin);
bin = reshape(bin,32,32,24,[]);
trainind = 1:size(bin,4)-valnum-1;
trainlog = false(1,size(bin,4));
trainlog(trainind) = true;
valind = size(bin,4)-valnum:size(bin,4);
epochnum = 40;
anch_ind = [16,16];
verbose.epoch =0;
verbose.anchind = anch_ind;
talk(bin,trainind,valind,verbose);
thistrainlog = trainlog;
net = struct();
net.weights = {};
bintemp = bin;
for epoch =  1 : epochnum
%      thistrainlog = trainlog;
%      thistrainlog(randperm(numel(find(trainlog)),4000)) = false;
    %% Standardize bin
        probspat = mean(mean(bin,4),3);
        entspat = entropy(probspat);
        entspat(entspat==0) = inf;
        [~,lin_ind] = min(entspat(:),[],1);
        [anch_ind(1),anch_ind(2)] = ind2sub(size(entspat),gather(lin_ind));
%            anch_ind = gpuArray([randperm(32,1),randperm(32,1)]);
%        anch_ind = 2- anch_ind+16;
      
%           bias = (mean(bin(anch_ind(1),anch_ind(2),:,thistrainlog),4)>0.5);
            bias =(mean(mean(mean(bin(:,:,:,thistrainlog),1),2),4)>0.5);
%            bias = mean(bin,4)>0.5;
%             bias = 0;
           bin = xor(bin, bias);
    %     [~,linind] = max(prob(:),[],1);
    %     [anch_ind(1),anch_ind(2)] = ind2sub(size(prob),linind);
    %%
    
    rel_inds_cell = cell(1,size(bin,3));
    path_cell = cell(1,size(bin,3));
    method.RND1 = rand;
    method.RND2 = rand;
    for chan_ind = 1 : size(bin,3)
        [bin(:,:,:,thistrainlog),rel_inds,paths] = calc_cause_ind(bin(:,:,:,thistrainlog),[anch_ind,chan_ind],method);
        rel_inds_cell{chan_ind} = rel_inds;
        path_cell{chan_ind} = paths;
        % inds are the indices of the bits for conditioning
        % paths are the signs
    end
    net.weights = cat(1,net.weights,{rel_inds_cell,path_cell,bias});
    [bin,diffbin] = gxor_conv_2d(bin,rel_inds_cell,path_cell);
    %     bin = xor(bin,diffbin);
    %%  Verbose
    verbose.epoch = epoch;
    verbose.anchind = anch_ind;
    talk(bin,trainind,valind,verbose);
    figure(2);
    visualize(diffbin,1);
    figure(1);
end
prob = mean(bin,4);
net.prob = prob;
[~,bintemp] = netEval(bintemp,net);
bin = reshape(bin,origsize);
end
function [] = talk(bin,trainind,valind,verbose)
%% verbose
sz = size(bin);
bitnum = prod(sz(1:end-1));
probtrain = mean(bin(:,:,:,trainind),4);
probtrain = mean(mean(probtrain,1),2);
enttrain = entropy(probtrain);
% enttrainanch = enttrain(verbose.anchind(1),verbose.anchind(2),:);
enttrain = mean(enttrain(:),'omitnan');
%             probbittrain = mean(mean(bin(:,:,:,:,trainind),2),1);
probrest = mean(bin(:,:,:,valind),4);
probrest = mean(mean(probrest,1),2);
entval = entropy(probrest);
% entvalanch = entval(verbose.anchind(1),verbose.anchind(2),:);

entval = mean(entval(:),'omitnan');
%             probbitval = mean(mean(bin(:,:,:,:,valind),2),1);
fprintf('Epoch:%d  ',verbose.epoch);
fprintf('EntropyTrain %.10f',enttrain);
fprintf('EntropyVal %.6f ',entval);
%         fprintf('EntropyUpTrain %.6f -',entropy(probbittrain));
%         fprintf('EntropyUpVal %.6f -',entropy(probbitval));
fprintf('\n');
imnum = randperm(size(bin,4),1);
imnum = 1;
visualize(bin,imnum);
end
function [] = visualize(bin,imnum)
im = bin(:,:,:,imnum);
im = reshape(im,32,32,3,8);
im = sum(im.*(2.^reshape((-1:-1:-8),1,1,1,8)),4);
imshow(double(im(:,:,:)),[],'InitialMagnification','fit');
drawnow update;
end
