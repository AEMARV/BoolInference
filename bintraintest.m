function [ output_args ] = bintraintest( dstrain,dsTEST,method )
%BINTRAINTEST Summary of this function goes here
%   Detailed explanation goes here
verbose.exist = true;
verbose.likelihood = false;
verbose.depth = 0;
CheckCode = true;
Valnum = 500;
train_data = gather(tobin(gpuArray(dstrain.data)));
TEST_data = gather(tobin(gpuArray(dsTEST.data)));
TEST_data = logical(TEST_data);
train_labels = dstrain.labels;
TEST_labels = dsTEST.labels;
wait(gpuDevice());
logprobs = [];
for i= 0: max(train_labels(:))
    traindata_i = (train_data(:,:,:,:,train_labels==i));
    trainind = 1:size(traindata_i,5)-Valnum;
    valind = size(traindata_i,5)-Valnum+1:size(traindata_i,5);
    traindata_i = cat(5,traindata_i,TEST_data);
    testind = valind(end)+1:size(traindata_i,5);
    [ CodeMatrix,CodeMatrixinv ] = bubblewrap( traindata_i,trainind,valind,testind,verbose,method );
%     [ CodeMatrix,traindata_i ] = turbodividewrapper( traindata_i,trainind,valind,testind,verbose,method );
%     [~,traintestdata_i]= xoraccv1(traintestdata_i,trainind);
%     [probs,traindata_i]= batchlearnconv(gpuArray(traindata_i),trainind,valind,testind,verbose);
    
    testbin = traindata_i(:,testind);
    probtest = calclogprob(testbin,probs);
    logprobs = cat(2,logprobs,probtest');
    
end
inferredlabels = findlabel(logprobs);
acc = TEST_labels == inferredlabels;
fprintf('Accuracy: %d',mean(acc,1));
end
function labels  = findlabel(logprobs)
logprobs = -abs(logprobs);
[~,labels] = max(logprobs,[],2);
labels = labels -1;
end
function logprob = calclogprob(testbin,probs)
    testbin = (testbin + 1)/2;
    testbin = testbin.*probs;
    testbin = testbin + ((~testbin).*(1-probs));
    testbin = -log2(testbin);
    for i = 1 : ndims(testbin)-1
        testbin = sum(testbin,i);
    end
    logprob = testbin;
    logprob = squeeze(logprob);
end


function [datai] = prepareData(datai)
%     datai = reshape(datai,[numel(datai)/size(datai,4),1,1,size(datai,4)]);
%             datai = permute(datai,[4,1,2,3]);
end
