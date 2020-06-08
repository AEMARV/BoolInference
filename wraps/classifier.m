function [ ] = classifier( data,CDFStructs,ProbEvalfun )
%CLASSIFIER Summary of this function goes here
%   Detailed explanation goes here
labels = data.labels;
labels = labels +1;
data = data.data;
data = gather(tobin(data,false));
data = reshape(data,32,32,24,[]);
logprobs = [];
for curlabel = min(labels):max(labels)
     net = CDFStructs{curlabel};
     [logprobsthis,~] = netEval(data,net);
       logprobsthis = logprobsthis-abs(log2(mean(2.^-logprobsthis,1)));
     logprobs = cat(2,logprobs,logprobsthis);
     
     [~,inf_labels] = min(logprobs,[],2);
inf_labels = squeeze(inf_labels);
corrects = inf_labels==labels;
acc = mean(corrects,1);
fprintf('Acc = %d\n',acc);
end
[~,inf_labels] = min(logprobs,[],2);
inf_labels = squeeze(inf_labels);
corrects = inf_labels==labels;
acc = mean(corrects,1);
fprintf('Acc = %d\n',acc);
end

