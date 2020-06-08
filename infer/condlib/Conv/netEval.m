function [ logprobs,bin] = netEval(bin,net )
%[ logprobs,bin] = netEval(bin,net )
%NETEVAL Summary of this function goes here
%   Detailed explanation goes here
for layer = 1 : size(net.weights,1)
    thisWeights = net.weights(layer,:);
    rel_inds_cell = thisWeights{1};
    path_cell = thisWeights{2};
    bias = thisWeights{3};
    bin = xor(bin,bias);
    [bin,~] = gxor_conv_2d(bin,rel_inds_cell,path_cell);
end
infered_probs = net.prob;
infered_probs = mean(mean(infered_probs,1),2);
cur_probs = mean(bin,1);
cur_probs = mean(cur_probs,2);
p1 = cur_probs.*log2(cur_probs./infered_probs);
p0 = (1-cur_probs).*log2((1-cur_probs)./(1-infered_probs));
% p1 = log2(bin.*infered_probs);
% p0 = log2(~bin.*(1- infered_probs));
p1(isnan(p1)) = 0;
p0(isnan(p0)) = 0;
logprobs = sum(p1+p0,3);
logprobs = squeeze(logprobs);
% logprobs = (sum(sum(sum(p1+p0,1),2),3));
% logprobs = squeeze(logprobs);
end

