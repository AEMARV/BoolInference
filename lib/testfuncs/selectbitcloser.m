function [ sender,receiver ] = selectbitcloser(prob)
%SELECTBITCLOSER Summary of this function goes here
%   Detailed explanation goes here
prob = prob.*log2(prob) + (1-prob).*log2(1-prob);
[~,sender] = min(prob);
dists = abs(prob(sender) - prob);
dists(sender) = inf;
[~,receiver]  = min(dists); 

end

