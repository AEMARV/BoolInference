function [ ent ] = entropy(p )
%ENTROPY Summary of this function goes here
%   Detailed explanation goes here
ent = -p.*log2(p) - (1-p).*log2(1-p);
ent(isnan(ent)) = 0;
end

