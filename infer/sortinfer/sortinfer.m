function [ bin,depth ] = sortinfer( bin,mask,bitmask,trainlog,depth )
%SORTINFER Summary of this function goes here
%   Detailed explanation goes here
if depth > 2
%     return;
end
depth = depth +1;
depth0 = 0;
depth1 = 0;
if isempty(mask)
    return;
end
if sum(mask&trainlog)<1
    return;
end
probbitwise = mean(bin(bitmask,mask&trainlog),2);
[~,I] = sortmini(probbitwise);
if ~isequal(I,(1:sum(bitmask))')
    Index = find(bitmask);
    bin(bitmask,mask) = bin(Index(I),mask);
    if depth ~=1
     return;
    end
end
probbitwisenew = mean(bin(:,mask&trainlog),2);
entbitwise = entropy(probbitwisenew);
bitmask = bitmask &(entbitwise~=0);
% Cond bit
cand_condbits  = find(bitmask);
if isempty(cand_condbits)| numel(cand_condbits)<3
    return;
end
[bin,trainlog,~,~,Indcand] = choosebitkl(bin,trainlog,mask,bitmask);
mask1 = mask & bin(Indcand,:);
mask0 = mask & ~bin(Indcand,:);
bitmask(Indcand) = false;
probcand = mean(bin(Indcand,mask&trainlog),2);   
[bin,depth1] = sortinfer(bin,mask1,bitmask,trainlog,depth);   
[bin,depth0] = sortinfer(bin,mask0,bitmask,trainlog,depth);

depth = max(depth1,depth0);

end
function [bin,trainlog,dum1,dum2,Indcand] = choosebitent(bin,trainlog,mask,bitmask)
dum1 = 0;
dum2 = 0;
probbitwise = mean(bin(bitmask,mask&trainlog),2);
ents = entropy(probbitwise);
[~,Irel] = max(ents,[],1);
Indcand = find(bitmask);
Indcand = Indcand(Irel);
end
function [bin,trainlog,dum1,dum2,Indcand] = choosebitrand(bin,trainlog,mask,bitmask)
dum1 = 0;
dum2 = 0;
Indcand = find(bitmask);
Irel = randperm(numel(Indcand),1);
Indcand = Indcand(Irel);
end


