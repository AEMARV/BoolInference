function [covarmat,sender,receiver ] = pcasenderrec(covarmat,probpol,prevrec )
%PCASENDERREC Summary of this function goes here
%   Detailed explanation goes here

% covarmat = abs(covarmat);
% probpol = abs(probpol);
% covarmat = ((1-covarmat)./2).*log2(((1-covarmat)./2)) + ((1+covarmat)./2).*log2(((1+covarmat)./2));
% covarmat(isnan(covarmat)) = 0;
% covarmat = covarmat ~=0;
covarmat(sub2ind(size(covarmat),1:size(covarmat,1),1:size(covarmat,2))) = 0;
probpol = ((1-probpol)./2).*log2(((1-probpol)./2)) + ((1+probpol)./2).*log2(((1+probpol)./2));
probpol(isnan(probpol)) = 0;
% covarmat = covarmat - probpol;
probpol = 0;
%  covarmat= max(covarmat,0);
% [m,I] = max(covarmat(:));
covarmat = abs(covarmat);
if isempty(prevrec) || ~any(prevrec)
sender = sum(covarmat,2);
else
sender = sum(covarmat(prevrec,:)>rand(numel(find(prevrec)),size(covarmat,2)),2);
end
[m,sender] = max(sender);
receiver = logical(covarmat(sender,:));
% receivers= sum(covarmat,1);
% [~,receiver] = max(receivers);
%  sender = logical(covarmat(:,receiver));
% receiver = receivers>0;
% receiver = receivers > rand(size(receivers));
% [sender,receiver] = ind2sub(size(covarmat),I);

end
