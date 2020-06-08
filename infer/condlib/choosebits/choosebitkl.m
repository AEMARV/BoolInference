function [ bin,trainlog,mask,bitmask,index ] = choosebitkl( bin,trainlog,mask,bitmask )
%GXORFULL Summary of this function goes here
%   Detailed explanation goes here
method = 'kl';

thismask = mask&trainlog;
bitnum = sum(bitmask,1);
Sumperdata1 = sum(bin(bitmask,thismask),1);
S=(sum(bin(bitmask,thismask),2).*(bitnum-1));
SALL = (bitnum-1).*sum(thismask);
SBAR = SALL - S;

s1_1 =  sum((Sumperdata1-1).*bin(bitmask,thismask),2);
s1_0 = S - s1_1;
% p1 = s1_1./(sum(bin(bitmask,thismask),2).*(bitnum-1));
p1 = s1_1./S;
s0_1 =  sum((Sumperdata1).*(~bin(bitmask,thismask)),2);
% s0_0 = (sum((~bin(bitmask,thismask)),2).*(bitnum-1)) - s0_1;
s0_0 = SBAR - s0_1;
p0 = s0_1./SBAR;
% p0 = s0_1./(sum((~bin(bitmask,thismask)),2).*(bitnum-1));
p = (s1_1 + s0_1)./((s1_1 + s0_1)+(s1_0 + s0_0));
switch method
    case 'maxprob'
        probbit = mean(bin(bitmask,thismask),2);
        [~,indexrel] = min(probbit,[],1);
        path = 1;
    case 'kl'
        probbit = mean(bin(bitmask,thismask),2);
        [divg,path,kl1,kl0] = calcklbound(p,p0,p1,1-probbit,probbit);
        [divg,indexrel] = max([kl0,kl1],[],1);
        [~,path] = max(divg,[],2);
        indexrel = indexrel(path);
        path = path ==2;
%         [divmax,indexrel] = max(divg,[],1);
%         path = path(indexrel);
%         probcurbit = probbit(indexrel);
%         path = p1(indexrel)>p0(indexrel);
    case 'fullkl'
        probbit = mean(bin(bitmask,thismask),2);
        for i = 1 : numel(probbit)
            probcond = probbit(i);
            
        end
    case 'maxdif'
        diff1 = s1_1 - s1_0;
        diff0 = s0_1 - s0_0;
        [m1,indexrel1] = max(diff1,[],1);
        [m0,indexrel0] = max(diff0,[],1);
        [~,path] = max([m0;m1],[],1);
        path = path -1;
        if path ==0
            indexrel = indexrel0;
        else
            indexrel = indexrel1;
        end
end
Indall = find(bitmask);
index = Indall(indexrel);
bitmask(index) = false;
if path%( p1(indexrel)>p0(indexrel))
    mask = mask& bin(index,:);
else
    mask = mask&(~bin(index,:));
end
end
function [rhoM,path,kl2,kl1] = calcklbound(pi,pi1,pi2,p1,p2)
kl1 = kldiv(pi1,pi);
kl2 = kldiv(pi2,pi);
rhoM = p1.*kl1+ p2.*(kl2);
path = kl2>kl1;
end
function div = kldiv(p1,p2)
div = p1.*log2(p1./p2) + (1-p1).*log2((1-p1)./(1-p2));
div(isnan(div)) = 0;
end
