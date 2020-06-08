function [bin,depth,mask,trainlog] = gxor(bin,mask,trainlog,method)
%CONDLINEAR Summary of this function goes here
%   Detailed explanation goes here
bitmask = true(size(bin(:,1)));
trainlog = trainlog &(rand(size(trainlog))>0);
%  bitmask = bitmask &(rand(size(bitmask))>0.5);
probbitwise = mean(bin(:,trainlog),2);
bitmask(probbitwise==0 | probbitwise==1) = false;
   bin = xor(bin,probbitwise>0.5);
   
depth =0;
while(true)
    if depth > 2
         return;
    end
    prob = mean(bin(bitmask,trainlog&mask),2);
    if(any(prob>0.5))
        [m,i] = max(prob,[],1);
        prob(:) = 0;
        prob(i) =1;
         bin(bitmask,mask) = xor(prob>0.5, bin(bitmask,mask));
        return;
    else
        if sum(mask&trainlog)<2 || sum(bitmask)<2
            depth = 0;
            return;
        end
        switch method.trainer.bitselect
            case 'mixing'
                [bin,trainlog,mask,bitmask] = choosebitkl(bin,trainlog,mask,bitmask);
            case 'random'
                [bin,trainlog,mask,bitmask] = choosebitrand(bin,trainlog,mask,bitmask);
            case 'fullinvestigate-not'
                [bin,trainlog,mask,bitmask] = choosebitrand(bin,trainlog,mask,bitmask);
        end
        depth = depth +1;
    end
    
end

end
function [bin,trainlog,mask,bitmask] = choosebitrand(bin,trainlog,mask,bitmask)
    bitinds = find(bitmask);
    indrel = randperm(numel(bitinds),1);
    Ind = bitinds(indrel);
    bitmask(Ind) = false;
    mask(xor(bin(Ind,:),rand>0.5)) = false;
end
function [bin,trainlog,mask,bitmask] = choosebit(bin,trainlog,mask,bitmask)
thismask = mask&trainlog;
proball = mean(mean(bin(bitmask,thismask),1),2);
probperbit = mean(bin(bitmask,thismask),2);
% condbits = probperbit<proball;
condbits = probperbit>10;
condbits(1:ceil(numel(probperbit).*0.5)) = true;
condbits = xor(condbits,rand>0.5);
obsbits = ~condbits;
bitinds = find(bitmask);
condbitsind_global = bitinds(condbits);
obsbitsind_global = bitinds(obsbits);

probcondbits = mean(probperbit(condbits),1);
probobsbits = mean(probperbit(obsbits),1);

probcondpersample = mean(bin(condbitsind_global,:),1);

mask1 = probcondpersample > probcondbits;
mask0 = ~mask1;

% choosing mask
probmask1 = mean(mean(bin(obsbitsind_global,mask1&thismask),1),2);
probmask0 = mean(mean(bin(obsbitsind_global,mask0&thismask),1),2);

% if probmask1 >= probmask0
if rand > 0.5
    mask = mask&mask1;
else
    mask = mask&mask0;
end
bitmask(condbitsind_global) = false;


end
function [bin,trainlog,mask,bitmask,ind,path] = choosebitfullbitaccess(bin,trainlog,mask,bitmask)
thismask = mask&trainlog;
Sumperdata= sum(bin(bitmask,mask&trainlog),1);
% deltasum = sum(bin(bitmask,mask&trainlog),1);
% %  deltasum = 2.*sum(bin(bitmask,mask&trainlog),1) - numel(find(bitmask));
%  grad1 = sum(bin(bitmask,mask&trainlog).*(deltasum-1),2);
%  grad0 = sum((~bin(bitmask,mask&trainlog)).*(deltasum+1),2);
%  [mgrad1,ind1] = max(grad1,[],1);
%  [mgrad0,ind0] = max(grad0,[],1);
%  if mgrad1>=mgrad0
%      indp = ind1;
%      path = true;
%  else
%      indp = ind0;
%      path = false;
%  end
%  inds = find(bitmask);
%  ind = inds(indp);

bitnum = sum(bitmask,1);
p1 =  sum((Sumperdata-1).*bin(bitmask,thismask),2);
p1 = p1./(sum(bin(bitmask,thismask),2).*(bitnum-1));

p0 =  sum((Sumperdata).*(~bin(bitmask,thismask)),2);
p0 = p0./(sum((~bin(bitmask,thismask)),2).*(bitnum-1));
diverg = abs(p1-p0);
divvergprob = diverg./sum(diverg);
divvergcumprob = cumsum(divvergprob);
R = rand;
pind = divvergcumprob<R ;
pind = find(pind,1,'last');
% [divval,pind] = max(diverg,[],1);
bitinds = find(bitmask);
ind = bitinds(pind);
pi = mean(bin(ind,thismask),2);
path= rand < (p1(pind)./(p0(pind)+p1(pind)));
% path = rand < ((pi.*p1(pind))./((pi.*p1(pind))+((1-pi).*p0(pind))));
% path = p1(pind)>=p0(pind);




% SUM = sum(SUM.*bin(bitmask,mask&trainlog),2);
% SUM = SUM - SUMperbit;
% prob = SUM./(numel(bitmask)-1);
% prob = prob./SUMperbit;
%
% SUM
end

