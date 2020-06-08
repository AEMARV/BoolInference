function [MAX_DIF,ind,path,swap1,swap2,bin ] = choosebitfull(bin)
%CHOOSEBITFULL Summary of this function goes here
%   Detailed explanation goes here
prob = mean(bin,2);
bitinds = 1:(size(bin,1));
bitmask = gpuArray.true(numel(bitinds),1);
MAX_DIF = 0;
ind = nan;
path = nan;
swap1 = nan;
swap2 = nan;
for i = 1: numel(bitinds)
    mask_new = bin(i,:);
    probbiti = prob(i);
    bitmask(i)= false;
    bitmask_inds = find(bitmask);
    prob1 = mean(bin(bitmask,mask_new),2);
    prob0 = (prob(bitmask) - prob1.*probbiti)./(1-probbiti);
    [m1,swap1_1_tmp,swap2_1_tmp] = maxdiff2(prob1);
    [m0,swap1_0_tmp,swap2_0_tmp] = maxdiff2(prob0);
    if (m1*probbiti) >MAX_DIF
        path = 1;
        ind = i;
        MAX_DIF = (m1*probbiti);
        swap1 = bitmask_inds(swap1_1_tmp);
        swap2 = bitmask_inds(swap2_1_tmp);
        
    end
    if (m0*(1-probbiti))>MAX_DIF
        path = 0;
        ind = i;
        MAX_DIF = (m0*(1-probbiti));
        swap1 = bitmask_inds(swap1_0_tmp);
        swap2 = bitmask_inds(swap2_0_tmp);
    end
    bitmask(i) = true;
end

end
function [m,s1,s2] = maxdiff(x)
diffs = diff(x);
diffs(diffs>0) = 0;
[m,s1] = min(diffs,[],1);
s2 = s1+1;
m = abs(m);
end
function [m,s1,s2] = maxdiff2(x)
diffs = x - x';
[c,r] = meshgrid(1:size(x,1),1:size(x,1));
diffs(r>c) = 0;
[m,s2] = max(diffs,[],2);
[m,s1] = max(m,[],1);
s2 = s2(s1);
if m<=0
    s1 = 1;
    s2 = 1;
end

end
