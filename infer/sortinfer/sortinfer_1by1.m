function [bin,done] = sortinfer_1by1( bin,trainlog )
%SORTINFER_1BY1 Summary of this function goes here
%   Detailed explanation goes here
done = false;
[maxdiff,Ind,path,swap1,swap2,bin(:,trainlog)] = choosebitfull(bin(:,trainlog));

if maxdiff ==0
    done = true;
    return
end
mask = bin(Ind,:);
if ~path
    mask = ~mask;
end
s1 = bin(swap1,mask);
bin(swap1,mask) = bin(swap2,mask);
bin(swap2,mask) = s1;

end

