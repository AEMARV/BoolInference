function [bin,ind,path] = choosebitcondconv( bin,anchinds ,method,depth )
%CHOOSEBITCONDCONV Summary of this function goes here
%   Detailed explanation goes here
ind = [];
path = [];
if depth>5
    return
end
if isempty(bin)
    return;
end
p = mean(bin,4);



end

