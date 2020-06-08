function [x, I ] = sortmini( x ,direction)
%SORTMINI Summary of this function goes here
%   Detailed explanation goes here
if nargin < 2
    direction = 'ascend';
end
I = gpuArray(1:numel(x));
diffs = diff(x);
switch direction
    case 'ascend'
        diffs(diffs>0) = 0;
        [m,ind] = min(diffs,[],1);
        if(m<0)
            I(ind) = ind+1;
            I(ind+1) = ind;
        end
    case 'descend'
end
I = I';

end

