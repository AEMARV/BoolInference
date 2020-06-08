function [ x3] = xorh( x1,x2,isid )
%XORH Summary of this function goes here
%   Detailed explanation goes here
x1b = ptobin(x1);
x2b = ptobin(x2);

if isid
% x3 = x1.*(1-x2) + x2.*(1-x1);
%return;
bit1 = rand(1,size(x1b,5))>0.5;
bit2 = rand(1,size(x1b,5))>0.5;
bit1 = find(bit1,1);
bit2 = find(bit2,1);
x3 = xor(x1b(:,:,:,:,bit1),x2b(:,:,:,:,bit2));
return;
end

x1b = ptobin(x1);
x2b = ptobin(x2);
bitnum1 = size(x1b,5);
bitnum2 = size(x2b,5);
bitnummin = min(bitnum1,bitnum2);
x1b = xor(x1b(:,:,:,:,1:bitnummin),x2b(:,:,:,:,1:bitnummin));
x3 = bintop(x1b);

end

function xb = ptobin(xr)
xrs = xr;
xb = gpuArray.false([size(xr),1,1,8]);
for bitplane = 1:8
    xr = xr.*2;
    cb = floor(xr);
    xr= xr - cb;
    cb = cb==1;
    cb(xrs==1) = 1;
    xb(:,:,:,:,bitplane)= cb;
    check = xr==0;
    if all(check(:))
        break
    end
end
end
function xr = bintop(xb)
bitnum = size(xb,5);
coefs = 1:bitnum;
coefs = reshape(coefs,1,1,1,1,bitnum);
coefs = 2.^(-coefs);
xr = sum(xb.*coefs,5);
xr = double(xr);
end