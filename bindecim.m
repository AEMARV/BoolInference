function [ bin,p ] = bindecim( im )
%BINDECIM Summary of this function goes here
%   Detailed explanation goes here
ims = im;
bin = [];
p = [];
while(true)
    im(im>1) = 1;
    im(im<0) = 0;
    ent = entropy(im);
    [entmax,I] = max(ent,[],1);
    if (entmax)==0
        break
    end
    curp = im(sub2ind(I,1:size(im,2)));
    curp = reshape(curp,1,size(im,2));
    curp = (curp<=0.5)+(((-1).^((curp<=0.5))).*curp);
   
    bin1 = im>0.5;
    bin = cat(2,bin,bin1);
    im = (im - (curp.*bin1))./(1-curp);
    thisp = prod(p,
    p = cat(2,p,curp);
    pc = cumprod([1,p]);
    figure(1);
    im_show = reshape(im,[32,32,3]);
        imshow(double(im_show),[],'InitialMagnification','fit');title([num2str(pc(end))]);
        drawnow update;
    figure(2)
    im_show = reshape(bin1,[32,32,3]);
        imshow(double(im_show),[],'InitialMagnification','fit');title([num2str(pc(end-1))]);
        drawnow update;

    fprintf('Entropy: %d\n',sum(entropy(pc)));
    
    
end

end
function ent = entropy(p)
ent = -p.*log2(p) - (1-p).*log2(1-p);
ent(isnan(ent)) = 0;
end

