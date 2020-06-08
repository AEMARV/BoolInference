function [ means] = extractmeans( data )
%EXTRACTMEANS Summary of this function goes here
%   Detailed explanation goes here
means = [];
data = data./sum(data,1);
while(true)
    Mean = mean(data(:,rand(1,size(data,2))>0.9),2,'omitnan');
%     means = cat(2,means,Mean);
    p1 = data./Mean;
    p1 = min(p1,[],1);
    p2 = (1-data)./(1-Mean);
    p2 = min(p2,[],1);
    p2 = 1;
    p = min(p1,p2);
    p = p1;
    data = data - p.*Mean;
    data = data./(1-p);
    data(data>1) = 1;
    data(data<0) = 0;
    figure(1);
    im =data;
    im_show = reshape(im(:,1),[32,32,3]);
        imshow(double(im_show)*1000,[],'InitialMagnification','fit');
        drawnow update;
    fprintf('Ent: %.8f-----',entropy(data));
    fprintf('AVGp: %.8f',sum(p==0,2));
    fprintf('\n');
    
    
end

end
function ent = entropy(data)
 ent = -data.*log2(data) - (1-data).*log2(1-data);
 ent = sum(ent(:),'omitnan');
end

