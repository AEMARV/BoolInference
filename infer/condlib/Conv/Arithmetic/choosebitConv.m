function [ bin,ind,path] = choosebitConv( bin,anchor_inds ,method,depth)
%CHOOOSEBITCONV Summary of this function goes here
%   Detailed explanation goes here
ind = [];
path = [];
if depth>5
    return;
end
if isempty(bin)
    return;
end
switch method.trainer.bitselect
    case 'mixing'
       p = mean(bin,4);
       panch = mean(bin(anchor_inds(1),anchor_inds(2),anchor_inds(3),:),4);
%        panchtend = panch>0.5;
       if panch>0.5
%            warning('srehit')
       end
       if panch ==0 || panch ==1
           return;
       end
       p1 = mean(and(bin(anchor_inds(1),anchor_inds(2),anchor_inds(3),:),bin),4); 
       p0 = mean(and(bin(anchor_inds(1),anchor_inds(2),anchor_inds(3),:),~bin),4);
       p1 =p1./ p;
       p0 = p0./(1-p);
       [divg,path,kl2,kl1] = calcklbound(panch,p0,p1,1-p,p);
       
%        if method.RND1 > 0.5
%            divg(1:anchor_inds(1),:,:) = -inf;
%            
%        else
%            divg(anchor_inds(1):end,:,:) = -inf;
%            
%        end
%        if method.RND2 > 0.5
%            divg(:,1:anchor_inds(2),:) = -inf;
%        else
%            divg(:,anchor_inds(2):end,:) = -inf;
%        end
%        divg(anchor_inds(1),anchor_inds(2),anchor_inds(3)) = -inf;
        divg(anchor_inds(1),anchor_inds(2),:) = -inf;
       [m,indlin] = max(divg(:),[],1);
       ind = gpuArray.zeros(1,3);
       [ind(1),ind(2),ind(3)] = ind2sub(size(divg),indlin);
       path = p1(indlin)>p0(indlin);
       if p1(indlin)>0.5 || p0(indlin)>0.5
           return;
       else
           logtrain = squeeze(bin(ind(1),ind(2),ind(3),:)==path);
           [bin(:,:,:,logtrain),ind1,path1] = choosebitConv(bin(:,:,:,logtrain),anchor_inds,method,depth+1);
           [bin(:,:,:,~logtrain),ind2,path2] = choosebitConv(bin(:,:,:,~logtrain),anchor_inds,method,depth+1);
           
           if isempty(path1)
               path =[];
               ind = [];
               return;
           end
           ind = cat(1,ind,ind1);
           path= cat(1,path,path1);
       end
       
    otherwise 
        error('not defined')
end


end

function [rhoM,path,kl2,kl1] = calcklbound(pi,pi1,pi2,p1,p2)
kl1 = kldiv(pi1,pi);
kl2 = kldiv(pi2,pi);
rhoM = p1.*kl1+ p2.*(kl2);
path = kl2>kl1;
end
function div = kldiv(p1,p2)
if any(p1(:)>1)
%     error('sg')
    p1(p1>1) =1;
end
k1 = p1.*log2(p1./p2);
k1(isnan(k1)) = 0;
k2 = (1-p1).*log2((1-p1)./(1-p2));
k2(isnan(k2)) = 0;
div = k1 + k2;
end