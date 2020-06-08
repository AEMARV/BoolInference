function [ bin,dum] = h_lmcl(bin,method)
%LMCL Summary of this function goes here
%   Detailed explanation goes here
dum = [];
valnum = 500;
bitnum = size(bin,1);
trainind = 1:size(bin,2)-valnum-1;
trainindlogical = false(1,size(bin,2));
trainindlogical(trainind) = true;
valind = size(bin,2)-valnum:size(bin,2);
maxiters = 100;
epochnum = 100;
epoch = 1;
[~] = extractmeans(bin);
[binp,p] = bindecim(mean(bin,2));

%visualize(bin);
for epoch = 1:100
for iters = 1:4500
[c,creal,bin(:,trainindlogical)] = contradicts(bin(:,trainindlogical));
% creal(creal==0) = nan;
% [m,sender] = min(creal,[],1);
% receiver = creal>0;
cind= find(c);
if numel(cind)<2
    continue;
end
 sender = randperm(numel(cind),1);
% [~,sender] = max(creal,[],1);
 sender = cind(sender);
c(sender) = false;
bin(c,:) = xorh(bin(sender,:),bin(c,:),false);
%% verbose
        enttrain = mean(bin(:,trainind),2);
        enttrain = entropy(enttrain);
        enttrain = sum(enttrain(:));
        probbittrain = mean(mean(bin(:,trainind),2),1);
        probrest = mean(bin(:,valind),2);
        entrest = entropy(probrest);
        entrest = sum(entrest,'omitnan');
        probbitval = mean(mean(bin(:,valind),2),1);
        fprintf('Epoch:%d Iters:%d%%',epoch,iters);
        fprintf('EntropyTrain %.10f -',enttrain./size(bin,1));
        fprintf('EntropyVal %.6f -',entrest./size(bin,1));
%         fprintf('Depth %d -',depth);
        %fprintf('EntropyUpTrain %.6f -',entropy(probbittrain));
        %fprintf('EntropyUpVal %.6f -',entropy(probbitval));
        fprintf('\n');        
        im = reshape(bin(:,3),[32,32,3]);
        imshow(double(im(:,:,:)),[],'InitialMagnification','fit');
        drawnow update;
    
end
end
end


function [c,creal,bin] = contradicts(bin)
a = randperm(size(bin,2),2);
%a = [1,2];
bin1 = bin(:,a(1));
bin2 = bin(:,a(2));
c = xorh(bin1,bin2,true);
creal = c;
% c = sample(c);
c = c>=mean(c(:));
end
function s = sample(p)
s = rand(size(p));
s = s<p;
end
%%EXPERIMENTAL
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function []= visualize(bin)
bin = reshape(bin,32,32,3,[]);
pinit = mean(bin,4);
exlenght = mean(-log2(bin),4);
% l1norm = sum(p(:));
% lambda = 1./l1norm;
% lambda = 1-lambda;
for lambda1 = 0.01:0.01:1
    lambda = 1./exlenght;
    p = pinit;
for i = 1 : 500
    if (i ==1)
        p = p./lambda;
        bits = floor(p);
        p = p - bits;
        
        imshow(double(bits),[],'InitialMagnification','fit');
        drawnow ;
        continue;
    end
        p = p./(1-lambda);
        bits = floor(p);
        p = p - bits;
        
        imshow(double(bits),[],'InitialMagnification','fit');%title(['lambda:',num2str(lambda)]);
        drawnow ;
end
end
end
