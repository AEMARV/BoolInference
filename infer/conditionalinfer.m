function [ codem,codeminv,bin,sampprob,binent ] = conditionalinfer(bin,codem,codeminv,method,verbose )
%CONDITIONALINFER Summary of this function goes here
%   Detailed explanation goes here
%% General Variables
SAMPSZ = size(bin,2);
valnum = 500;
trainind = 1:size(bin,2)-valnum-1;
valind =  size(bin,2)-valnum:size(bin,2);
EPOCHNUM = 100;
%% Training
for epoch = 1:EPOCHNUM
       %% VERBOSE
    probtrain = mean(bin(:,trainind),2);
    probval = mean(bin(:,valind),2);
    enttrain = entropy(probtrain);
    entval = entropy(probval);
    enttrain = sum(enttrain(:),'omitnan');
    entval = sum(entval(:),'omitnan');
    fprintf('Epoch:%d',epoch);
    fprintf('EntropyTrain %.6f -',enttrain/size(bin,1));
    fprintf('EntropyVal %.6f -',entval/size(bin,1));
    fprintf('\n');
    
    
    bin = condinfer(bin,trainind);
 
end


end
function bin = condinfer(bin,trainind)
TR_SZ = numel(trainind);
%% General Variables
probtrain = mean(bin(:,trainind),2);
ent = entropy(probtrain);
trainorder = randperm(numel(trainind));
samporder = trainind(trainorder);

%% Condition
if TR_SZ>2
    bitind = selectbit(probtrain,ent);
    bitselect = true(size(bin,1),1);
    bitselect(bitind) = 0;
    setainds = find(bin(bitind,:));
    setbinds = find(~bin(bitind,:));
    traininda = intersect(setainds,trainind);
    trainindalocal = 1:numel(traininda);
    trainindb = intersect(setbinds,trainind);
    trainindblocal = 1:numel(trainindb);
    % divide
    bin(bitselect,bin(bitind,:)) = condinfer(bin(bitselect,bin(bitind,:)),trainindalocal);
    bin(bitselect,~bin(bitind,:)) = condinfer(bin(bitselect,~bin(bitind,:)),trainindblocal);
    
    
end

for i = 1 : numel(trainind)
    if numel(trainind)==1
        break;
    end
    if i ==1
        samp = bin(: ,samporder(i));
        contradicts = samp;
        
    else
        samp = bin(:,samporder(i));
        belief = mean(bin(:,samporder(1:i-1)),2);
        curprob = belief.*samp + (1-belief).*(~samp);
        contradicts = curprob ==0;
    end
    bin = operaterandpar(bin,contradicts);
        %% VERBOSE
        valind = trainind(end)+1:size(bin,2);
    probtrain = mean(bin(:,trainind),2);
    probval = mean(bin(:,valind),2);
    enttrain = entropy(probtrain);
    entval = entropy(probval);
    enttrain = sum(enttrain(:),'omitnan');
    entval = sum(entval(:),'omitnan');
    fprintf('Epoch:%d',size(bin,1));
    fprintf('EntropyTrain %.6f -',enttrain/size(bin,1));
    fprintf('EntropyVal %.6f -',entval/size(bin,1));
    fprintf('\n');
end



end
function [bin] = operaterandpar(bin,contradicts)
contrainds = find(contradicts);
while numel(contrainds)>1

    contranum = numel(contrainds);
    recind = 1:floor(contranum/2);
    sendind = floor(contranum/2)+1:(2.*floor(contranum/2));
    contrainds = contrainds(randperm(contranum));
    sender = contrainds(sendind);
    receivers = contrainds(recind);
bin(receivers,:) = xor(bin(receivers,:),bin(sender,:));
contrainds = contrainds(sendind(1):end);
end
end
function bitind = selectbit(probtrain,ent)
methodnum = 2;
switch methodnum
    case 1
        method = 'maxent';
    case 2
        method = 'random';
end
switch method
    case 'maxent'
        [~,bitind] = max(ent,[],1);
    case 'random'
        bitind = randperm(numel(ent),1);
end
end

