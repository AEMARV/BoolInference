function [ codem,codeminv,bin,sampprob,binent] = xorinfermat( bin,codem,codeminv,method,verbose)
%function [ codematrix,bin] = xorinfermat( bin,codematrix,method)

% The function infers the codematrix to multiply by bin to remove
% redundancy.
% ! NOTE ! the output bin is the operated binary.
% ! NOTE ! the codematrix is the operated codematrix meaning it takes the
% primitvie version of input to the current space
% leaving the code matrix empty results in assumption of identity.
% bin =  cat(1,bin,ones(1,size(bin,2),'like',bin));
if isempty(codem)
    codem  = eye(size(bin,1),'like',bin);
    codeminv= codem;
end
switch method.trainorder
    case 'permute'
        index = randperm(size(bin,2));
    case 'original'
        index  = 1 : size(bin,2);
    otherwise
        error('not implemented');
end
valnum = 500;
trainind = 1:size(bin,2)-valnum-1;
valind = size(bin,2)-valnum:size(bin,2);
samplesize = numel(trainind)*10;
sampindserie = ceil(rand(1,samplesize).*numel(trainind));
sampindserie = trainind(sampindserie);
start = 1;
for i = 1 : numel(sampindserie)
    %% probabilities
    probtrain  = mean(bin(:,trainind),2);
    probval = mean(bin(:,valind),2);

    %% Some Options for recursion (SKIP)
    opts.totalenttrain = -sum(probtrain.*log2(probtrain) + (1-probtrain).*log2(probtrain));
    opts.totalentval = -sum(probval.*log2(probval) + (1-probval).*log2(probval));
    opts.depth = 0;
    opts.trainind = trainind;
    opts.valind = valind;
   %% OPERATION
%     [bin,codem,codeminv,contradicts,opts] = operaterecursive(bin,codem,codeminv,contradicts,opts);
while true
    [contradicts,bin] = gencontradicts(bin,sampindserie,start,i);
    
    if numel(find(contradicts))<2
        break;
    end
 [codem,codeminv,bin] = operaterandpar(codem,codeminv,bin,contradicts,1);
end
% if numel(find(contradicts))==0
%     start = i;
%     continue;
% end
    %% NL operation
              
%           bin = operateNL(bin,~samp(senders),senders,receiver);
    %% VERBOSE
    proball = mean(bin(:,1:end-valnum-1),2);
    entall = proball.*log2(proball) + (1-proball).*log2(1-proball);
    entall = sum(entall,'omitnan');
    probrest = mean(bin(:,end-valnum:end),2);
    entrest = probrest.*log2(probrest) + (1-probrest).*log2(1-probrest);
    entrest = sum(entrest,'omitnan');
    codement = sum(codem(:));
    
    % binary sampling entropy
    
    
    if verbose.exist
        fprintf('%.2f%%',(100*i)/size(bin,2))
        fprintf('EntropyTrain %.6f -',entall./size(bin,1));
        fprintf('EntropyVal %.6f -',entrest./size(bin,1));
        fprintf('EntropyCodem %.4f -',codement);
        fprintf('\n');
        
    end
    %     % show images
    %     im = prob./max(prob);
    %     sender = find(sender);
    %     im = codeminv(:,sender(randperm(numel(sender),1)));
    contrainds = find(contradicts);
    if isempty(contrainds)
        continue;
    end
    index = contrainds(randperm(numel(contrainds),1));
    im = codeminv(:,index);
    im = reshape(im,32,32,3,8);
    im = im.*reshape(2.^(-1:-1:-8),1,1,1,8);
    im = sum(im,4);
    f = imshow(real(single(reshape(im,32,32,3))),[],'InitialMagnification','fit');
    %     im = covarmat;
    %      f = imshow(imresize(single(gather(codeminv)),[100,100]),[],'InitialMagnification','fit');
    drawnow update;
end

fprintf('Entropy %.4f -',entall);
fprintf('\n');
sampprob = proball.*bin + (1-proball).*(~bin);
sampprob = sum(log2(sampprob),1);
binent = mean(bin,2);
binent = -(binent.*log2(binent) + (1-binent).*log2(1-binent));
binent(isnan(binent)) = 0;
end
function [contradicts,bin] = gencontradicts(bin,sampindserie,start,curind)
methodnum = 2;
switch methodnum
    case 1
        %% looks at data in order and generates contradiction based on the distance with most probabale sample
        method = 'accumulative_soft';
    case 2
        method = 'accumulative_hard';
end
        
switch method
    case 'accumulative_soft'
  samp = bin(:,sampindserie(curind));
    accprob = mean(bin(:,sampindserie(start:max(curind-1,start))),2);
    accsamp = accprob > 0.5;
    contradicts = xor(samp,accsamp);
    case 'accumulative_hard'
        samp = bin(:,sampindserie(curind));
    accprob = mean(bin(:,sampindserie(start:max(curind-1,start))),2);
    accsamp = accprob > 0.5;
    contradicts = xor(samp,accsamp);
    contradicts(abs(accprob-0.5)~=0.5) = 0;
end
end
function [bin,codem,codeminv,contradicts,opts] = operaterecursive(bin,codem,codeminv,contradicts,opts)
if size(bin,1)<3
    return;
end
if size(bin,1)>inf
    
    inda = 1: floor(size(contradicts,1)/2);
    indb = floor(size(contradicts,1)/2)+1:size(contradicts,1);
    indsperm = 1:size(contradicts,1);
    inda = indsperm(inda);
    indb = indsperm(indb);
    inda = rand(size(bin(:,1)))>0.5;
    indb = rand(size(bin(:,1)))>0.5;
    %% divde

        opts.depth = opts.depth+1;
        [bin(inda,:),codem(inda,:),codeminv(:,inda),contradicts(inda,:),opts] = operaterecursive(bin(inda,:),codem(inda,:),codeminv(:,inda),contradicts(inda,:),opts);
        opts.depth = opts.depth-1;


        opts.depth = opts.depth+1;
        [bin(indb,:),codem(indb,:),codeminv(:,indb),contradicts(indb,:),opts] = operaterecursive(bin(indb,:),codem(indb,:),codeminv(:,indb),contradicts(indb,:),opts);
        opts.depth = opts.depth-1;
%% Sort Bits
%         ent = mean(bin,2);
% ent = -ent.*log2(ent) - (1-ent).*log2(1-ent);
% agthanb = sum(ent(inda),'omitnan')>sum(ent(indb),'omitnan');
% if agthanb
%     sortind = [inda,indb];
% else
%     sortind = [indb,inda];
% end
% % [~,sortind] = sort(ent,1,'ascend');
% bin = bin(sortind,:);
% codem=  codem(sortind,:);
% codeminv = codeminv(:,sortind);
end
if size(bin,1)>inf
    return;
end
%% conquer
entwholetrain = opts.totalenttrain;
entwholeval = opts.totalentval;
probtrain = mean(bin(:,opts.trainind),2);
entthistrain = -sum(probtrain.*log2(probtrain) + (1-probtrain).*log2(1-probtrain));
entresttrain = entwholetrain - entthistrain;

probval = mean(bin(:,opts.valind),2);
entthisval = -sum(probval.*log2(probval) + (1-probval).*log2(1-probval));
entrestval = entwholeval- entthisval;

for j = 1 :1
     ind = randperm(numel(opts.trainind),numel(opts.trainind));
%  bin = xor(bin,bin(:,ind(1)));


bestval = inf;
samplesize = numel(opts.trainind);
sampindserie = ceil(rand(1,samplesize).*numel(opts.trainind));
sampindserie = opts.trainind(sampindserie);
start = 1;
for i = 2 : numel(opts.trainind)
%% setsamples
    
    if i ==1 
        prevsamp= bin(:,1).*0;
        samp = bin(:,sampindserie(i));
        bin = xor(bin,samp);
        continue;
    else
        prevsamp  = bin(:,sampindserie(start:i-1));
        prevprob = mean(prevsamp,2);
        proball = mean(bin(:,sampindserie),2);
        prevsamp = logical(mod(sum(prevsamp,2),2));
        
%         prevsamp = logical(mod(floor((randperm(i,1)).*prevsamp),2));
        samp = bin(:,sampindserie(i));
    end
    if ~any(prevprob ==0 | prevprob==1)
%          break;
    end
%% form contradicts
%      contradicts = xor(prevsamp,samp);
       probcur= samp.*proball + (~samp).*(1-proball);
%         bin = xor(bin,proball>0.5);
    sampgen = proball>rand(size(proball));
%     sampgen = bin(:,sampindserie(min(i+1,numel(sampindserie))));
       contradicts = xor(samp,sampgen);
%         contradicts = probcur<0.5;
%        contradicts = and(contradicts,rand(size(contradicts))>0.2);
% contradicts = prevsamp;
%%
if numel(find(contradicts))<1
    continue
end



%     samp = randperm(numel(ind),1);  
    contrainds = find(contradicts);
       sender = randperm(numel(contrainds),1);
%      [~,sender] = min(probcur(contrainds));
%     sender = 1;
    sender = contrainds(sender);
%     receivers = and(contradicts,rand(size(contradicts))>0.5);
    receivers = contradicts;
    receivers(sender) = 0;
% %     [codem,codeminv,bin] = operate(codem,codeminv,bin,sender,receivers);
        [codem,codeminv,bin] = operatefullrand(codem,codeminv,bin,contradicts);
    %% Nonlinearity
%      senders = ~contradicts;
%      receiver = sender;
%      bin = operateNL(bin,~samp(senders),senders,receiver);
    %% verbose
    entthistrain = mean(bin(:,opts.trainind),2);
    entthistrain = entthistrain.*log2(entthistrain) + (1-entthistrain).*log2(1-entthistrain);
    entthistrain = -sum(entthistrain(:),'omitnan');
    entthisval = mean(bin(:,opts.valind),2);
    entthisval = entthisval.*log2(entthisval) + (1-entthisval).*log2(1-entthisval);
    entthisval = -sum(entthisval(:),'omitnan');
    valratio = entthisval./size(bin,1);
    trainratio = entthistrain./size(bin,1);
    if valratio < bestval
        bestval = valratio;
        besttrain = trainratio;
    end
    fprintf('Depth:%d ',opts.depth);
    fprintf('%.2f%%',(100*i)/numel(opts.trainind))
    fprintf('EntropyTrain %.6f -',entthistrain/size(bin,1));
    fprintf('EntropyVal %.6f -',entthisval/size(bin,1));
    fprintf('MinEntropyTrain %.4f -',besttrain);
    fprintf('MinEntropyVal %.4f -',bestval);
    fprintf('\n');

    
end
end
opts.totalenttrain = entthistrain + entresttrain;
opts.totalentval = entthisval + entrestval;

% contrainds = find(contradicts);
% sender = randperm(numel(contrainds),1);
% sender = contrainds(sender);
% receivers = contradicts;
% receivers(sender) = 0;
% [codem,codeminv,bin] = operate(codem,codeminv,bin,sender,receivers);
% contradicts(sender) = 1;
% contradicts(receivers) = 0;
end




function [codem,codeminv,bin] = operatefullrand(codem,codeminv,bin,contradicts,remains)
contrainds = find(contradicts);
j= 1;
while numel(contrainds)>remains
%      if j==2;break;end
    sender = randperm(numel(contrainds),1);
    sender = contrainds(sender);
    receivers = contradicts;
    receivers(sender) = 0;
    drops = rand(size(contradicts))>0.1;
    drops(sender) = 1;
%     drops = or(drops,1);
%     recinds = find(receivers);
%     recinds = recinds(randperm(numel(recinds),1));
%     drops(recinds) = 0;
    receivers = and(receivers,~drops);
    contradicts = and(contradicts,drops);
codem(receivers,:) = xor(codem(receivers,:) ,codem(sender,:));
codeminv(:,sender) = xor(mod(sum(codeminv(:,receivers),2),2),codeminv(:,sender));
bin(receivers,:) = xor(bin(receivers,:),bin(sender,:));
contrainds = find(contradicts);
j = j+1;
end
end
function [codem,codeminv,bin] = operaterandpar(codem,codeminv,bin,contradicts,remains)
contrainds = find(contradicts);
j= 1;
while numel(contrainds)>remains
%      if j==2;break;end
    contranum = numel(contrainds);
    recind = 1:floor(contranum/2);
    sendind = floor(contranum/2)+1:(2.*floor(contranum/2));
    contrainds = contrainds(randperm(contranum));
    sender = contrainds(sendind);
    receivers = contrainds(recind);
codem(receivers,:) = xor(codem(receivers,:) ,codem(sender,:));
codeminv(:,sender) = xor(codeminv(:,receivers),codeminv(:,sender));
bin(receivers,:) = xor(bin(receivers,:),bin(sender,:));
contrainds = contrainds(sendind(1):end);
j = j+1;
end
end


function [bin] = operateNL(bin,signs,senders,receiver)
andSenders = xor(bin(senders,:),signs);
andSenders = all(andSenders,1);
bin(receiver,:) = xor(andSenders,bin(receiver,:));
end
function [codem,codeminv,bin] = operatems(codem,codeminv,bin,senders,receiver)
codem(receiver,:) = xor(codem(receiver,:) ,mod(sum(codem(senders,:),1),2));
codeminv(:,senders) = xor(codeminv(:,receiver),codeminv(:,senders));
bin(receiver,:) = xor(bin(receiver,:),mod(sum(bin(senders,:),1),2));
end
function [codem,codeminv,bin] = operatemsmr(codem,codeminv,bin,senders,receivers)
codem(receivers,:) = xor(codem(receivers,:) ,logical(mod(sum(codem(senders,:),1),2)));
codeminv(:,senders) = xor(mod(sum(codeminv(:,receivers),2),2),codeminv(:,senders));
bin(receivers,:) = xor(bin(receivers,:),mod(sum(bin(senders,:),1),2));
end
function [codem,codeminv,bin] = operate(codem,codeminv,bin,sender,receivers)
codem(receivers,:) = xor(codem(receivers,:) ,codem(sender,:));
codeminv(:,sender) = xor(mod(sum(codeminv(:,receivers),2),2),codeminv(:,sender));
bin(receivers,:) = xor(bin(receivers,:),bin(sender,:));
end
function primebitind = selectbit(contradictions,prob,method)
contradictionNum = numel(find(contradictions));
switch method
    case 'random'
        primebitind = [];
        if contradictionNum >0
            primebitind = randperm(contradictionNum,1);
        end
        
    case 'maxprob'
        primebitind = [];
        if contradictionNum >0
            [~,primebitind] =   max(prob(contradictions));
        end
    case 'maxent'
        primebitind = [];
        if contradictionNum >0
            [~,primebitind] =   min(abs(prob(contradictions)-0.5));
        end
    case 'minent'
        primebitind = [];
        if contradictionNum >0
            [~,primebitind] =   max(abs(prob(contradictions)-0.5));
        end
    case 'first'
        primebitind = find(contradictions);
        primebitind = primebitind(1);
    case 'halfhalf'
        prob = prob(contradictions);
        ents =  -prob.*log2(prob) - (1-prob).*log2(1-prob);
        [ents,primebitind] = sort(prob,'ascend');
        cument = cumsum(ents);
        ind = find(cument>cument(end)/2,1);
        %          [~,primebitind] = sort(prob(contradictions),'ascend');
        %         sendind = 1:floor(contradictionNum/2);
        sendind = 1:ind;
        primebitind = primebitind(sendind);
end

end
function bin = shiftrandom(bin,origsize)
bin = reshape(bin,origsize);
sz1 = size(bin,1);
sz2 = size(bin,2);
rand = gpuArray.rand(1,2);
rand = floor(rand.*[sz1,sz2]/2)+1;
bin = circshift(bin,[rand,0,0]);
% bin(1:rand(1),:,:,:) = nan;
% bin(:,1:rand(2),:,:) = nan;
bin = bin(:);
end
function [CodeMatrix,CodeMatrixinv] = regularize(CodeMatrix,CodeMatrixinv)
imresponse = logical(mod(sum(CodeMatrix,2),2));
sender = find(imresponse,1);
imresponse = ~imresponse;
CodeMatrix(imresponse,:) = xor(CodeMatrix(sender,:),CodeMatrix(imresponse,:));
CodeMatrixinv(:,sender) = xor(mod(sum(CodeMatrixinv(:,imresponse),2),2),CodeMatrixinv(:,sender));
end
function r = randsampexp(sz,bitnum)
r = rand([sz,bitnum])>0.5;
[~,r] = max(r,[],ndims(r));

end