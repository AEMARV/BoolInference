function [ codem,codeminv,bin,sampprob,binent] = xorinfermatrev( bin,codem,codeminv,method,verbose)
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
bin =logical(mod( single(codem) *single(bin),2));
% bin = xor(bin(:,1),bin);
bin = bin(:,randperm(size(bin,2),size(bin,2)));
valnum = 500;
start = 1;
memory = floor(size(bin,2)/2);
memory =2;
indcomp =0;
bitnum = 1;
ctemp = inf;
primebit = [];
prevrec = [];
% prob = sqrt(prob);
% map = (1:32).*(1:32)';
% map = repmat(map(:),3,1);
% prob = prob.*exp(1i.*map);
prob  = mean(bin(:,1:size(bin,2)-valnum),2);
probwave = prob + (1-prob).*pi./2;
   complexinf(prob,100,1000);
for i = 1 : size(bin,2).^2
    %% Indices which need change
    probpol = mean(2.*bin(:,1:size(bin,2)-valnum)-1,2)';
    prob  = mean(bin(:,1:size(bin,2)-valnum),2);
    %     covarmat = single(2.*bin(:,1:size(bin,2)-valnum)-1)*single(2.*bin(:,1:size(bin,2)-valnum)'-1);
    %     covarmat = covarmat./(size(bin,2)-valnum);
    %        complexinf(prob,30,30);
    %     [contradicts] = prob2bin(prob,bitnum);
    %     receivers = contradicts;
    
    contradicts = bin(:,randperm(size(bin,2)-valnum,memory));
    
    %     contradicts = bin(:,[rand(size(bin,2)-valnum,1)>0.5;zeros([valnum,1],'like',bin)]);
    contradicts = logical(mod(sum(contradicts,2),2));
    contradicts =logical(mod(floor(prob.*-log2(rand)),2));
    contradictsrec= contradicts;
    if rand>1
    contradictsorig = logical(mod(sum(codeminv(:,contradicts),2),2));
    diffinvcol = mod(contradictsorig+codeminv(:,contradicts),2);
    sumdiffinvcol = sum(diffinvcol,1);
     [~,invcol] = max(sumdiffinvcol,[],2);
%     invcol = randperm(numel(sumdiffinvcol),1);
    contradictsind = find(contradicts);
    invcolind = contradictsind(invcol);
    invcol = codeminv(:,invcolind);
    contradicts = logical(mod(invcol+contradictsorig,2));
    primebit = find(contradicts);
     if isempty(primebit)
        continue;
    end
    primebitind = selectbit(contradicts,prob,method.bitselect);
    %% selects which bit to xor to the rest of contradictions
    %     contradictionNum = numel(primebit);
    receivers = contradicts;
    sender = primebit(primebitind);
    receivers(sender) = 0;
   
    %         [covarmat, sender,receivers] = pcasenderrec(covarmat,probpol,prevrec);
    %         prevrec = receivers;
    [codem,codeminv,bin] = operatemsmrrev(codem,codeminv,bin,sender,receivers);
    else
        if mean(contradicts)<0.5
            contradicts = ~contradicts;
        end
        primebit = find(contradicts);
        primebitind = selectbit(contradicts,prob,method.bitselect);
        %% selects which bit to xor to the rest of contradictions
        %     contradictionNum = numel(primebit);
        receivers = contradicts;
        sender = primebit(primebitind);
        receivers(sender) = 0;
        
        %         [covarmat, sender,receivers] = pcasenderrec(covarmat,probpol,prevrec);
        %         prevrec = receivers;
        [codem,codeminv,bin] = operatemsmr(codem,codeminv,bin,sender,receivers);
    end
    
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
        fprintf('EntropyTrain %.4f -',entall);
        fprintf('EntropyVal %.4f -',entrest);
        fprintf('EntropyCodem %.4f -',codement);
        fprintf('\n');
        
    end
    %     % show images
    %     im = prob./max(prob);
    %     sender = find(sender);
    %     im = codeminv(:,sender(randperm(numel(sender),1)));
    im = codeminv(:,sender);
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
function [codem,codeminv,bin] = operatemsmr(codem,codeminv,bin,senders,receivers)
codem(receivers,:) = xor(codem(receivers,:) ,logical(mod(sum(codem(senders,:),1),2)));
codeminv(:,senders) = xor(mod(sum(codeminv(:,receivers),2),2),codeminv(:,senders));
bin(receivers,:) = xor(bin(receivers,:),mod(sum(bin(senders,:),1),2));
end
function [codem,codeminv,bin] = operatems(codem,codeminv,bin,senders,receiver)
codem(receiver,:) = xor(codem(receiver,:) ,mod(sum(codem(senders,:),1),2));
codeminv(:,senders) = xor(codeminv(:,receiver),codeminv(:,senders));
bin(receiver,:) = xor(bin(receiver,:),mod(sum(bin(senders,:),1),2));
end
function [codem,codeminv,bin] = operatemsmrrev(codem,codeminv,bin,senders,receivers)
codeminv(receivers,:) = xor(codeminv(receivers,:) ,logical(mod(sum(codeminv(senders,:),1),2)));
codem(:,senders) = xor(mod(sum(codem(:,receivers),2),2),codem(:,senders));
binstochange = bin(senders,:);
 bin(:,binstochange) = xor(bin(:,binstochange),mod(sum(codem(:,receivers),2),2));
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