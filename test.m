datamat = [0;1];
for i = 1 : 4
    datamat= cat(2,zeros(size(datamat,1),1),datamat);
    datamat = cat(1,datamat,~datamat);
end
datamatnew = datamat;
for i = 1:5
    if i ==1 || i==5
        continue;
    end
    datamatnew(:,i) = mod(datamat(:,i)+(datamat(:,i-1).*datamat(:,i+1)),2);
        
end
datamatnew = sum(datamatnew .* (2.^(0:4)),2);