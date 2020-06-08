function [ imdb] = createImdb()
%CREATEIMDB Summary of this function goes here
%   Detailed explanation goes here
folder = './DATA/cifar-10-batches-mat';
fileprefix = 'data_batch_';
fileext = '.mat';
traindata = [];
trainlabels = [];
imdb = [];
for i = 1:5
    batch = load([folder,'/',fileprefix,int2str(i),fileext]);
    traindata = cat(1,traindata,batch.data);
    trainlabels = cat(1,trainlabels,batch.labels);
end
    imdb.train = [];
    imdb.TEST = [];
    imdb.train.data = datareshape(traindata);
    imdb.train.labels = trainlabels;
    testbatch = load([folder,'/','test_batch',fileext]);
    testlabels = testbatch.labels;
    testdata = testbatch.data;
    imdb.TEST.data =  datareshape(testdata);
    imdb.TEST.labels = testlabels;
save('imdb_vanilla.mat','imdb');
end
function dataresh = datareshape(data)
    datap = data';
    datap = reshape(datap,32,32,3,size(data,1));
    dataresh = permute(datap,[2,1,3,4]);
end
