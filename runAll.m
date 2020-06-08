startup;
if exist('imdb','var')~=1
load('imdb_vanilla.mat')
end
method = parsemethod('kl-mean'); % View parse method function located in parser folder for customization
if exist('compmat','var')~=1
[imdb,compmat,decompmat] = datacompressor(imdb.train,method,method.verbose);
%save('NETZ','compmat','-mat')
classifier(imdb.TEST,compmat);    
end

