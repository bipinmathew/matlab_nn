[X,y]=read_mnist('../data/train.csv');
[M,N]=size(X);
K = length(unique(y));
nn_structure=[N,5,K];
L = length(nn_structure);

w = nn_initweights(nn_structure);

for i = 1:M
   [response,activation]=nn_feedforward(X(i,:),w,nn_structure);
end

