[X,y]=read_mnist('../data/train.csv');
[M,N]=size(X);
K = length(unique(y));
nn_structure=[N,5,K];
L = length(nn_structure);

w = nn_initweights(nn_structure);

response=nn_feedforward(X(1,:),w,nn_structure)

