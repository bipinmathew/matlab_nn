more off;
page_output_immediately(true);
warning ('off', 'Octave:broadcast');
warning('off', 'Octave:possible-matlab-short-circuit-operator');

rand("seed",1);

[X,y]=read_mnist('../data/train.csv');
%X=X(1:1000,:);
%y=y(1:1000);

[M,N]=size(X);

breaks=[0,cumsum(M*[0.60,0.30,0.10])];

X_tr=X(1+breaks(1):breaks(2),:);
y_tr=y(1+breaks(1):breaks(2))(:);

X_cv=X(1+breaks(2):breaks(3),:);
y_cv=y(1+breaks(2):breaks(3))(:);

X_t=X(1+breaks(3):breaks(4),:);
y_t=y(1+breaks(3):breaks(4))(:);


[Xi] = read_mnist_test('../data/test.csv');

K = length(unique(y));

%subsample=10;
%Xss = zeros(K*subsample,N);
%Yss = zeros(1,K*subsample);
%
%for i=1:K
%    idx=find(y_tr==(i-1))(1:subsample);
%    Xss(1+(i-1)*subsample:i*subsample,:)=X_tr(idx,:);
%    yss(1+(i-1)*subsample:i*subsample)=y_tr(idx);
%end

%p = randperm(subsample*K);
%X_tr = Xss(p,:);
%y_tr = yss(:)(p);


[M,N]=size(X_tr);

NUMEPOCHS = 10;
BLOCKSIZE = 10;
NUMBLOCKS = floor(M/BLOCKSIZE);

alpha = 0.1;
lambda = 0.1;
num_epochs = 1;

mu = mean(X_tr,1);
sigma= 1+std(X_tr,1);


e_cv_max = Inf;
mc_cv_max = 1;


nn_structure=[N,500,K];
L = length(nn_structure);

w_init = nn_initweights(nn_structure);


grad = zeros(length(w_init),1);

y_sample=zeros(length(y_tr),1);

Y = y_tr==[0:K-1];

w_opt = w_init;

e_cv_max = Inf;

lambdas= 10.^[-1:1:1]; %10.^[-4:0.5:-1]; %10.^[1];

e_tr = zeros(length(lambdas),1);
e_cv = zeros(length(lambdas),1);
k=1

best_lambda=lambdas(1);

for lambda=lambdas
    w = w_init;
    for epoch=1:NUMEPOCHS
        disp(sprintf("Starting epoch: %d\r\n",epoch))
        for j=0:NUMBLOCKS-1
            disp(sprintf("... Processing block: %d of %d \r\n",j+1,NUMBLOCKS));
            idx = [(j*BLOCKSIZE)+1:1:min((j+1)*BLOCKSIZE,M)];
            costFunc = @(w) nn_train((X_tr(idx,:)-mu)./sigma,Y(idx,:),w,lambda,nn_structure);

            options = optimset('MaxIter', 500);
            [w, cost] = fmincg(costFunc, w, options);

        end
    end
        yp_tr = nn_test((X_tr(idx,:)-mu)./sigma,w,nn_structure)(:)-1;
        yp_cv = nn_test((X_cv-mu)./sigma,w,nn_structure)(:)-1;
        e_tr(k) = sum((y_tr(idx)!=yp_tr))/length(idx);
        e_cv(k) = sum((y_cv!=yp_cv))/length(y_cv);
        disp(sprintf("lambda: %f, e_tr error: %f, e_cv: %f \r\n",lambda,e_tr(k),e_cv(k)));
        k+=1;
    if(e_cv<e_cv_max)
        e_cv_max = e_cv;
        w_opt = w;
        best_lambda = lambda;
    end
end

yp_t = nn_test((X_t-mu)./sigma,w_opt,nn_structure)(:)-1;
e_t = sum(yp_t!=y_t)/length(y_t)

%yp_tr = nn_test((X_tr-mu)./sigma,w,nn_structure)(:)-1;
%yp_cv = nn_test((X_cv-mu)./sigma,w,nn_structure)(:)-1;



