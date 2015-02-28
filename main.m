more off;
page_output_immediately(true);
warning ('off', 'Octave:broadcast');
warning('off', 'Octave:possible-matlab-short-circuit-operator');

rand("seed",1);

[X,y]=read_mnist('../data/train.csv');
%X=X(1:8000,:);
%y=y(1:8000);

[M_f,N_f]=size(X);

NUMPC = 10;
[v,mu] = pca(X(1:1000,:));


v = v(:,1:NUMPC);

X=(X-mu)*v;

breaks=[0,cumsum(M_f*[0.60,0.30,0.10])];

X_tr=X(1+breaks(1):breaks(2),:);
y_tr=y(1+breaks(1):breaks(2))(:);

X_cv=X(1+breaks(2):breaks(3),:);
y_cv=y(1+breaks(2):breaks(3))(:);

X_t=X(1+breaks(3):breaks(4),:);
y_t=y(1+breaks(3):breaks(4))(:);


[Xi] = read_mnist_test('../data/test.csv');
Xi = (Xi-mu)*v;

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
BLOCKSIZE = 1000;
NUMBLOCKS = floor(M/BLOCKSIZE);

alpha = 0.1;
lambda = 0.1;

e_cv_max = Inf;
mc_cv_max = 1;


nn_structure=[N,500,K];
L = length(nn_structure);

w_init = nn_initweights(nn_structure);


grad = zeros(length(w_init),1);

y_sample=zeros(length(y_tr),1);

Y_tr = y_tr==[0:K-1];
Y = y==[0:K-1];

w_opt = w_init;

e_cv_max = Inf;

lambdas= 10.^[-2:0.25:1]; %10.^[-4:0.5:-1]; %10.^[1];

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
            
            [j,jw]=nn_train(X_tr(idx,:),Y_tr(idx,:),w,lambda,nn_structure);
            disp(sprintf("cost: %f\r\n",j));
            w -= alpha*jw;
        end
    end
        yp_tr = nn_test(X_tr(idx,:),w,nn_structure)(:)-1;
        yp_cv = nn_test(X_cv,w,nn_structure)(:)-1;
        e_tr(k) = sum((y_tr(idx)!=yp_tr))/length(idx);
        e_cv(k) = sum((y_cv!=yp_cv))/length(y_cv);
        disp(sprintf("lambda: %f, e_tr error: %f, e_cv: %f \r\n",lambda,e_tr(k),e_cv(k)));
    if(e_cv(k)<e_cv_max)
        e_cv_max = e_cv(k);
        w_opt = w;
        best_lambda = lambda;
    end
    k+=1;
end


w = w_init;
NUMBLOCKS = floor(M_f/BLOCKSIZE);
for epoch=1:NUMEPOCHS
    disp(sprintf("Starting epoch: %d\r\n",epoch))
    for j=0:NUMBLOCKS-1
        disp(sprintf("... Processing block: %d of %d \r\n",j+1,NUMBLOCKS));
        idx = [(j*BLOCKSIZE)+1:1:min((j+1)*BLOCKSIZE,M_f)];
        
        [j,jw]=nn_train(X(idx,:),Y(idx,:),w,best_lambda,nn_structure);
        disp(sprintf("cost: %f\r\n",j));
        w -= alpha*jw;
    end
end
w_opt = w;

yp_t = nn_test(X_t,w_opt,nn_structure)(:)-1;
e_t = sum(yp_t!=y_t)/length(y_t)

disp(sprintf("Test error: %f\r\n",e_t));

disp("Writing out result...\r\n");

yp_i = nn_test(Xi,w_opt,nn_structure)(:)-1;
dlmwrite('output.csv',[[1:1:length(yp_i)].',yp_i],'delimiter',',');
