more off;
page_output_immediately(true);
rand("seed",1);

[X,y]=read_mnist('../data/train.csv');

%X = [0 0;
%    0 1;
%    1 0;
%    1 1];

%y = [0;
%    1;
%    1;
%    0];

K = length(unique(y));
[M,N]=size(X);

subsample=10;
Xss = zeros(K*subsample,N);
Yss = zeros(K*subsample,1);

for i=1:K
    idx=find(y==(i-1))(1:subsample);
    Xss(1+(i-1)*subsample:i*subsample,:)=X(idx,:);
    yss(1+(i-1)*subsample:i*subsample)=y(idx);
end

X = Xss;
y = yss;


[M,N]=size(X);

maxIters = 10*M;

alpha = 0.1;


nn_structure=[N,50,K];
L = length(nn_structure);
w_init = nn_initweights(nn_structure);

grad = zeros(length(w_init),1);

y_sample=zeros(length(y),1);

Y = y==[0:K-1];


mu = mean(X,1);
sigma=1+std(X,1);

X=(X-mu)./sigma;

J=zeros(maxIters,1);


costFunc = @(w_init) nn_train(X,Y,w_init,nn_structure);

options = optimset('MaxIter', 1000);
[w_opt, cost] = fmincg(costFunc, w_init, options);


y_sample = nn_test(X,w_opt,nn_structure);

[j,grad] = costFunc(w_opt);

disp(sprintf("Accuracy: %d\r\n",1-sum(y!=y_sample-1)/length(y)))

%   numgrad = computeNumericalGradient(costFunc, w_opt);
%
%   disp([numgrad grad]);
%   fprintf(['The above two columns you get should be very similar.\n' ...
%            '(Left-Your Numerical Gradient, Right-Analytical Gradient)\n\n']);

%   % Evaluate the norm of the difference between two solutions.  
%   % If you have a correct implementation, and assuming you used EPSILON = 0.0001 
%   % in computeNumericalGradient.m, then diff below should be less than 1e-9
%   diff = norm(numgrad-grad)/norm(numgrad+grad);  
    
%   fprintf(['If your backpropagation implementation is correct, then \n' ...
%            'the relative difference will be small (less than 1e-9). \n' ...
%            '\nRelative Difference: %g\n'], diff);

