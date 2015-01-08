more off;
page_output_immediately(true);

%[X,y]=read_mnist('../data/train.csv');

X = [0 0;
    0 1;
    1 0;
    1 1];

y = [0;
    0;
    0;
    1];

[M,N]=size(X);
alpha = 0.1;
K = length(unique(y));
nn_structure=[N,1,K];
L = length(nn_structure);

w = nn_initweights(nn_structure);
grad = zeros(length(w),1);

y_sample=zeros(length(y),1);

Y = y==[0:K-1];

J=zeros(M,1);


for i=1:1000
    w = w-(alpha*grad);
    
    costFunc = @(w) nn_train(X,Y,w,nn_structure);
    [j,grad] = costFunc(w);
    %[j,grad] = nn_train(X,Y,w,nn_structure);

    %j 

    J(i)=j;
end


%for i=1:M
%    y_sample(i) = nn_test(X,w,nn_structure);
%end



    numgrad = computeNumericalGradient(costFunc, w);

    disp([numgrad grad]);
    fprintf(['The above two columns you get should be very similar.\n' ...
             '(Left-Your Numerical Gradient, Right-Analytical Gradient)\n\n']);

    % Evaluate the norm of the difference between two solutions.  
    % If you have a correct implementation, and assuming you used EPSILON = 0.0001 
    % in computeNumericalGradient.m, then diff below should be less than 1e-9
    diff = norm(numgrad-grad)/norm(numgrad+grad);  
     
    fprintf(['If your backpropagation implementation is correct, then \n' ...
             'the relative difference will be small (less than 1e-9). \n' ...
             '\nRelative Difference: %g\n'], diff);





