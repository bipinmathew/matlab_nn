X = [0 0;0 1;1 0;1 1];
y = [0 0 0 1](:);

[M,N]=size(X);
K = length(unique(y));

Y = y==[0:K-1];
nn_structure=[N,50,K];
lambda = 0;
w = nn_initweights(nn_structure);


costFunc = @(w) nn_train(X,Y,w,lambda,nn_structure);

[j_b,grad] = costFunc(w);

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
