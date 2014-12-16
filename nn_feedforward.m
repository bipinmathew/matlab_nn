function [response,activation]=nn_feedforward(X,w,nn_structure)
% [response,activation]=nn_feedforward(X,w,nn_stucture)
% Get response of neural network to input X size Mx1
% w is array of weights to use.
% nn_structure. Array that lists the number of input, intermediary and output nodes. ie. [M,i1,i2,i3..,K]
% assumes network is fully connected.


    N=length(X);
    K = nn_structure(end);
    if N!=nn_structure(1)
        error("Neural network must have same number of input nodes as input values in the data");
    endif;

    L = length(nn_structure);

    activation = zeros(sum(nn_structure),1);
    response   = zeros(sum(nn_structure),1);

    ai = [0,cumsum(nn_structure)];
    ws = 0;

    l=1;
    activation(1+ai(l):ai(l+1))=X;
    response(1+ai(l):ai(l+1))=X;

    for l=2:L
        w_size =  (1+nn_structure(l-1))*nn_structure(l);
        activation((1+ai(l)):ai(l+1)) = reshape(w((ws+1):(ws+(w_size))),nn_structure(l),1+nn_structure(l-1))*[1;response((1+ai(l-1)):ai(l))];

        response((1+ai(l)):ai(l+1)) = sigmoid(activation((1+ai(l)):ai(l+1)));
        ws += w_size;
    end
end
