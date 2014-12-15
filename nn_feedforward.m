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

disp("Stopped inside nn_feedforward ...\r\n");
keyboard;

ai = cumsum(nn_structure);
ws = 1;

l=1;
activation(1:ai(l))=X;

for l=2:L
    w_size =  (1+nn_structure(l-1))*nn_structure(l) 
    activation((1+ai(l-1)):ai(l)) = reshape(w(ws:(ws+(w_size)-1)),1+nn_structure(l-1),nn_structure(l))*
    ws += (w_size+1);
    disp("Stopped in main loop.");
    keyboard;
end

keyboard;

end
