function [j,jw] = nn_train(X,Y,w,lambda,nn_structure)
    o=nn_feedforward(X,w,nn_structure);
    [j,jw]=nn_backprop(Y,o,w,lambda,nn_structure);
end
