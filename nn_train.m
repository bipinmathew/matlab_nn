function [j,jw] = nn_train(X,Y,w,nn_structure)
    %X=((X-mean(X,2)))./std(X,0,2);
    %disp("Stopped to inspect X");
    %keyboard;
    o=nn_feedforward(X,w,nn_structure);
    [j,jw]=nn_backprop(Y,o,w,nn_structure);
end
