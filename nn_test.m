function [i]=nn_test(X,w,nn_structure)
    o = nn_feedforward(X,w,nn_structure);
    [v,i]=max(o(:,end-nn_structure(end)+1:end),[],2);
end

