function [jw] = nn_backprop(y,o,w,nn_structure)
    L = length(nn_structure);    
    d = zeros(sum(nn_structure),1);
    e = length(o);
    we = length(w);

    l = L;
    d(e-nn_structure(l)+1:e)=(o(e-nn_structure(l)+1:e)-y.').*(1-o(e-nn_structure(l)+1:e)).*o(e-nn_structure(l)+1:e);

    for l=(L-1):-1:1
        temp=reshape(w(we-(1+nn_structure(l))*(nn_structure(l+1))+1:we),nn_structure(l+1),1+nn_structure(l))'*d(e-nn_structure(l+1)+1:e);

        we -=(1+nn_structure(l))*(nn_structure(l+1));
        e = e-nn_structure(l+1);
        d(e-nn_structure(l)+1:e+1)=temp;
    end
    
    
    disp("Stopped in nn_backprop");
    keyboard;
end
