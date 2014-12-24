function [jw] = nn_backprop(y,o,w,nn_structure)
    L = length(nn_structure);    
    d = zeros(sum(nn_structure),1);
    e = length(d);
    we = length(w);
    w_size=length(w);
    jw = zeros(w_size,1);

    l = L;
    d(e-nn_structure(l)+1:e)=(o(e-nn_structure(l)+1:e)-y.').*(1-o(e-nn_structure(l)+1:e)).*o(e-nn_structure(l)+1:e);

    for l=(L-1):-1:1
        W=reshape(w(we-(1+nn_structure(l))*(nn_structure(l+1))+1:we),nn_structure(l+1),1+nn_structure(l))'(2:end,:);
        temp=(W*d(e-nn_structure(l+1)+1:e));

        we -=(1+nn_structure(l))*(nn_structure(l+1));
        e = e-nn_structure(l+1);

        d(e-nn_structure(l)+1:e)=temp.*(1-o(e-nn_structure(l)+1:e)).*o(e-nn_structure(l)+1:e);
    end


    ai=[0,cumsum(nn_structure)];
    ws=0;

    for l=1:L-1
        w_size =  (1+nn_structure(l))*nn_structure(l+1);
        jw(1+ws:ws+w_size)=kron([1;o(1+ai(l):ai(l+1))],d(1+ai(l):ai(l+1)))
        ws += w_size;
    end

    disp("Stopped")
    keyboard;    
end
