function [j,jw] = nn_backprop(y,o,w,nn_structure)
    L = length(nn_structure);    
    d = zeros(sum(nn_structure(2:end)),1);
    e = length(d);
    we = length(w);
    w_size=length(w);
    jw = zeros(w_size,1);

    l = L;


    j = (norm((o(end-nn_structure(l)+1:end)-y.'))^2)/2;

    d(end-nn_structure(l)+1:end)=(o(end-nn_structure(l)+1:end)-y.').*(1-o(end-nn_structure(l)+1:end)).*o(end-nn_structure(l)+1:end);

    for l=(L-1):-1:2
        W=reshape(w(we-(1+nn_structure(l))*(nn_structure(l+1))+1:we),nn_structure(l+1),1+nn_structure(l))'(2:end,:);
        temp=(W*d(e-nn_structure(l+1)+1:e));

        we -=(1+nn_structure(l))*(nn_structure(l+1));
        e = e-nn_structure(l+1);

        d(e-nn_structure(l)+1:e)=temp.*(1-o(e-nn_structure(l)+1:e)).*o(e-nn_structure(l)+1:e);
    end

    ai=[0,cumsum(nn_structure)];
    di=[0,cumsum(nn_structure(2:end))];
    ws=0;

    for l=1:L-1
        w_size =  (1+nn_structure(l))*nn_structure(l+1);
        jw(1+ws:ws+w_size)=kron([1;o(1+ai(l):ai(l+1))],d(1+di(l):di(l+1)));
        ws += w_size;
    end
end
