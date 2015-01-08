function [j,jw] = nn_backprop(y,o,w,nn_structure)
    L = length(nn_structure);    
    M = length(y);
    d = zeros(sum(nn_structure(2:end)),1);
    jw = zeros(length(w),1);

    l = L;



    j = sum(norm(o(:,end-nn_structure(l)+1:end)-y,2,"rows"))/(2*M);


    ri = cumsum(nn_structure(end:-1:2));

    for i=1:M
        we = length(w);
        l = L;
        e = length(d);
        d(end-nn_structure(l)+1:end)=(o(i,end-nn_structure(l)+1:end)-y(i,:)).*(1-o(i,end-nn_structure(l)+1:end)).*o(i,end-nn_structure(l)+1:end);

            

        for l=(L-1):-1:2
            W=reshape(w(we-(1+nn_structure(l))*(nn_structure(l+1))+1:we),nn_structure(l+1),1+nn_structure(l))'(2:end,:);
            temp=(W*d(e-nn_structure(l+1)+1:e));

            disp("Stopped to investigate d 1")
            keyboard;

            d(e-nn_structure(l)+1:e)=temp.*(1-o(i,e-nn_structure(l)+1:e)).'.*o(i,e-nn_structure(l)+1:e).';

            we -=(1+nn_structure(l))*(nn_structure(l+1));
            e = e-nn_structure(l+1);
        end


        ai=[0,cumsum(nn_structure)];
        di=[0,cumsum(nn_structure(2:end))];
        ws=0;

        
        disp("Stopped to investigate d")
        keyboard;

        for l=1:L-1
            w_size =  (1+nn_structure(l))*nn_structure(l+1);
            jw(1+ws:ws+w_size)+=kron([1;o(i,1+ai(l):ai(l+1)).'],d(1+di(l):di(l+1)));
            ws += w_size;
        end
    end

    jw;

end
