function [U_d,S_d,V_d,threshold,w,sort1,sort2] = digit_trainer(digit1,digit2,feature)
    
    nd = size(digit1,2);
    nc = size(digit2,2);
    [U_d,S_d,V_d] = svd([digit1 digit2],'econ'); 
    digits = S_d*V_d';
    U_d = U_d(:,1:feature); % Add this in
    d1 = digits(1:feature,1:nd);
    d2 = digits(1:feature,nd+1:nd+nc);
    m1 = mean(d1,2);
    m2 = mean(d2,2);

    S_dw = 0;
    for k=1:nd
        S_dw = S_dw + (d1(:,k)-m1)*(d1(:,k)-m1)';
    end
    for k=1:nc
        S_dw = S_dw + (d2(:,k)-m2)*(d2(:,k)-m2)';
    end
    S_db = (m1-m2)*(m1-m2)';
    
    [V_d2,D] = eig(S_db,S_dw);
    [lambda,ind] = max(abs(diag(D)));
    w = V_d2(:,ind);
    w = w/norm(w,2);
    vd1 = w'*d1;
    vd2 = w'*d2;
    
    if mean(vd1)>mean(vd2)
        w = -w;
        vd1 = -vd1;
        vd2 = -vd2;
    end
    
    % Don't need plotting here
    sort1 = sort(vd1);
    sort2 = sort(vd2);
    t1 = length(sort1);
    t2 = 1;
    while sort1(t1)>sort2(t2)
    t1 = t1-1;
    t2 = t2+1;
    end
    threshold = (sort1(t1)+sort2(t2))/2;

    % We don't need to plot results
end

