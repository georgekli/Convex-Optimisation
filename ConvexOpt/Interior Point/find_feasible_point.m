function [xFeas]=find_feasible_point(A,b,c)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% M-file that implements an algorithm for finding a Feasible Point for %
% a linear function with a linear inequality constraint and positivity % 
% constraints                                                          %
%                                                                      %
% Objective function: c^T x                                            %
%                                                                      %
% Inequality constraint:  A^T x - b \le 0````                          %
%                         x_i >= 0, i=1, 2                             %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    threshold_1=10^-6;
    threshold_2=10^-6;     
    mu=2;
    alpha=0.01;
    beta=0.9;
    h=10^(-6);
    n=size(A,2);
    t=1;
    x=sym('x',[n 1]);
    s=sym('s',[1 1]);
    x_init=A\b;                         % start from a point that satisfies Ax=b
    I=eye(n+1);
    iter=1;
    xOuter(:,iter)=x_init;
    sval=max(x_init)+1;
    outer_iter=1;
    while(1)
        f=symfun(t.*s-sum(log(x+s)),[x;s]); %barrier cost function
        xk(:,1)=xOuter(:,outer_iter);       % Start new optimization from the previous solution
        iter=1;
        while(1)
            if(all(xk(:,iter)>=0) && (any((A*xk(:,iter)-b)<threshold_1) || any((A*xk(:,iter)-b)>(0-threshold_1))))
                break;
            end
            xCell=num2cell([xk(:,iter);sval]);
            fun_val=double(f(xCell{:}));
            gval=t.*I(:,n+1);
            gval_t=0;
            for i=1:n
                gval_t = gval_t + ((1./(xk(i,iter)+sval)).*(I(:,i)+I(:,n+1)));
                if gval(i)==0
                    gval(i)=h;
                end
            end
            gval=gval-gval_t;
            hval=0;
            for i=1:n
                hval=hval+((1./(xk(i,iter)+sval).^2))*(I(:,i)+I(:,n+1))*((I(:,i)+I(:,n+1))');
                if hval(i)==0
                    hval(i)=h;
                end
            end
            A_new = [A zeros(size(A,1),1)];
            firstPart=[hval A_new'; A_new zeros(size(A_new,1))];
            secondPart=[-gval; zeros(size(A_new,1),1)];
            %secondPart=[-gval; A_new*[xk(:,iter);sval]-b];
            [sol]=linsolve(firstPart,secondPart);
            xDelta=sol(1:size(A,2));
            sDelta=sol(size(A,2)+1);
            tau=1;
            x_new=xk(:,iter)+tau*xDelta;
            s_new=sval+tau*sDelta;
            l_x=sqrt([xDelta;sDelta]'*hval*[xDelta;sDelta]);       % Newton decrement
            if l_x^2/2 <= threshold_1          % Newton iterations termination condition
               break; 
            end    
            % Backtracking
            xCellTmp=num2cell([x_new; s_new]);
            while (double(f(xCellTmp{:}))>(fun_val+alpha*tau*(gval')*[xDelta;sDelta]))
                tau = beta * tau; 
                x_new = xk(:,iter) + tau * xDelta;
                s_new = sval + tau * sDelta;
                xCellTmp = num2cell([x_new; s_new]);
            end
            xk(:,iter+1)=xk(:,iter)+tau*xDelta;
            sval=sval+tau*sDelta;
            iter=iter+1;
        end
        xFeas=xk(:,iter);
        if(all(xk(:,iter)>=0) && (any((A*xk(:,iter)-b)<threshold_1) || any((A*xk(:,iter)-b)>(0-threshold_1))))
            break;
        end
        xOuter(:,outer_iter+1)=xk(:,iter);   % xk: solution of optimization problem
        fprintf("1/t %f \n",1/t);
        if 1/t < threshold_2                     % Algorithm termination condition
            break; 
        end              
        outer_iter=outer_iter + 1;
        t=t*mu;
    end
end