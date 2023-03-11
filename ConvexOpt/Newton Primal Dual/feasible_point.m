function [xk]=feasible_point(A,b,c,x0,notChangeT)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% M-file that implements an interior-point algorithm for the           %
% minimization of a linear function with a linear inequality           % 
% constraint and positivity constraints                                %
%                                                                      %
% Objective function: c^T x                                            %
%                                                                      %
% Inequality constraint:  A^T x - b \le 0````                          %
%                         x_i >= 0, i=1, 2                             %
%                                                                      %
% Method of solution: Interior Point                                   %
%                                                                      %
% A. P. Liavas, Jan. 24, 2012                                          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialization 
threshold_1=10^-3;               
threshold_2=10^-6;              
alpha=0.25; beta = 0.5;       % backtracking parameters
h=10^(-6);                    % "small" parameter for numerical computation of grad, Hess
n=size(A,2);
for i=1:n
    if(x0(i)==0)
        x0(i)=h;
    end
end
mu=2;                         % parameter for increasing t 
t=1;                          % initialize parameter t of interior point methods
x=sym('x',[n 1]);
s=sym('s',[1 1]);
x_init=x0;                    % start from a feasible point
outer_iter = 1;
I=eye(n);
xk(:,outer_iter)=x_init;
sval=max(x_init)+1;
while(1)                                    % Outer loop    
     f=symfun(t.*s-sum(log(x+s)),[x;s]); %barrier cost function
     xval(:,1)=xk(:,outer_iter);            % Start new optimization from the previous solution
     inner_iter=1;
     while(1)                               % Inner loop
        xCell=num2cell(xval(:,inner_iter));
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
        fprintf("lambda:%f\n",l_x^2/2)
        if l_x^2/2 <= threshold_1          % Newton iterations termination condition
           break; 
        end    
         % !!! Check feasibility !!!
         tau = 1;
         x_new = xval(:,inner_iter) + tau * xDelta;
         while  (all(x_new<0) || any((A*x_new-b)>threshold_2))  
%          while  (all(x_new<0) || (any((A*x_new-b)>threshold_1) || any((A*x_new)<0-threshold_1)))
             tau = beta * tau; 
             x_new = xval(:,inner_iter) + tau * xDelta;
             return
         end 
         % !!!! x_new is FEASIBLE here !!!
        xCellTmp=num2cell([x_new; s_new]);
        while (double(f(xCellTmp{:}))>(fun_val+alpha*tau*(gval')*[xDelta;sDelta]))
            tau = beta * tau; 
            x_new = xk(:,iter) + tau * xDelta;
            s_new = sval + tau * sDelta;
            xCellTmp = num2cell([x_new; s_new]);
        end
         % Update x
         xval(:,inner_iter+1)=xval(:,inner_iter) + tau * xDelta;
         sval=sval+tau*sDelta;
         inner_iter=inner_iter + 1;
     end
     if(notChangeT)
         break;
     end
     xk(:,outer_iter+1)=xval(:,inner_iter);   % xk: solution of optimization problem
     fprintf("1/t %f \n",1/t);
     if 1/t < threshold_2                     % Algorithm termination condition
         break; 
     end              
     outer_iter=outer_iter + 1;
     t=t*mu;
end