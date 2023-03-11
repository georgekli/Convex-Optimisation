function [xk,rl,outer_iter,inner_iter,cumIters]=interior_point(A,b,c,x0)
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
t=1;                          % initialize parameter t of interior point methods
x=sym('x',[n 1]);
x_init=x0;                    % start from a feasible point
outer_iter = 1;
mu=2;
I=eye(n);
xk(:,outer_iter)=x_init;
rl=0;
cumIters=0;                   % cumulative number of Newton inner iterations
while(1)                                    % Outer loop    
     f=symfun(t.*(c'*x)-sum(log(x)),x); 
     xval(:,1)=xk(:,outer_iter);            % Start new optimization from the previous solution
     inner_iter=1;
     while(1)                               % Inner loop
         xCell=num2cell(xval(:,inner_iter));
         fun_val=double(f(xCell{:}));
         gval=t*c - ((1./xval(:,inner_iter))'*I)';
         for i=1:n
             if gval(i)==0
                gval(i)=h;
             end
         end
         hval_tmp=0;
         for i=1:n
            hval_tmp=hval_tmp+(1./(xval(i, inner_iter)^2))*I(:,i)*I(:,i)';
            if hval_tmp==0
                hval_tmp=h;
            end
         end
         hval=hval_tmp;
         w=-((A*(hval\(A')))\A)*(hval\gval);
         Dx_Nt=-(hval\(gval+A'*w));         % Newton step
         l_x=sqrt(Dx_Nt'*hval*Dx_Nt);       % Newton decrement
         rl=[rl 1/t];
         cumIters=cumIters+1;
         %fprintf("lambda:%f\n",l_x^2/2)
         if l_x^2/2 <= threshold_1          % Newton iterations termination condition
            break; 
         end    
         % !!! Check feasibility !!!
         tau = 1;
         x_new = xval(:,inner_iter) + tau * Dx_Nt;
         while  (all(x_new<0) || any((A*x_new-b)>threshold_2))  
%          while  (all(x_new<0) || (any((A*x_new-b)>threshold_1) || any((A*x_new)<0-threshold_1)))
             tau = beta * tau; 
             x_new = xval(:,inner_iter) + tau * Dx_Nt;
         end 
         % !!!! x_new is FEASIBLE here !!!
         % Backtracking
         xCellTmp=num2cell(x_new);
         while (double(f(xCellTmp{:}))>(fun_val+alpha*tau*(gval')*Dx_Nt))
              tau = beta * tau; 
              x_new = xval(:,inner_iter) + tau * Dx_Nt;
              xCellTmp = num2cell(x_new);
              %fprintf("Thres: %f\t", double(f(xCellTmp{:})))
              %fprintf("Backtracking with: %f\n", fun_val+alpha*tau*(gval')*Dx_Nt)
         end
         % Update x
         xval(:,inner_iter+1)=xval(:,inner_iter) + tau * Dx_Nt;
         inner_iter=inner_iter + 1;
     end
     xk(:,outer_iter+1)=xval(:,inner_iter);   % xk: solution of optimization problem
     fprintf("1/t %f \n",1/t);
     if 1/t < threshold_2                     % Algorithm termination condition
         break; 
     end              
     outer_iter=outer_iter + 1;
     t=t*mu;
end