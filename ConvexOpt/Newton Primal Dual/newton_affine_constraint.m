function [xk,fun_vals]=newton_affine_constraint(A,f,g,h,x0,alpha,beta,epsilon)
% Newton's method with backtracking for problems with affine constraints
%
% INPUT
%=======================================
% A ......... Ax=b contraint
% f ......... objective function
% g ......... gradient of the objective function
% h ......... hessian of the objective function
% x0......... initial point
% alpha ..... tolerance parameter for the stepsize selection strategy
% beta ...... the proportion in which the stepsize is multiplied
%             at each backtracking step (0<beta<1)
% epsilon ... tolerance parameter for stopping rule
% OUTPUT
%=======================================
% xk......... solution steps
% fun_vals... steps to optimal value

x=x0;
xCell=num2cell(x);
fun_val=double(f(xCell{:}));
gval=double(g(xCell{:}));
hval=double(h(xCell{:}));
iter=0;
fun_vals=(fun_val);
xk=xCell;
while (1)
    iter=iter+1;
    w=-inv(A*inv(hval)*A')*A*inv(hval)*gval;
    xDelta=-inv(hval)*(gval+A'*w);
    lambda=sqrt(xDelta'*hval*xDelta);
    if lambda^2<=epsilon
        return
    end
    t=1;
    xCellTmp=num2cell(x+t*xDelta);
    while(double(f(xCellTmp{:}))>fun_val+alpha*t*gval'*xDelta)
        %fprintf('\tBackTracking...\n'); 
        t=beta*t;
        xCellTmp=num2cell(x+t*xDelta);
    end
    x=x+t*xDelta;
    xCell = num2cell(x);
    fun_val=double(f(xCell{:}));
    fprintf('iter= %2d f(x)=%10.10f\n',iter,fun_val)
    fun_vals=[fun_vals fun_val];
    xk=[xk xCell];
    gval=double(g(xCell{:}));
    hval=double(h(xCell{:}));
end