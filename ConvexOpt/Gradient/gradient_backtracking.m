function [x,fun_val,fk]=gradient_backtracking(P,q,x0,s,alpha,beta,epsilon)
% Gradient method with backtracking stepsize rule
%
% INPUT
%=======================================
% P ....... the positive definite matrix
% q ....... a column vector associated with the linear part
% x0......... initial point
% s ......... initial choice of stepsize
% alpha ..... tolerance parameter for the stepsize selection
% beta ...... the constant in which the stepsize is multiplied 
%             at each backtracking step (0<beta<1)
% epsilon ... tolerance parameter for stopping rule
% OUTPUT
%=======================================
% x ......... optimal solution (up to a tolerance) 
%             of min f(x)
% fun_val ... optimal function value
x=x0;
grad=(P*x+q);
fun_val=(1/2)*x'*P*x+q'*x;
fk=[x(1) x(2) fun_val];
iter=0;
while (norm(grad)>epsilon)
    iter=iter+1;
    t=s;
    while (fun_val-((1/2)*(x-t*grad)'*P*(x-t*grad)+q'*(x-t*grad))<alpha*t*norm(grad)^2)
        fprintf('\tbacktracking with t=%f...\n',t);
        t=beta*t;
    end
    x=x-t*grad;
    fun_val=(1/2)*x'*P*x+q'*x;
    grad=(P*x+q);
    fk=[fk; x(1) x(2) fun_val];
    fprintf('iter_number = %3d norm_grad = %2.6f fun_val = %2.6f \n',iter,norm(grad),fun_val);
end
